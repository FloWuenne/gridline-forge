//! Gaussian blur refinement for grid regions
//!
//! Implements MindaGap's iterative Gaussian blur algorithm:
//! 1. Initialize grid pixels to min non-zero value
//! 2. Loop N times: blur entire image, copy back only grid pixels
//!
//! Uses separable convolution with OpenCV-matching sigma for performance.

use crate::types::Result;
use ndarray::{Array2, ArrayViewMut2};
use rayon::prelude::*;

/// Refine grid regions using MindaGap-style iterative Gaussian blur
///
/// Blurs the full image each iteration (matching MindaGap's cv2.GaussianBlur behavior),
/// then copies blurred values back only to grid pixels.
///
/// # Arguments
/// * `image` - Mutable view of the image layer to refine (grid pixels should already be
///   initialized to min_nonzero before calling)
/// * `mask` - Boolean mask of grid pixels (true = grid)
/// * `kernel_size` - Size of Gaussian kernel (must be odd, default 5)
/// * `iterations` - Number of refinement iterations (default 40)
pub fn refine_grid(
    mut image: ArrayViewMut2<u16>,
    mask: &Array2<bool>,
    kernel_size: usize,
    iterations: usize,
) -> Result<()> {
    if kernel_size % 2 == 0 {
        return Err(crate::types::Error::Config(
            "Kernel size must be odd".to_string(),
        ));
    }

    let (height, width) = (image.nrows(), image.ncols());
    let kernel = generate_gaussian_kernel_1d(kernel_size);
    let half_k = kernel_size / 2;

    // Build flat mask — use contiguous slice if available, else indexed
    let flat_mask: Vec<bool> = if let Some(slice) = mask.as_slice() {
        slice.to_vec()
    } else {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| mask[(y, x)]))
            .collect()
    };

    // Pre-compute grid pixel indices (sparse — ~0.1% of image)
    let grid_indices: Vec<usize> = flat_mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(i, _)| i)
        .collect();

    // Pre-compute horizontal influence ranges per row.
    // temp[y][x] depends on work[y][x-half_k..x+half_k+1]. Since only grid pixels
    // in work change between iterations, temp[y][x] needs recomputing only if
    // work[y] has a grid pixel within half_k columns of x.
    // Derived from the actual mask, so any grid line width is handled automatically.
    let h_influence_ranges = compute_influence_ranges(&flat_mask, height, width, half_k);

    // Work buffer: copy image as f32
    let mut work: Vec<f32> = if let Some(slice) = image.as_slice() {
        slice.iter().map(|&v| v as f32).collect()
    } else {
        let mut buf = Vec::with_capacity(height * width);
        for y in 0..height {
            for x in 0..width {
                buf.push(image[(y, x)] as f32);
            }
        }
        buf
    };

    // Temp buffer for horizontal pass results
    let mut temp: Vec<f32> = vec![0.0; height * width];

    // Pre-allocated buffer for vertical pass results (avoids allocation per iteration)
    let mut blurred = vec![0.0f32; grid_indices.len()];

    // Initial full horizontal blur — computes temp for ALL positions.
    // "Static" positions (not influenced by grid pixels) won't change in subsequent
    // iterations, so this one-time computation remains valid throughout the loop.
    horizontal_blur_full(&work, &mut temp, width, &kernel, half_k);

    for iter in 0..iterations {
        // Vertical pass: compute blur only at grid pixel positions
        grid_indices
            .par_iter()
            .zip(blurred.par_iter_mut())
            .for_each(|(&idx, out)| {
                let y = idx / width;
                let x = idx % width;
                let mut sum = 0.0f32;
                for (i, &k_val) in kernel.iter().enumerate() {
                    let sy = (y + i).saturating_sub(half_k).min(height - 1);
                    sum += temp[sy * width + x] * k_val;
                }
                // Round to integer each iteration to match MindaGap's uint16 behavior
                *out = sum.round();
            });

        // Scatter blurred values to work at grid pixel positions
        for (i, &idx) in grid_indices.iter().enumerate() {
            work[idx] = blurred[i];
        }

        // Selective horizontal pass: only recompute temp at positions whose
        // work inputs changed (positions near grid pixels in each row).
        // Skip on the last iteration — temp won't be read again.
        if iter < iterations - 1 {
            horizontal_blur_selective(
                &work,
                &mut temp,
                width,
                &kernel,
                half_k,
                &h_influence_ranges,
            );
        }
    }

    // Copy results back to image (grid pixels only)
    for &idx in &grid_indices {
        let y = idx / width;
        let x = idx % width;
        image[(y, x)] = work[idx] as u16;
    }

    Ok(())
}

/// Full horizontal blur: work -> temp for all pixels.
/// Uses edge/interior split to enable SIMD auto-vectorization in the interior.
fn horizontal_blur_full(
    work: &[f32],
    temp: &mut [f32],
    width: usize,
    kernel: &[f32],
    half_k: usize,
) {
    let kernel_size = kernel.len();
    temp.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, temp_row)| {
            let row_offset = y * width;
            let work_row = &work[row_offset..row_offset + width];

            // Left edge (clamped boundary)
            for x in 0..half_k.min(width) {
                let mut sum = 0.0f32;
                for (i, &k_val) in kernel.iter().enumerate() {
                    let sx = (x + i).saturating_sub(half_k).min(width - 1);
                    sum += work_row[sx] * k_val;
                }
                temp_row[x] = sum;
            }

            // Interior (no bounds checks — enables SIMD auto-vectorization)
            if width > kernel_size {
                for x in half_k..(width - half_k) {
                    let mut sum = 0.0f32;
                    let base = x - half_k;
                    for (i, &k_val) in kernel.iter().enumerate() {
                        sum += work_row[base + i] * k_val;
                    }
                    temp_row[x] = sum;
                }
            }

            // Right edge (clamped boundary)
            for x in width.saturating_sub(half_k)..width {
                let mut sum = 0.0f32;
                for (i, &k_val) in kernel.iter().enumerate() {
                    let sx = (x + i).saturating_sub(half_k).min(width - 1);
                    sum += work_row[sx] * k_val;
                }
                temp_row[x] = sum;
            }
        });
    // Sync point: par_chunks_mut guarantees all rows are complete before returning.
    // This ensures temp is fully populated and the correctness invariant holds:
    // for rows with no influence ranges (no grid pixels nearby), these temp values
    // are now "static" — they depend only on non-grid work values which never change,
    // so they remain valid for all subsequent iterations without recomputation.
}

/// Selective horizontal blur: only recompute temp at column ranges that are
/// influenced by grid pixels. Each row's ranges were pre-computed from the mask.
fn horizontal_blur_selective(
    work: &[f32],
    temp: &mut [f32],
    width: usize,
    kernel: &[f32],
    half_k: usize,
    influence_ranges: &[Vec<(usize, usize)>],
) {
    temp.par_chunks_mut(width)
        .zip(influence_ranges.par_iter())
        .enumerate()
        .for_each(|(y, (temp_row, ranges))| {
            if ranges.is_empty() {
                return;
            }
            let row_offset = y * width;
            let work_row = &work[row_offset..row_offset + width];

            for &(start, end) in ranges.iter() {
                for x in start..end {
                    let mut sum = 0.0f32;
                    for (i, &k_val) in kernel.iter().enumerate() {
                        let sx = (x + i).saturating_sub(half_k).min(width - 1);
                        sum += work_row[sx] * k_val;
                    }
                    temp_row[x] = sum;
                }
            }
        });
}

/// Compute per-row column ranges that need horizontal blur recomputation.
///
/// A column x in row y is "influenced" if any grid pixel exists within
/// half_k columns of x in the same row. This is equivalent to horizontally
/// dilating the mask by half_k. Adjacent/overlapping ranges are merged.
///
/// Derived from the actual zero-pixel mask, so any grid line width is
/// handled automatically — wider grid lines simply produce wider ranges.
fn compute_influence_ranges(
    flat_mask: &[bool],
    height: usize,
    width: usize,
    half_k: usize,
) -> Vec<Vec<(usize, usize)>> {
    (0..height)
        .map(|y| {
            let row_offset = y * width;
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let mut x = 0;
            while x < width {
                if flat_mask[row_offset + x] {
                    // Found a grid pixel — expand influence by half_k in both directions
                    let start = x.saturating_sub(half_k);
                    // Skip consecutive grid pixels (handles grid_width > 1)
                    let mut end_grid = x;
                    while end_grid + 1 < width && flat_mask[row_offset + end_grid + 1] {
                        end_grid += 1;
                    }
                    let end = (end_grid + half_k + 1).min(width);

                    // Merge with previous range if overlapping
                    if let Some(last) = ranges.last_mut() {
                        if start <= last.1 {
                            last.1 = last.1.max(end);
                        } else {
                            ranges.push((start, end));
                        }
                    } else {
                        ranges.push((start, end));
                    }
                    x = end_grid + 1;
                } else {
                    x += 1;
                }
            }
            ranges
        })
        .collect()
}

/// Generate 1D Gaussian kernel matching OpenCV's getGaussianKernel behavior
///
/// For ksize <= 7 with sigma=0, OpenCV uses fixed binomial kernels (bit-exact).
/// For larger sizes, it computes Gaussian with auto-sigma:
///   sigma = 0.3 * ((ksize-1)*0.5 - 1) + 0.8
fn generate_gaussian_kernel_1d(size: usize) -> Vec<f32> {
    // Match OpenCV's fixed kernels for small sizes (getGaussianKernelBitExact)
    // These are binomial approximation kernels used when sigma <= 0
    match size {
        1 => return vec![1.0],
        3 => return vec![0.25, 0.5, 0.25],
        5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
        7 => return vec![0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
        _ => {}
    }

    // For larger sizes, compute Gaussian with OpenCV's auto-sigma
    let sigma = 0.3 * ((size as f32 - 1.0) * 0.5 - 1.0) + 0.8;
    let center = (size / 2) as f32;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0;

    for i in 0..size {
        let x = i as f32 - center;
        let value = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }

    for k in &mut kernel {
        *k /= sum;
    }

    kernel
}

/// Expand mask by dilating it (add margin around grid pixels)
#[allow(dead_code)]
fn expand_mask(mask: &Array2<bool>, margin: usize) -> Array2<bool> {
    let (height, width) = (mask.nrows(), mask.ncols());
    let mut expanded = Array2::from_elem((height, width), false);

    for y in 0..height {
        for x in 0..width {
            if mask[(y, x)] {
                let y_start = y.saturating_sub(margin);
                let y_end = (y + margin + 1).min(height);
                let x_start = x.saturating_sub(margin);
                let x_end = (x + margin + 1).min(width);

                for ny in y_start..y_end {
                    for nx in x_start..x_end {
                        expanded[(ny, nx)] = true;
                    }
                }
            }
        }
    }

    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_gaussian_kernel() {
        let kernel = generate_gaussian_kernel_1d(5);
        assert_eq!(kernel.len(), 5);

        // Check normalization
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Check symmetry
        assert!((kernel[0] - kernel[4]).abs() < 0.001);
        assert!((kernel[1] - kernel[3]).abs() < 0.001);

        // Center should be largest
        assert!(kernel[2] > kernel[1]);
        assert!(kernel[2] > kernel[0]);
    }

    #[test]
    fn test_opencv_fixed_kernel() {
        // For ksize=5, OpenCV uses fixed binomial kernel [1/16, 4/16, 6/16, 4/16, 1/16]
        let kernel = generate_gaussian_kernel_1d(5);
        assert_eq!(kernel, vec![0.0625, 0.25, 0.375, 0.25, 0.0625]);

        // For ksize=3, OpenCV uses [1/4, 2/4, 1/4]
        let kernel3 = generate_gaussian_kernel_1d(3);
        assert_eq!(kernel3, vec![0.25, 0.5, 0.25]);
    }

    #[test]
    fn test_expand_mask() {
        let mut mask = Array2::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        let expanded = expand_mask(&mask, 1);

        assert!(expanded[(2, 2)]);
        assert!(expanded[(1, 2)]);
        assert!(expanded[(3, 2)]);
        assert!(expanded[(2, 1)]);
        assert!(expanded[(2, 3)]);
        assert!(!expanded[(0, 0)]);
    }

    #[test]
    fn test_refine_grid() {
        let mut image = Array2::from_elem((5, 5), 100u16);
        image[(2, 2)] = 50; // Pre-initialized grid pixel (min_nonzero)

        let mut mask = Array2::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        let result = refine_grid(image.view_mut(), &mask, 3, 2);
        assert!(result.is_ok());

        // Grid pixel should converge toward surrounding values
        let refined = image[(2, 2)];
        assert!(refined > 50, "Should increase from init value: {}", refined);
    }
}
