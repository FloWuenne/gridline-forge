//! Gaussian blur refinement for grid regions
//!
//! Implements MindaGap's iterative Gaussian blur algorithm:
//! 1. Initialize grid pixels to min non-zero value
//! 2. Loop N times: blur entire image, copy back only grid pixels
//!
//! Uses separable convolution with OpenCV-matching sigma for performance.

use crate::types::Result;
use ndarray::{Array2, ArrayViewMut2};

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

    // Build flat mask for fast indexing
    let flat_mask: Vec<bool> = (0..height)
        .flat_map(|y| (0..width).map(move |x| mask[(y, x)]))
        .collect();

    // Work buffer: copy image as f32
    let mut work: Vec<f32> = Vec::with_capacity(height * width);
    for y in 0..height {
        for x in 0..width {
            work.push(image[(y, x)] as f32);
        }
    }

    // Temp buffer for horizontal pass results
    let mut temp: Vec<f32> = vec![0.0; height * width];

    for _ in 0..iterations {
        // Horizontal pass: blur work -> temp (all pixels)
        for y in 0..height {
            let row_offset = y * width;
            for x in 0..width {
                let mut sum = 0.0f32;
                for (i, &k_val) in kernel.iter().enumerate() {
                    let sx = (x as isize + i as isize - half_k as isize)
                        .max(0)
                        .min(width as isize - 1) as usize;
                    sum += work[row_offset + sx] * k_val;
                }
                temp[row_offset + x] = sum;
            }
        }

        // Vertical pass: blur temp -> write to work, but only grid pixels
        // Non-grid pixels in work stay unchanged (they are the original image values)
        for y in 0..height {
            let row_offset = y * width;
            for x in 0..width {
                if !flat_mask[row_offset + x] {
                    continue;
                }

                let mut sum = 0.0f32;
                for (i, &k_val) in kernel.iter().enumerate() {
                    let sy = (y as isize + i as isize - half_k as isize)
                        .max(0)
                        .min(height as isize - 1) as usize;
                    sum += temp[sy * width + x] * k_val;
                }
                // Round to integer each iteration to match MindaGap's uint16 behavior
                // (cv2.GaussianBlur on uint16 returns uint16, so rounding happens each iteration)
                work[row_offset + x] = sum.round();
            }
        }
    }

    // Copy results back to image (grid pixels only)
    for y in 0..height {
        let row_offset = y * width;
        for x in 0..width {
            if flat_mask[row_offset + x] {
                image[(y, x)] = work[row_offset + x].round() as u16;
            }
        }
    }

    Ok(())
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
