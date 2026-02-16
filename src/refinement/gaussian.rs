//! Gaussian blur refinement for grid regions

use crate::types::Result;
use ndarray::{Array2, ArrayViewMut2};

/// Refine grid regions using localized Gaussian blur
///
/// # Arguments
/// * `image` - Mutable view of the image layer to refine
/// * `mask` - Boolean mask of grid pixels (true = grid)
/// * `kernel_size` - Size of Gaussian kernel (must be odd)
/// * `iterations` - Number of refinement iterations
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

    // Generate Gaussian kernel
    let kernel = generate_gaussian_kernel_1d(kernel_size);

    // Expand mask to include small margin around grid pixels
    let expanded_mask = expand_mask(mask, 2);

    // Create work buffer for separable convolution
    let (height, width) = (image.nrows(), image.ncols());
    let mut temp_buffer = Array2::zeros((height, width));

    // Iterative refinement
    for _ in 0..iterations {
        // Apply separable Gaussian blur

        // Horizontal pass
        apply_horizontal_blur(&image.view(), &mut temp_buffer.view_mut(), &kernel, &expanded_mask);

        // Vertical pass (back to image, but only update grid pixels)
        apply_vertical_blur(&temp_buffer.view(), &mut image, &kernel, mask);
    }

    Ok(())
}

/// Generate 1D Gaussian kernel
fn generate_gaussian_kernel_1d(size: usize) -> Vec<f32> {
    let sigma = (size as f32) / 6.0; // Standard rule: kernel_size ≈ 6σ
    let center = (size / 2) as f32;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0;

    for i in 0..size {
        let x = i as f32 - center;
        let value = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }

    // Normalize kernel
    for k in &mut kernel {
        *k /= sum;
    }

    kernel
}

/// Expand mask by dilating it (add margin around grid pixels)
fn expand_mask(mask: &Array2<bool>, margin: usize) -> Array2<bool> {
    let (height, width) = (mask.nrows(), mask.ncols());
    let mut expanded = Array2::from_elem((height, width), false);

    for y in 0..height {
        for x in 0..width {
            if mask[(y, x)] {
                // Set pixel and surrounding margin
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

/// Apply horizontal Gaussian blur (only to expanded mask region)
fn apply_horizontal_blur(
    input: &ndarray::ArrayView2<u16>,
    output: &mut ArrayViewMut2<u16>,
    kernel: &[f32],
    mask: &Array2<bool>,
) {
    let (height, width) = (input.nrows(), input.ncols());
    let half_kernel = kernel.len() / 2;

    for y in 0..height {
        for x in 0..width {
            // Only process pixels in expanded mask
            if !mask[(y, x)] {
                output[(y, x)] = input[(y, x)];
                continue;
            }

            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, &k_val) in kernel.iter().enumerate() {
                let offset = i as isize - half_kernel as isize;
                let nx = (x as isize + offset).max(0).min(width as isize - 1) as usize;

                sum += input[(y, nx)] as f32 * k_val;
                weight_sum += k_val;
            }

            output[(y, x)] = if weight_sum > 0.0 {
                (sum / weight_sum).round() as u16
            } else {
                input[(y, x)]
            };
        }
    }
}

/// Apply vertical Gaussian blur (only update grid mask pixels in output)
fn apply_vertical_blur(
    input: &ndarray::ArrayView2<u16>,
    output: &mut ArrayViewMut2<u16>,
    kernel: &[f32],
    mask: &Array2<bool>,
) {
    let (height, width) = (input.nrows(), input.ncols());
    let half_kernel = kernel.len() / 2;

    for y in 0..height {
        for x in 0..width {
            // Only update grid pixels in final output
            if !mask[(y, x)] {
                continue;
            }

            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, &k_val) in kernel.iter().enumerate() {
                let offset = i as isize - half_kernel as isize;
                let ny = (y as isize + offset).max(0).min(height as isize - 1) as usize;

                sum += input[(ny, x)] as f32 * k_val;
                weight_sum += k_val;
            }

            output[(y, x)] = if weight_sum > 0.0 {
                (sum / weight_sum).round() as u16
            } else {
                input[(y, x)]
            };
        }
    }
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
    fn test_expand_mask() {
        let mut mask = Array2::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        let expanded = expand_mask(&mask, 1);

        // Center and immediate neighbors should be true
        assert!(expanded[(2, 2)]);
        assert!(expanded[(1, 2)]);
        assert!(expanded[(3, 2)]);
        assert!(expanded[(2, 1)]);
        assert!(expanded[(2, 3)]);

        // Corners should still be false
        assert!(!expanded[(0, 0)]);
    }

    #[test]
    fn test_refine_grid() {
        let mut image = Array2::from_elem((5, 5), 100u16);
        image[(2, 2)] = 0; // Grid pixel

        let mut mask = Array2::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        let result = refine_grid(image.view_mut(), &mask, 3, 2);
        assert!(result.is_ok());

        // Grid pixel should now have a smoothed value
        let refined = image[(2, 2)];
        assert!(refined > 0);
        assert!(refined < 100);
    }
}
