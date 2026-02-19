//! Parallelization helpers for processing multiple layers

use crate::types::{GridPattern, ProcessingConfig, Result};
use ndarray::{Array2, ArrayViewMut2, Axis};

/// Process a single layer: create mask, init grid pixels, and refine via Gaussian blur
///
/// Matches MindaGap's algorithm:
/// 1. Mask zero-valued pixels
/// 2. Initialize grid pixels to min non-zero value
/// 3. Iterative Gaussian blur (full image), copying back only grid pixels
pub fn process_single_layer(
    mut layer: ArrayViewMut2<u16>,
    _pattern: &GridPattern,
    config: &ProcessingConfig,
) -> Result<()> {
    let (height, width) = (layer.nrows(), layer.ncols());

    // Create mask from actual zero-valued pixels
    let mut mask = Array2::from_elem((height, width), false);
    let mut min_nonzero: u16 = u16::MAX;

    for y in 0..height {
        for x in 0..width {
            if layer[(y, x)] == 0 {
                mask[(y, x)] = true;
            } else {
                min_nonzero = min_nonzero.min(layer[(y, x)]);
            }
        }
    }

    // Fallback if all pixels are zero
    if min_nonzero == u16::MAX {
        min_nonzero = 1;
    }

    // Initialize grid pixels to min non-zero value (matches MindaGap: im_copy[grid_coords] = min(img_array[img_array > 0]))
    for y in 0..height {
        for x in 0..width {
            if mask[(y, x)] {
                layer[(y, x)] = min_nonzero;
            }
        }
    }

    // Iterative Gaussian blur refinement
    crate::refinement::refine_grid(
        layer,
        &mask,
        config.kernel_size,
        config.refinement_rounds,
    )?;

    Ok(())
}

/// Process multiple layers in parallel
pub fn process_layers_parallel(
    layers: &mut ndarray::Array3<u16>,
    pattern: &GridPattern,
    config: &ProcessingConfig,
) -> Result<()> {
    let num_layers = layers.shape()[0];

    // For now, process sequentially until we can figure out parallel mutable iteration
    // (ndarray's mutable axis iterators don't work well with rayon)
    for (idx, mut layer) in layers.axis_iter_mut(Axis(0)).enumerate() {
        if config.show_progress {
            eprintln!("Processing layer {}/{}", idx + 1, num_layers);
        }
        process_single_layer(layer.view_mut(), pattern, config)?;
    }

    Ok(())
}

/// Create grid mask for a given pattern
pub fn create_grid_mask(
    width: u32,
    height: u32,
    pattern: &GridPattern,
) -> Array2<bool> {
    let mut mask = Array2::from_elem((height as usize, width as usize), false);

    for y in 0..height {
        for x in 0..width {
            if pattern.is_grid_pixel(x, y) {
                mask[(y as usize, x as usize)] = true;
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GridPattern;

    #[test]
    fn test_create_grid_mask() {
        let pattern = GridPattern {
            x_spacing: 10,
            y_spacing: 10,
            x_offset: 0,
            y_offset: 0,
            grid_width: 1,
            confidence: 1.0,
        };

        let mask = create_grid_mask(20, 20, &pattern);

        // Check that grid positions are marked
        assert!(mask[(0, 0)]); // Vertical line at x=0
        assert!(mask[(0, 10)]); // Vertical line at x=10
        assert!(mask[(10, 0)]); // Horizontal line at y=10
        assert!(!mask[(5, 5)]); // Non-grid position
    }
}
