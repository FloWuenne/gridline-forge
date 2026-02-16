//! Fast Marching Method for image inpainting

use crate::types::Result;
use ndarray::{Array2, ArrayViewMut2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Priority queue entry for Fast Marching Method
#[derive(Debug, Clone)]
struct FMMNode {
    x: usize,
    y: usize,
    distance: f32,
}

impl PartialEq for FMMNode {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for FMMNode {}

impl PartialOrd for FMMNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap (BinaryHeap is max-heap by default)
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for FMMNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Perform Fast Marching Method inpainting on a masked region
///
/// # Arguments
/// * `image` - Mutable view of the image layer to inpaint
/// * `mask` - Boolean mask where `true` indicates pixels to fill
/// * `radius` - Neighborhood radius for interpolation (default: 3)
pub fn fast_marching_inpaint(
    mut image: ArrayViewMut2<u16>,
    mask: &Array2<bool>,
    radius: usize,
) -> Result<()> {
    let (height, width) = (image.nrows(), image.ncols());

    // Compute distance map from each mask pixel to nearest known pixel
    let mut distance_map = Array2::from_elem((height, width), f32::MAX);
    let mut state = Array2::from_elem((height, width), PixelState::Unknown);

    // Initialize: mark known pixels and find boundary
    let mut heap = BinaryHeap::new();

    for y in 0..height {
        for x in 0..width {
            if !mask[(y, x)] {
                // Known pixel
                state[(y, x)] = PixelState::Known;
                distance_map[(y, x)] = 0.0;

                // Check if it's a boundary pixel (adjacent to mask)
                if is_boundary(x, y, width, height, mask) {
                    // Add boundary neighbors to heap
                    add_neighbors_to_heap(&mut heap, x, y, width, height, mask, &state);
                }
            }
        }
    }

    // Fast Marching: process pixels in order of distance from boundary
    while let Some(node) = heap.pop() {
        let (x, y) = (node.x, node.y);

        // Skip if already processed
        if state[(y, x)] == PixelState::Known {
            continue;
        }

        // Compute value by weighted interpolation of neighbors
        let value = interpolate_value(&image, &distance_map, x, y, radius);
        image[(y, x)] = value;

        // Mark as known
        state[(y, x)] = PixelState::Known;
        distance_map[(y, x)] = node.distance;

        // Add unfilled neighbors to heap
        add_neighbors_to_heap(&mut heap, x, y, width, height, mask, &state);
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PixelState {
    Unknown,
    Band,
    Known,
}

/// Check if a known pixel is on the boundary (adjacent to masked region)
fn is_boundary(x: usize, y: usize, width: usize, height: usize, mask: &Array2<bool>) -> bool {
    let neighbors = [
        (x.wrapping_sub(1), y),
        (x + 1, y),
        (x, y.wrapping_sub(1)),
        (x, y + 1),
    ];

    for (nx, ny) in neighbors {
        if nx < width && ny < height && mask[(ny, nx)] {
            return true;
        }
    }

    false
}

/// Add unfilled neighbors of a pixel to the priority queue
fn add_neighbors_to_heap(
    heap: &mut BinaryHeap<FMMNode>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mask: &Array2<bool>,
    state: &Array2<PixelState>,
) {
    let neighbors = [
        (x.wrapping_sub(1), y),
        (x + 1, y),
        (x, y.wrapping_sub(1)),
        (x, y + 1),
    ];

    for (nx, ny) in neighbors {
        if nx < width && ny < height && mask[(ny, nx)] && state[(ny, nx)] == PixelState::Unknown {
            // Compute distance to nearest known pixel (approximate as Euclidean)
            let distance = compute_distance(nx, ny, width, height, state);
            heap.push(FMMNode {
                x: nx,
                y: ny,
                distance,
            });
        }
    }
}

/// Compute approximate distance from a pixel to nearest known pixel
fn compute_distance(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    state: &Array2<PixelState>,
) -> f32 {
    let mut min_dist = f32::MAX;

    // Check immediate neighbors
    let neighbors = [
        (x.wrapping_sub(1), y),
        (x + 1, y),
        (x, y.wrapping_sub(1)),
        (x, y + 1),
    ];

    for (nx, ny) in neighbors {
        if nx < width && ny < height && state[(ny, nx)] == PixelState::Known {
            return 1.0; // Direct neighbor is known
        }
    }

    // Diagonal neighbors
    let diag_neighbors = [
        (x.wrapping_sub(1), y.wrapping_sub(1)),
        (x + 1, y.wrapping_sub(1)),
        (x.wrapping_sub(1), y + 1),
        (x + 1, y + 1),
    ];

    for (nx, ny) in diag_neighbors {
        if nx < width && ny < height && state[(ny, nx)] == PixelState::Known {
            min_dist = min_dist.min(1.414); // sqrt(2)
        }
    }

    if min_dist < f32::MAX {
        min_dist
    } else {
        2.0 // Estimate for farther pixels
    }
}

/// Interpolate pixel value using weighted average of neighbors
fn interpolate_value(
    image: &ArrayViewMut2<u16>,
    distance_map: &Array2<f32>,
    x: usize,
    y: usize,
    radius: usize,
) -> u16 {
    let (height, width) = (image.nrows(), image.ncols());
    let mut sum_value = 0.0;
    let mut sum_weight = 0.0;

    let x_start = x.saturating_sub(radius);
    let x_end = (x + radius + 1).min(width);
    let y_start = y.saturating_sub(radius);
    let y_end = (y + radius + 1).min(height);

    for ny in y_start..y_end {
        for nx in x_start..x_end {
            // Only use known pixels (distance < infinity)
            if distance_map[(ny, nx)] < f32::MAX {
                let dx = (nx as f32 - x as f32).abs();
                let dy = (ny as f32 - y as f32).abs();
                let dist = (dx * dx + dy * dy).sqrt();

                // Gaussian-like weight: closer pixels have more influence
                let weight = (-dist * dist / (2.0 * (radius as f32 / 2.0).powi(2))).exp();

                // Gradient-aware weight: prefer pixels with similar gradient direction
                // (simplified version - in full implementation would compute gradients)
                let value = image[(ny, nx)] as f32;

                sum_value += value * weight;
                sum_weight += weight;
            }
        }
    }

    if sum_weight > 0.0 {
        (sum_value / sum_weight).round() as u16
    } else {
        // Fallback: use any non-zero neighbor
        for ny in y_start..y_end {
            for nx in x_start..x_end {
                if image[(ny, nx)] > 0 {
                    return image[(ny, nx)];
                }
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_fmm_simple() {
        // Create a simple 5x5 image with center pixel as grid
        let mut image = Array2::from_elem((5, 5), 100u16);
        image[(2, 2)] = 0;

        let mut mask = Array2::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        let result = fast_marching_inpaint(image.view_mut(), &mask, 2);
        assert!(result.is_ok());

        // Center should now be filled with a value close to neighbors
        let filled = image[(2, 2)];
        assert!(filled > 0);
        assert!(filled <= 100);
    }

    #[test]
    fn test_is_boundary() {
        let mut mask = Array2::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        // Adjacent pixels should be boundary
        assert!(is_boundary(1, 2, 5, 5, &mask));
        assert!(is_boundary(3, 2, 5, 5, &mask));
        assert!(is_boundary(2, 1, 5, 5, &mask));
        assert!(is_boundary(2, 3, 5, 5, &mask));

        // Corner pixel should not be boundary
        assert!(!is_boundary(0, 0, 5, 5, &mask));
    }
}
