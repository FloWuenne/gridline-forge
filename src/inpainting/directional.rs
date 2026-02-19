//! Directional interpolation for axis-aligned grid lines.
//!
//! Fills zero-valued grid pixels by linearly interpolating from the nearest
//! non-zero pixels perpendicular to the grid line direction. Single-pass,
//! much faster than iterative Gaussian diffusion.

use crate::types::Result;
use ndarray::Array2;

/// Fill grid pixels using directional linear interpolation.
///
/// For each masked pixel:
/// - On a vertical grid line → interpolate horizontally (left/right)
/// - On a horizontal grid line → interpolate vertically (above/below)
/// - At intersection → average of horizontal and vertical interpolations
/// - Isolated zeros → local neighborhood average
pub fn directional_interpolate(image: &mut Array2<u16>, mask: &Array2<bool>) -> Result<()> {
    let fallback = compute_min_nonzero(image);
    let col_is_grid = classify_grid_columns(mask);
    let row_is_grid = classify_grid_rows(mask);

    // Initialize all masked pixels to fallback so edge grid lines
    // (where only one boundary exists) get a reasonable value.
    let (height, width) = (image.nrows(), image.ncols());
    for y in 0..height {
        for x in 0..width {
            if mask[(y, x)] {
                image[(y, x)] = fallback;
            }
        }
    }

    // Pass 1: interpolate across vertical grid lines (scan each row horizontally)
    interpolate_rows(image, mask, &col_is_grid);

    // Pass 2: interpolate across horizontal grid lines (scan each column vertically)
    interpolate_cols(image, mask, &row_is_grid);

    // Pass 3: fix intersections by averaging both directions
    fill_intersections(image, mask, &row_is_grid, &col_is_grid, fallback);

    Ok(())
}

/// Compute the minimum non-zero pixel value in the image.
/// Used as fallback for edge grid lines where only one boundary exists.
fn compute_min_nonzero(image: &Array2<u16>) -> u16 {
    let mut min_val = u16::MAX;
    for &v in image.iter() {
        if v > 0 && v < min_val {
            min_val = v;
        }
    }
    if min_val == u16::MAX { 1 } else { min_val }
}

/// Classify which columns are vertical grid lines (>50% of pixels masked).
fn classify_grid_columns(mask: &Array2<bool>) -> Vec<bool> {
    let (height, width) = (mask.nrows(), mask.ncols());
    let threshold = height / 2;
    (0..width)
        .map(|x| {
            let count = (0..height).filter(|&y| mask[(y, x)]).count();
            count > threshold
        })
        .collect()
}

/// Classify which rows are horizontal grid lines (>50% of pixels masked).
fn classify_grid_rows(mask: &Array2<bool>) -> Vec<bool> {
    let (height, width) = (mask.nrows(), mask.ncols());
    let threshold = width / 2;
    (0..height)
        .map(|y| {
            let count = (0..width).filter(|&x| mask[(y, x)]).count();
            count > threshold
        })
        .collect()
}

/// Interpolate across vertical grid lines by scanning each row horizontally.
/// For each masked pixel in a grid column, find nearest non-zero pixel left and right,
/// then linearly interpolate by inverse distance.
fn interpolate_rows(image: &mut Array2<u16>, mask: &Array2<bool>, col_is_grid: &[bool]) {
    let (height, width) = (image.nrows(), image.ncols());

    for y in 0..height {
        let mut x = 0;
        while x < width {
            // Skip non-grid columns
            if !mask[(y, x)] || !col_is_grid[x] {
                x += 1;
                continue;
            }

            // Find the start of a contiguous masked span in grid columns
            let span_start = x;
            while x < width && mask[(y, x)] && col_is_grid[x] {
                x += 1;
            }
            let span_end = x; // exclusive

            // Find left boundary (nearest non-zero pixel before span)
            let left_val = if span_start > 0 {
                Some(image[(y, span_start - 1)])
            } else {
                None
            };

            // Find right boundary (nearest non-zero pixel after span)
            let right_val = if span_end < width {
                Some(image[(y, span_end)])
            } else {
                None
            };

            // Interpolate the span
            let span_len = span_end - span_start;
            for i in 0..span_len {
                let val = match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        // Linear interpolation: weight by inverse distance
                        let t = (i as f32 + 1.0) / (span_len as f32 + 1.0);
                        (l as f32 * (1.0 - t) + r as f32 * t).round() as u16
                    }
                    (Some(l), None) => l,
                    (None, Some(r)) => r,
                    (None, None) => image[(y, span_start + i)], // keep fallback
                };
                image[(y, span_start + i)] = val;
            }
        }
    }
}

/// Interpolate across horizontal grid lines by scanning each column vertically.
/// Extracts column to temp vec for cache friendliness, processes, writes back.
fn interpolate_cols(image: &mut Array2<u16>, mask: &Array2<bool>, row_is_grid: &[bool]) {
    let (height, width) = (image.nrows(), image.ncols());
    let mut col_buf = vec![0u16; height];
    let mut col_mask = vec![false; height];

    for x in 0..width {
        // Extract column
        for y in 0..height {
            col_buf[y] = image[(y, x)];
            col_mask[y] = mask[(y, x)] && row_is_grid[y];
        }

        let mut y = 0;
        while y < height {
            if !col_mask[y] {
                y += 1;
                continue;
            }

            let span_start = y;
            while y < height && col_mask[y] {
                y += 1;
            }
            let span_end = y;

            let top_val = if span_start > 0 {
                Some(col_buf[span_start - 1])
            } else {
                None
            };

            let bottom_val = if span_end < height {
                Some(col_buf[span_end])
            } else {
                None
            };

            let span_len = span_end - span_start;
            for i in 0..span_len {
                let val = match (top_val, bottom_val) {
                    (Some(t), Some(b)) => {
                        let frac = (i as f32 + 1.0) / (span_len as f32 + 1.0);
                        (t as f32 * (1.0 - frac) + b as f32 * frac).round() as u16
                    }
                    (Some(t), None) => t,
                    (None, Some(b)) => b,
                    (None, None) => col_buf[span_start + i], // keep fallback
                };
                col_buf[span_start + i] = val;
            }
        }

        // Write column back
        for y in 0..height {
            if mask[(y, x)] && row_is_grid[y] {
                image[(y, x)] = col_buf[y];
            }
        }
    }
}

/// Fill intersection pixels (both row and column are grid lines).
/// Scans both directions and averages the two interpolated values.
fn fill_intersections(
    image: &mut Array2<u16>,
    mask: &Array2<bool>,
    row_is_grid: &[bool],
    col_is_grid: &[bool],
    fallback: u16,
) {
    let (height, width) = (image.nrows(), image.ncols());

    for y in 0..height {
        if !row_is_grid[y] {
            continue;
        }
        for x in 0..width {
            if !col_is_grid[x] || !mask[(y, x)] {
                continue;
            }

            // Horizontal interpolation: find left and right non-grid-column pixels
            let h_val = {
                let mut left = None;
                for lx in (0..x).rev() {
                    if !col_is_grid[lx] {
                        left = Some(image[(y, lx)]);
                        break;
                    }
                }
                let mut right = None;
                for rx in (x + 1)..width {
                    if !col_is_grid[rx] {
                        right = Some(image[(y, rx)]);
                        break;
                    }
                }
                match (left, right) {
                    (Some(l), Some(r)) => {
                        let ld = (0..x).rev().take_while(|&lx| col_is_grid[lx]).count() as f32 + 1.0;
                        let rd = (x + 1..width).take_while(|&rx| col_is_grid[rx]).count() as f32 + 1.0;
                        let t = ld / (ld + rd);
                        (l as f32 * (1.0 - t) + r as f32 * t).round() as u16
                    }
                    (Some(l), None) => l,
                    (None, Some(r)) => r,
                    (None, None) => fallback,
                }
            };

            // Vertical interpolation: find top and bottom non-grid-row pixels
            let v_val = {
                let mut top = None;
                for ly in (0..y).rev() {
                    if !row_is_grid[ly] {
                        top = Some(image[(ly, x)]);
                        break;
                    }
                }
                let mut bottom = None;
                for ry in (y + 1)..height {
                    if !row_is_grid[ry] {
                        bottom = Some(image[(ry, x)]);
                        break;
                    }
                }
                match (top, bottom) {
                    (Some(t), Some(b)) => {
                        let td = (0..y).rev().take_while(|&ly| row_is_grid[ly]).count() as f32 + 1.0;
                        let bd = (y + 1..height).take_while(|&ry| row_is_grid[ry]).count() as f32 + 1.0;
                        let frac = td / (td + bd);
                        (t as f32 * (1.0 - frac) + b as f32 * frac).round() as u16
                    }
                    (Some(t), None) => t,
                    (None, Some(b)) => b,
                    (None, None) => fallback,
                }
            };

            // Average horizontal and vertical
            image[(y, x)] = ((h_val as u32 + v_val as u32) / 2) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertical_grid_gap() {
        // 5x5 image with a vertical grid line at column 2
        let mut image = Array2::from_elem((5, 5), 100u16);
        for y in 0..5 {
            image[(y, 2)] = 0;
        }
        let mut mask = Array2::from_elem((5, 5), false);
        for y in 0..5 {
            mask[(y, 2)] = true;
        }

        directional_interpolate(&mut image, &mask).unwrap();

        // Column 2 should be interpolated from columns 1 (100) and 3 (100)
        for y in 0..5 {
            assert_eq!(image[(y, 2)], 100, "row {y} should be 100");
        }
    }

    #[test]
    fn test_horizontal_grid_gap() {
        // 5x5 image with a horizontal grid line at row 2
        let mut image = Array2::from_elem((5, 5), 200u16);
        for x in 0..5 {
            image[(2, x)] = 0;
        }
        let mut mask = Array2::from_elem((5, 5), false);
        for x in 0..5 {
            mask[(2, x)] = true;
        }

        directional_interpolate(&mut image, &mask).unwrap();

        for x in 0..5 {
            assert_eq!(image[(2, x)], 200, "col {x} should be 200");
        }
    }

    #[test]
    fn test_intersection() {
        // 7x7 image with vertical grid at col 3 and horizontal grid at row 3
        let mut image = Array2::from_elem((7, 7), 100u16);
        for y in 0..7 {
            image[(y, 3)] = 0;
        }
        for x in 0..7 {
            image[(3, x)] = 0;
        }
        let mut mask = Array2::from_elem((7, 7), false);
        for y in 0..7 {
            mask[(y, 3)] = true;
        }
        for x in 0..7 {
            mask[(3, x)] = true;
        }

        directional_interpolate(&mut image, &mask).unwrap();

        // Intersection at (3,3) should be ~100 (avg of h and v interpolations)
        assert!(
            image[(3, 3)] > 90 && image[(3, 3)] < 110,
            "intersection value {} should be near 100",
            image[(3, 3)]
        );
        // Non-intersection grid pixels should also be ~100
        assert!(image[(1, 3)] > 90 && image[(1, 3)] < 110);
        assert!(image[(3, 1)] > 90 && image[(3, 1)] < 110);
    }

    #[test]
    fn test_gradient_interpolation() {
        // Verify linear interpolation across a vertical grid gap with different boundary values.
        // Use 5 rows (columns classified as grid) and 10 columns (rows NOT classified as grid).
        let rows = 5;
        let cols = 10;
        let mut image = Array2::from_elem((rows, cols), 150u16);
        // Columns 3,4,5 are the grid gap
        for y in 0..rows {
            image[(y, 3)] = 0;
            image[(y, 4)] = 0;
            image[(y, 5)] = 0;
        }
        // Set distinct boundary values
        for y in 0..rows {
            image[(y, 2)] = 100; // left boundary
            image[(y, 6)] = 200; // right boundary
        }
        let mut mask = Array2::from_elem((rows, cols), false);
        for y in 0..rows {
            mask[(y, 3)] = true;
            mask[(y, 4)] = true;
            mask[(y, 5)] = true;
        }

        directional_interpolate(&mut image, &mask).unwrap();

        // Linear interpolation: 100 -> 125 -> 150 -> 175 -> 200
        for y in 0..rows {
            assert_eq!(image[(y, 3)], 125, "row {y} col 3");
            assert_eq!(image[(y, 4)], 150, "row {y} col 4");
            assert_eq!(image[(y, 5)], 175, "row {y} col 5");
        }
    }

    #[test]
    fn test_edge_grid_line() {
        // Grid line at column 0 (left edge) — only right boundary exists
        let mut image = Array2::from_elem((3, 5), 500u16);
        for y in 0..3 {
            image[(y, 0)] = 0;
        }
        let mut mask = Array2::from_elem((3, 5), false);
        for y in 0..3 {
            mask[(y, 0)] = true;
        }

        directional_interpolate(&mut image, &mask).unwrap();

        // Should get the right boundary value (500)
        for y in 0..3 {
            assert_eq!(image[(y, 0)], 500, "edge pixel row {y}");
        }
    }

    #[test]
    fn test_wide_gap() {
        // 2-pixel wide vertical grid line
        let mut image = Array2::from_elem((5, 6), 100u16);
        for y in 0..5 {
            image[(y, 2)] = 0;
            image[(y, 3)] = 0;
        }
        let mut mask = Array2::from_elem((5, 6), false);
        for y in 0..5 {
            mask[(y, 2)] = true;
            mask[(y, 3)] = true;
        }

        directional_interpolate(&mut image, &mask).unwrap();

        // Both grid columns should be interpolated smoothly
        for y in 0..5 {
            assert!(image[(y, 2)] > 90 && image[(y, 2)] < 110, "wide gap col 2 row {y}: {}", image[(y, 2)]);
            assert!(image[(y, 3)] > 90 && image[(y, 3)] < 110, "wide gap col 3 row {y}: {}", image[(y, 3)]);
        }
    }

    #[test]
    fn test_compute_min_nonzero() {
        let mut image = Array2::from_elem((3, 3), 0u16);
        image[(0, 0)] = 50;
        image[(1, 1)] = 30;
        image[(2, 2)] = 100;
        assert_eq!(compute_min_nonzero(&image), 30);
    }

    #[test]
    fn test_classify_grid_columns() {
        let mut mask = Array2::from_elem((4, 5), false);
        // Column 2: all masked (grid line)
        for y in 0..4 {
            mask[(y, 2)] = true;
        }
        // Column 4: only 1 masked (not a grid line)
        mask[(0, 4)] = true;

        let col_is_grid = classify_grid_columns(&mask);
        assert!(col_is_grid[2]);
        assert!(!col_is_grid[0]);
        assert!(!col_is_grid[4]);
    }

    #[test]
    fn test_classify_grid_rows() {
        let mut mask = Array2::from_elem((5, 4), false);
        // Row 1: all masked (grid line)
        for x in 0..4 {
            mask[(1, x)] = true;
        }
        // Row 3: only 1 masked (not a grid line)
        mask[(3, 0)] = true;

        let row_is_grid = classify_grid_rows(&mask);
        assert!(row_is_grid[1]);
        assert!(!row_is_grid[0]);
        assert!(!row_is_grid[3]);
    }

    #[test]
    fn test_no_masked_pixels() {
        let mut image = Array2::from_elem((3, 3), 100u16);
        let mask = Array2::from_elem((3, 3), false);
        directional_interpolate(&mut image, &mask).unwrap();
        // Image should be unchanged
        assert!(image.iter().all(|&v| v == 100));
    }
}
