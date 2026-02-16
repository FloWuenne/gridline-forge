//! FFT-based grid pattern detection

use crate::types::{DetectionConfig, GridPattern, Result};
use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

/// Detect grid pattern using FFT frequency analysis
pub fn detect_grid_pattern(
    image: &Array2<u16>,
    config: &DetectionConfig,
) -> Result<Option<GridPattern>> {
    let (_height, _width) = (image.nrows(), image.ncols());

    // Project image along both axes to find periodic patterns
    let row_projection = project_rows(image);
    let col_projection = project_cols(image);

    // Analyze frequencies in both projections
    let x_spacing = find_dominant_frequency(&col_projection, config.min_spacing, config.max_spacing);
    let y_spacing = find_dominant_frequency(&row_projection, config.min_spacing, config.max_spacing);

    // If no clear frequencies found, try direct zero-detection fallback
    if x_spacing.is_none() && y_spacing.is_none() {
        return detect_grid_direct(image, config);
    }

    // Build preliminary grid pattern
    let mut pattern = GridPattern {
        x_spacing: x_spacing.unwrap_or(0),
        y_spacing: y_spacing.unwrap_or(0),
        x_offset: 0,
        y_offset: 0,
        grid_width: 1,
        confidence: 0.0,
    };

    // Find offsets by locating actual grid positions
    if pattern.x_spacing > 0 {
        pattern.x_offset = find_grid_offset(&col_projection, pattern.x_spacing);
    }
    if pattern.y_spacing > 0 {
        pattern.y_offset = find_grid_offset(&row_projection, pattern.y_spacing);
    }

    // Validate pattern against actual image data
    let confidence = validate_pattern(image, &pattern);
    pattern.confidence = confidence;

    if confidence >= config.confidence_threshold {
        Ok(Some(pattern))
    } else {
        // Low confidence, try fallback
        detect_grid_direct(image, config)
    }
}

/// Project image values along rows (sum columns)
fn project_rows(image: &Array2<u16>) -> Vec<f64> {
    let height = image.nrows();
    let mut projection = Vec::with_capacity(height);

    for row in image.rows() {
        let sum: u64 = row.iter().map(|&x| x as u64).sum();
        projection.push(sum as f64);
    }

    projection
}

/// Project image values along columns (sum rows)
fn project_cols(image: &Array2<u16>) -> Vec<f64> {
    let width = image.ncols();
    let mut projection = Vec::with_capacity(width);

    for col_idx in 0..width {
        let sum: u64 = image.column(col_idx).iter().map(|&x| x as u64).sum();
        projection.push(sum as f64);
    }

    projection
}

/// Find dominant frequency in a signal using FFT
fn find_dominant_frequency(signal: &[f64], min_spacing: u32, max_spacing: u32) -> Option<u32> {
    let n = signal.len();
    if n < min_spacing as usize {
        return None;
    }

    // Convert to complex and apply FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    fft.process(&mut buffer);

    // Find peak in the magnitude spectrum within valid frequency range
    let min_freq_idx = n / max_spacing as usize;
    let max_freq_idx = n / min_spacing as usize;

    if min_freq_idx >= max_freq_idx || max_freq_idx >= n / 2 {
        return None;
    }

    let mut max_magnitude = 0.0;
    let mut peak_idx = 0;

    for i in min_freq_idx..=max_freq_idx.min(n / 2) {
        let magnitude = buffer[i].norm();
        if magnitude > max_magnitude {
            max_magnitude = magnitude;
            peak_idx = i;
        }
    }

    // Check if peak is significant
    let mean_magnitude: f64 = buffer[min_freq_idx..=max_freq_idx.min(n / 2)]
        .iter()
        .map(|c| c.norm())
        .sum::<f64>() / (max_freq_idx - min_freq_idx) as f64;

    // Peak should be at least 3x the mean to be considered significant
    if max_magnitude > 3.0 * mean_magnitude && peak_idx > 0 {
        let spacing = n / peak_idx;
        Some(spacing as u32)
    } else {
        None
    }
}

/// Find offset of grid lines by finding minima in projection
fn find_grid_offset(projection: &[f64], spacing: u32) -> u32 {
    if spacing == 0 || projection.is_empty() {
        return 0;
    }

    // Look at first period to find minimum (grid line position)
    let period_len = (spacing as usize).min(projection.len());
    let mut min_val = f64::MAX;
    let mut min_idx = 0;

    for i in 0..period_len {
        if projection[i] < min_val {
            min_val = projection[i];
            min_idx = i;
        }
    }

    min_idx as u32
}

/// Validate detected pattern against actual image pixels
fn validate_pattern(image: &Array2<u16>, pattern: &GridPattern) -> f32 {
    let (height, width) = (image.nrows() as u32, image.ncols() as u32);

    // Sample grid positions and check how many are actually zero or near-zero
    let mut grid_pixels = 0;
    let mut zero_pixels = 0;

    // Sample vertical lines
    if pattern.x_spacing > 0 {
        let mut x = pattern.x_offset;
        while x < width {
            for y in (0..height).step_by(10) {
                // Sample every 10 pixels
                grid_pixels += 1;
                let val = image[(y as usize, x as usize)];
                if val == 0 {
                    zero_pixels += 1;
                }
            }
            x += pattern.x_spacing;
        }
    }

    // Sample horizontal lines
    if pattern.y_spacing > 0 {
        let mut y = pattern.y_offset;
        while y < height {
            for x in (0..width).step_by(10) {
                // Sample every 10 pixels
                grid_pixels += 1;
                let val = image[(y as usize, x as usize)];
                if val == 0 {
                    zero_pixels += 1;
                }
            }
            y += pattern.y_spacing;
        }
    }

    if grid_pixels == 0 {
        return 0.0;
    }

    // Confidence is the fraction of sampled grid pixels that are zero
    zero_pixels as f32 / grid_pixels as f32
}

/// Fallback: Direct detection of zero-valued pixels as grid
fn detect_grid_direct(image: &Array2<u16>, config: &DetectionConfig) -> Result<Option<GridPattern>> {
    let (height, width) = (image.nrows(), image.ncols());

    // Count zero pixels in each row and column
    let mut col_zeros = vec![0u32; width];
    let mut row_zeros = vec![0u32; height];

    for y in 0..height {
        for x in 0..width {
            if image[(y, x)] == 0 {
                col_zeros[x] += 1;
                row_zeros[y] += 1;
            }
        }
    }

    // Find columns with high zero count (potential vertical grid lines)
    let col_threshold = (height as f32 * 0.8) as u32;
    let vertical_lines: Vec<u32> = col_zeros
        .iter()
        .enumerate()
        .filter(|(_, &count)| count >= col_threshold)
        .map(|(idx, _)| idx as u32)
        .collect();

    // Find rows with high zero count (potential horizontal grid lines)
    let row_threshold = (width as f32 * 0.8) as u32;
    let horizontal_lines: Vec<u32> = row_zeros
        .iter()
        .enumerate()
        .filter(|(_, &count)| count >= row_threshold)
        .map(|(idx, _)| idx as u32)
        .collect();

    // Compute spacing from line positions
    let x_spacing = compute_spacing(&vertical_lines, config.min_spacing, config.max_spacing);
    let y_spacing = compute_spacing(&horizontal_lines, config.min_spacing, config.max_spacing);

    if x_spacing.is_none() && y_spacing.is_none() {
        return Ok(None);
    }

    let pattern = GridPattern {
        x_spacing: x_spacing.unwrap_or(0),
        y_spacing: y_spacing.unwrap_or(0),
        x_offset: vertical_lines.first().copied().unwrap_or(0),
        y_offset: horizontal_lines.first().copied().unwrap_or(0),
        grid_width: 1,
        confidence: 0.9, // High confidence for direct detection
    };

    Ok(Some(pattern))
}

/// Compute spacing from detected line positions
fn compute_spacing(lines: &[u32], min_spacing: u32, max_spacing: u32) -> Option<u32> {
    if lines.len() < 2 {
        return None;
    }

    // Compute differences between consecutive lines
    let mut diffs: Vec<u32> = lines.windows(2).map(|w| w[1] - w[0]).collect();

    if diffs.is_empty() {
        return None;
    }

    // Find most common spacing
    diffs.sort_unstable();
    let median = diffs[diffs.len() / 2];

    if median >= min_spacing && median <= max_spacing {
        Some(median)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_rows() {
        let mut image = Array2::zeros((3, 4));
        image[(0, 0)] = 1;
        image[(0, 1)] = 2;
        image[(0, 2)] = 3;
        image[(0, 3)] = 4;

        let projection = project_rows(&image);
        assert_eq!(projection.len(), 3);
        assert_eq!(projection[0], 10.0); // 1+2+3+4
        assert_eq!(projection[1], 0.0);
        assert_eq!(projection[2], 0.0);
    }

    #[test]
    fn test_compute_spacing() {
        let lines = vec![10, 110, 210, 310];
        let config = DetectionConfig::default();
        let spacing = compute_spacing(&lines, config.min_spacing, config.max_spacing);
        assert_eq!(spacing, Some(100));
    }
}
