//! GridlineForge: High-performance grid line removal and inpainting for panorama images
//!
//! This library provides fast algorithms for detecting and filling gridlines in
//! panoramic images, using a hybrid approach combining FFT-based detection,
//! Fast Marching Method inpainting, and localized iterative refinement.

pub mod detection;
pub mod image_io;
pub mod inpainting;
pub mod parallel;
pub mod refinement;
pub mod types;

use std::path::Path;
use std::time::Instant;
use types::{DetectionConfig, GridPattern, ImageData, ImageFormat, ProcessingConfig, Result};

/// Main entry point: process an image with grid line removal
///
/// # Arguments
/// * `input_path` - Path to input image
/// * `output_path` - Path for output image
/// * `manual_grid` - Optional manual grid pattern (None for auto-detection)
/// * `config` - Processing configuration
///
/// # Returns
/// Grid pattern that was used (detected or manual)
pub fn process_image(
    input_path: &Path,
    output_path: &Path,
    manual_grid: Option<GridPattern>,
    config: &ProcessingConfig,
) -> Result<GridPattern> {
    let start_time = Instant::now();

    // Load image
    if config.show_progress {
        println!("Loading image...");
    }
    let mut image_data = image_io::load_image(input_path)?;
    let load_time = start_time.elapsed();

    if config.benchmark {
        println!("  Load time: {:.2}s", load_time.as_secs_f32());
    }

    // Detect or use manual grid pattern
    let pattern = if let Some(grid) = manual_grid {
        if config.show_progress {
            println!(
                "Using manual grid: {}×{} pixels",
                grid.x_spacing, grid.y_spacing
            );
        }
        grid
    } else {
        if config.show_progress {
            println!("Detecting grid pattern...");
        }
        let detect_start = Instant::now();
        let detection_config = DetectionConfig::default();

        let pattern = match &image_data {
            ImageData::Gray16(arr) => detection::detect_grid_pattern(arr, &detection_config)?,
            ImageData::Gray8(arr) => {
                // Convert to u16 for detection
                let arr16 = arr.mapv(|x| x as u16);
                detection::detect_grid_pattern(&arr16, &detection_config)?
            }
            _ => None,
        };

        let detect_time = detect_start.elapsed();
        if config.benchmark {
            println!("  Detection time: {:.2}s", detect_time.as_secs_f32());
        }

        pattern.ok_or_else(|| {
            types::Error::Detection(
                "Failed to detect grid pattern. Try manual spacing with --x-spacing and --y-spacing"
                    .to_string(),
            )
        })?
    };

    if config.show_progress {
        println!(
            "  Grid: {}×{} pixels (offset: {}, {}), confidence: {:.2}",
            pattern.x_spacing,
            pattern.y_spacing,
            pattern.x_offset,
            pattern.y_offset,
            pattern.confidence
        );
    }

    // Process image based on type
    let process_start = Instant::now();
    match &mut image_data {
        ImageData::Gray8(arr) => {
            // Convert to u16 for processing
            let mut arr16 = arr.mapv(|x| x as u16);
            parallel::process_single_layer(arr16.view_mut(), &pattern, config)?;
            // Convert back to u8
            *arr = arr16.mapv(|x| (x / 256) as u8);
        }
        ImageData::Gray16(arr) => {
            parallel::process_single_layer(arr.view_mut(), &pattern, config)?;
        }
        ImageData::Multi8(_) => {
            return Err(types::Error::UnsupportedFormat(
                "Multi-layer 8-bit images not yet supported".to_string(),
            ));
        }
        ImageData::Multi16(arr) => {
            parallel::process_layers_parallel(arr, &pattern, config)?;
        }
    }

    let process_time = process_start.elapsed();
    if config.benchmark {
        println!("  Processing time: {:.2}s", process_time.as_secs_f32());
    }

    // Save output
    if config.show_progress {
        println!("Saving output...");
    }
    let save_start = Instant::now();

    let format = output_path
        .extension()
        .and_then(|s| s.to_str())
        .and_then(ImageFormat::from_extension)
        .unwrap_or(ImageFormat::Tiff);

    image_io::save_image(output_path, &image_data, format)?;

    let save_time = save_start.elapsed();
    if config.benchmark {
        println!("  Save time: {:.2}s", save_time.as_secs_f32());
    }

    let total_time = start_time.elapsed();
    if config.show_progress {
        println!("\n✓ Completed in {:.2}s", total_time.as_secs_f32());
    }

    Ok(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_grid() {
        let pattern = GridPattern::new(100, 100);
        assert_eq!(pattern.x_spacing, 100);
        assert_eq!(pattern.y_spacing, 100);
        assert_eq!(pattern.confidence, 1.0);
    }
}
