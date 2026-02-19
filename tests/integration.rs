//! Integration tests comparing GridlineForge with MindaGap

use gridline_forge::{
    process_image,
    types::{GridPattern, ProcessingConfig},
};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Generate a synthetic test image with a grid pattern
fn generate_test_image_with_grid(width: usize, height: usize, grid_spacing: u32) -> Array2<u16> {
    let mut image = Array2::from_elem((height, width), 1000u16); // Background value

    // Add vertical grid lines (set to 0)
    let mut x = 0;
    while x < width {
        for y in 0..height {
            image[(y, x)] = 0;
        }
        x += grid_spacing as usize;
    }

    // Add horizontal grid lines (set to 0)
    let mut y = 0;
    while y < height {
        for x in 0..width {
            image[(y, x)] = 0;
        }
        y += grid_spacing as usize;
    }

    // Add some variation to non-grid pixels (simulate real data)
    for y in 0..height {
        for x in 0..width {
            if image[(y, x)] != 0 {
                let offset = ((x * 7 + y * 13) % 200) as u16;
                image[(y, x)] = 900 + offset;
            }
        }
    }

    image
}

/// Save array as TIFF for testing
fn save_test_image(image: &Array2<u16>, path: &Path) -> std::io::Result<()> {
    use image::{DynamicImage, ImageBuffer, Luma};

    let (height, width) = (image.nrows() as u32, image.ncols() as u32);
    let mut buf = ImageBuffer::<Luma<u16>, Vec<u16>>::new(width, height);

    for y in 0..height as usize {
        for x in 0..width as usize {
            buf.put_pixel(x as u32, y as u32, Luma([image[(y, x)]]));
        }
    }

    let dynamic = DynamicImage::ImageLuma16(buf);
    dynamic.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Load TIFF image for comparison
fn load_test_image(path: &Path) -> std::io::Result<Array2<u16>> {
    use image::GenericImageView;

    let img = image::open(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let gray = img.to_luma16();
    let (width, height) = gray.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            array[(y as usize, x as usize)] = gray.get_pixel(x, y)[0];
        }
    }

    Ok(array)
}

/// Calculate Root Mean Square Error between two images
fn calculate_rmse(img1: &Array2<u16>, img2: &Array2<u16>) -> f64 {
    assert_eq!(img1.shape(), img2.shape(), "Images must have same dimensions");

    let mut sum_squared_diff = 0.0;
    let n = (img1.nrows() * img1.ncols()) as f64;

    for y in 0..img1.nrows() {
        for x in 0..img1.ncols() {
            let diff = img1[(y, x)] as f64 - img2[(y, x)] as f64;
            sum_squared_diff += diff * diff;
        }
    }

    (sum_squared_diff / n).sqrt()
}

/// Calculate Peak Signal-to-Noise Ratio
fn calculate_psnr(img1: &Array2<u16>, img2: &Array2<u16>) -> f64 {
    let rmse = calculate_rmse(img1, img2);
    if rmse == 0.0 {
        return f64::INFINITY;
    }

    let max_value = 65535.0; // 16-bit max
    20.0 * (max_value / rmse).log10()
}

/// Check if MindaGap is available
fn is_mindagap_available() -> bool {
    Path::new("../mindagap.py").exists()
}

/// Run MindaGap on a test image using pixi
/// Note: MindaGap automatically creates INPUT_gridfilled.tif, so output_path is for reference only
fn run_mindagap(
    input_path: &Path,
    _output_path: &Path,  // Not used - MindaGap creates its own output name
    kernel_size: usize,
    iterations: usize,
    x_spacing: u32,
    y_spacing: u32,
) -> std::io::Result<()> {
    // Convert paths to absolute for MindaGap
    let abs_input = std::fs::canonicalize(input_path)?;

    let status = Command::new("pixi")
        .arg("run")
        .arg("python")
        .arg("mindagap.py")
        .arg(&abs_input)
        .arg("-s")
        .arg(kernel_size.to_string())
        .arg("-r")
        .arg(iterations.to_string())
        .arg("-xt")
        .arg(x_spacing.to_string())
        .arg("-yt")
        .arg(y_spacing.to_string())
        .current_dir("../")
        .status()?;

    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "MindaGap failed",
        ))
    }
}

#[test]
fn test_synthetic_image_basic() {
    // Generate test image
    let width = 1000;
    let height = 1000;
    let grid_spacing = 100;

    let test_image = generate_test_image_with_grid(width, height, grid_spacing);

    // Save test image
    let test_dir = PathBuf::from("tests/fixtures");
    std::fs::create_dir_all(&test_dir).unwrap();

    let input_path = test_dir.join("test_synthetic.tif");
    save_test_image(&test_image, &input_path).unwrap();

    // Process with GridlineForge
    let output_path = test_dir.join("test_synthetic_gridfilled.tif");

    let pattern = GridPattern::new(grid_spacing, grid_spacing);
    let config = ProcessingConfig {
        kernel_size: 5,
        refinement_rounds: 5,

        show_progress: false,
        benchmark: false,
    };

    process_image(&input_path, &output_path, Some(pattern), &config).unwrap();

    // Verify output exists
    assert!(output_path.exists(), "Output file should exist");

    // Load and verify output
    let output_image = load_test_image(&output_path).unwrap();
    assert_eq!(output_image.shape(), test_image.shape());

    // Verify grid pixels are no longer zero
    let mut zero_count = 0;
    for y in 0..height {
        for x in 0..width {
            if output_image[(y, x)] == 0 {
                zero_count += 1;
            }
        }
    }

    // Most grid pixels should be filled (allow some tolerance)
    let total_grid_pixels = (width / grid_spacing as usize + 1) * height
        + (height / grid_spacing as usize + 1) * width;
    let fill_rate = 1.0 - (zero_count as f64 / total_grid_pixels as f64);

    println!("Grid fill rate: {:.2}%", fill_rate * 100.0);
    assert!(fill_rate > 0.95, "At least 95% of grid pixels should be filled");

    // Clean up
    std::fs::remove_file(input_path).ok();
    std::fs::remove_file(output_path).ok();
}

#[test]
fn test_auto_detection() {
    // Generate test image
    let width = 800;
    let height = 800;
    let grid_spacing = 80;

    let test_image = generate_test_image_with_grid(width, height, grid_spacing);

    // Save test image
    let test_dir = PathBuf::from("tests/fixtures");
    std::fs::create_dir_all(&test_dir).unwrap();

    let input_path = test_dir.join("test_autodetect.tif");
    save_test_image(&test_image, &input_path).unwrap();

    // Process with GridlineForge (auto-detection)
    let output_path = test_dir.join("test_autodetect_gridfilled.tif");

    let config = ProcessingConfig {
        kernel_size: 5,
        refinement_rounds: 5,

        show_progress: false,
        benchmark: false,
    };

    // Auto-detect (no manual pattern)
    let detected_pattern = process_image(&input_path, &output_path, None, &config).unwrap();

    println!("Detected grid: {}×{}", detected_pattern.x_spacing, detected_pattern.y_spacing);

    // Verify detection is close to actual
    assert!(
        (detected_pattern.x_spacing as i32 - grid_spacing as i32).abs() < 5,
        "X spacing detection should be accurate"
    );
    assert!(
        (detected_pattern.y_spacing as i32 - grid_spacing as i32).abs() < 5,
        "Y spacing detection should be accurate"
    );

    // Clean up
    std::fs::remove_file(input_path).ok();
    std::fs::remove_file(output_path).ok();
}

#[test]
#[ignore] // Only run when MindaGap is available: cargo test -- --ignored
fn test_compare_with_mindagap() {
    if !is_mindagap_available() {
        eprintln!("Skipping MindaGap comparison: mindagap.py not found");
        return;
    }

    // Generate test image
    let width = 1000;
    let height = 1000;
    let grid_spacing = 100;

    let test_image = generate_test_image_with_grid(width, height, grid_spacing);

    // Save test image
    let test_dir = PathBuf::from("tests/fixtures");
    std::fs::create_dir_all(&test_dir).unwrap();

    let input_path = test_dir.join("test_comparison.tif");
    save_test_image(&test_image, &input_path).unwrap();

    // Process with GridlineForge
    let gridforge_output = test_dir.join("test_comparison_gridforge.tif");
    let pattern = GridPattern::new(grid_spacing, grid_spacing);
    let config = ProcessingConfig {
        kernel_size: 5,
        refinement_rounds: 10, // Use more iterations for fair comparison

        show_progress: false,
        benchmark: false,
    };

    process_image(&input_path, &gridforge_output, Some(pattern), &config).unwrap();

    // Process with MindaGap (it will create test_comparison_gridfilled.tif)
    let mindagap_output = test_dir.join("test_comparison_gridfilled.tif");
    run_mindagap(&input_path, &mindagap_output, 5, 40, grid_spacing, grid_spacing).unwrap();

    // Load both outputs
    let gridforge_image = load_test_image(&gridforge_output).unwrap();
    let mindagap_image = load_test_image(&mindagap_output).unwrap();

    // Calculate metrics
    let rmse = calculate_rmse(&gridforge_image, &mindagap_image);
    let psnr = calculate_psnr(&gridforge_image, &mindagap_image);

    println!("\n=== Comparison Metrics ===");
    println!("RMSE: {:.2}", rmse);
    println!("PSNR: {:.2} dB", psnr);

    // Assert outputs are similar (RMSE < 50 on 16-bit scale)
    assert!(
        rmse < 50.0,
        "RMSE should be low (GridlineForge vs MindaGap should produce similar results)"
    );

    // PSNR should be reasonably high (> 40 dB is good)
    assert!(
        psnr > 40.0,
        "PSNR should be high (outputs are similar)"
    );

    // Clean up
    std::fs::remove_file(input_path).ok();
    std::fs::remove_file(gridforge_output).ok();
    std::fs::remove_file(mindagap_output).ok();
}

#[test]
fn test_edge_cases() {
    let test_dir = PathBuf::from("tests/fixtures");
    std::fs::create_dir_all(&test_dir).unwrap();

    // Test 1: Very small image
    let small_image = generate_test_image_with_grid(100, 100, 10);
    let small_path = test_dir.join("test_small.tif");
    save_test_image(&small_image, &small_path).unwrap();

    let output_path = test_dir.join("test_small_gridfilled.tif");
    let pattern = GridPattern::new(10, 10);
    let config = ProcessingConfig::default();

    let result = process_image(&small_path, &output_path, Some(pattern), &config);
    assert!(result.is_ok(), "Should handle small images");

    std::fs::remove_file(small_path).ok();
    std::fs::remove_file(output_path).ok();

    // Test 2: Irregular spacing (should still work)
    let irregular = generate_test_image_with_grid(500, 500, 73);
    let irregular_path = test_dir.join("test_irregular.tif");
    save_test_image(&irregular, &irregular_path).unwrap();

    let output_path = test_dir.join("test_irregular_gridfilled.tif");
    let pattern = GridPattern::new(73, 73);

    let result = process_image(&irregular_path, &output_path, Some(pattern), &config);
    assert!(result.is_ok(), "Should handle irregular spacing");

    std::fs::remove_file(irregular_path).ok();
    std::fs::remove_file(output_path).ok();
}
