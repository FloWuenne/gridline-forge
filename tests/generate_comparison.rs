//! Generate comparison images for visual inspection

use gridline_forge::{process_image, types::{GridPattern, ProcessingConfig}};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Array2;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("Generating comparison images...\n");

    // Generate synthetic test image with grid
    let width = 1000;
    let height = 1000;
    let grid_spacing = 100u32;

    println!("1. Creating test image ({}×{} with {}px grid)...", width, height, grid_spacing);
    let mut test_image = Array2::from_elem((height, width), 1000u16);

    // Add vertical grid lines (set to 0)
    let mut x = 0;
    while x < width {
        for y in 0..height {
            test_image[(y, x)] = 0;
        }
        x += grid_spacing as usize;
    }

    // Add horizontal grid lines (set to 0)
    let mut y = 0;
    while y < height {
        for x in 0..width {
            test_image[(y, x)] = 0;
        }
        y += grid_spacing as usize;
    }

    // Add some variation to non-grid pixels
    for y in 0..height {
        for x in 0..width {
            if test_image[(y, x)] != 0 {
                let offset = ((x * 7 + y * 13) % 200) as u16;
                test_image[(y, x)] = 900 + offset;
            }
        }
    }

    // Create output directory
    let output_dir = PathBuf::from("comparison_output");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Save original test image
    let original_path = output_dir.join("01_original_with_grid.tif");
    println!("   Saving: {}", original_path.display());
    save_image_u16(&test_image, &original_path);

    // Process with GridlineForge
    println!("\n2. Processing with GridlineForge...");
    let gridforge_output = output_dir.join("02_gridlineforge_output.tif");
    let pattern = GridPattern::new(grid_spacing, grid_spacing);
    let config = ProcessingConfig {
        kernel_size: 5,
        refinement_rounds: 10,
        fmm_radius: 3,
        show_progress: true,
        benchmark: true,
    };

    process_image(&original_path, &gridforge_output, Some(pattern), &config)
        .expect("GridlineForge processing failed");
    println!("   Saved: {}", gridforge_output.display());

    // Process with MindaGap
    println!("\n3. Processing with MindaGap (via pixi)...");
    let abs_original = std::fs::canonicalize(&original_path).expect("Failed to canonicalize path");

    let status = Command::new("pixi")
        .arg("run")
        .arg("python")
        .arg("mindagap.py")
        .arg(&abs_original)
        .arg("-s")
        .arg("5")
        .arg("-r")
        .arg("40")
        .arg("-xt")
        .arg(grid_spacing.to_string())
        .arg("-yt")
        .arg(grid_spacing.to_string())
        .current_dir("../")
        .status()
        .expect("Failed to run MindaGap");

    if !status.success() {
        eprintln!("Warning: MindaGap processing failed");
    }

    // MindaGap creates OUTPUT_gridfilled.tif, move it to our output directory
    let mindagap_auto_output = output_dir.join("01_original_with_grid_gridfilled.tif");
    let mindagap_renamed = output_dir.join("03_mindagap_output.tif");

    if mindagap_auto_output.exists() {
        std::fs::rename(&mindagap_auto_output, &mindagap_renamed)
            .expect("Failed to rename MindaGap output");
        println!("   Saved: {}", mindagap_renamed.display());
    }

    // Generate a difference image
    println!("\n4. Generating difference image...");
    let gridforge_img = load_image_u16(&gridforge_output);
    let mindagap_img = load_image_u16(&mindagap_renamed);

    let mut diff_image = Array2::zeros((height, width));
    let mut max_diff = 0u16;

    for y in 0..height {
        for x in 0..width {
            let diff = (gridforge_img[(y, x)] as i32 - mindagap_img[(y, x)] as i32).abs() as u16;
            diff_image[(y, x)] = diff * 1000; // Amplify for visibility
            max_diff = max_diff.max(diff);
        }
    }

    let diff_path = output_dir.join("04_difference_amplified.tif");
    save_image_u16(&diff_image, &diff_path);
    println!("   Saved: {}", diff_path.display());
    println!("   Max pixel difference: {} (amplified by 1000× for visibility)", max_diff);

    // Calculate metrics
    println!("\n5. Calculating metrics...");
    let rmse = calculate_rmse(&gridforge_img, &mindagap_img);
    let psnr = calculate_psnr(&gridforge_img, &mindagap_img);

    println!("   RMSE: {:.2}", rmse);
    println!("   PSNR: {:.2} dB", psnr);

    println!("\n✓ Done! Images saved to: {}", output_dir.display());
    println!("\nYou can now open these files in ImageJ, Fiji, or any TIFF viewer:");
    println!("  1. 01_original_with_grid.tif     - Original with grid lines (black lines)");
    println!("  2. 02_gridlineforge_output.tif   - GridlineForge result");
    println!("  3. 03_mindagap_output.tif        - MindaGap result");
    println!("  4. 04_difference_amplified.tif   - Difference (amplified 1000×)");
}

fn save_image_u16(array: &Array2<u16>, path: &std::path::Path) {
    let (height, width) = (array.nrows() as u32, array.ncols() as u32);
    let mut buf = ImageBuffer::<Luma<u16>, Vec<u16>>::new(width, height);

    for y in 0..height as usize {
        for x in 0..width as usize {
            buf.put_pixel(x as u32, y as u32, Luma([array[(y, x)]]));
        }
    }

    let dynamic = DynamicImage::ImageLuma16(buf);
    dynamic.save(path).expect("Failed to save image");
}

fn load_image_u16(path: &std::path::Path) -> Array2<u16> {
    let img = image::open(path).expect("Failed to load image");
    let gray = img.to_luma16();
    let (width, height) = gray.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            array[(y as usize, x as usize)] = gray.get_pixel(x, y)[0];
        }
    }

    array
}

fn calculate_rmse(img1: &Array2<u16>, img2: &Array2<u16>) -> f64 {
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

fn calculate_psnr(img1: &Array2<u16>, img2: &Array2<u16>) -> f64 {
    let rmse = calculate_rmse(img1, img2);
    if rmse == 0.0 {
        return f64::INFINITY;
    }

    let max_value = 65535.0;
    20.0 * (max_value / rmse).log10()
}
