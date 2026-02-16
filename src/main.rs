//! GridlineForge CLI

use anyhow::{Context, Result};
use clap::Parser;
use gridline_forge::{
    image_io, process_image,
    types::{GridPattern, ImageFormat, ProcessingConfig},
};
use std::path::PathBuf;

/// GridlineForge: Remove and inpaint grid lines in panorama images
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image file (TIFF, PNG, JPEG)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file (default: INPUT_gridfilled.EXT)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Gaussian kernel size (default: 5, must be odd)
    #[arg(short = 'k', long, default_value = "5")]
    kernel_size: usize,

    /// Refinement iterations (default: 5)
    #[arg(short = 'r', long, default_value = "5")]
    refinement_rounds: usize,

    /// Manual X grid spacing (auto-detect if not set)
    #[arg(long, value_name = "PIXELS")]
    x_spacing: Option<u32>,

    /// Manual Y grid spacing (auto-detect if not set)
    #[arg(long, value_name = "PIXELS")]
    y_spacing: Option<u32>,

    /// Disable auto-detection, use zero-valued pixels
    #[arg(long)]
    no_auto_detect: bool,

    /// Output format: tiff, png, jpeg (default: same as input)
    #[arg(short = 'f', long, value_name = "FORMAT")]
    format: Option<String>,

    /// Suppress progress output
    #[arg(short, long)]
    quiet: bool,

    /// Show detailed timing information
    #[arg(long)]
    benchmark: bool,

    /// Fast marching radius (default: 3)
    #[arg(long, default_value = "3")]
    fmm_radius: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate input
    if !args.input.exists() {
        anyhow::bail!("Input file does not exist: {}", args.input.display());
    }

    // Validate kernel size
    if args.kernel_size % 2 == 0 {
        anyhow::bail!("Kernel size must be odd, got {}", args.kernel_size);
    }

    // Determine output path
    let output_path = if let Some(out) = args.output {
        out
    } else {
        let format = args
            .format
            .as_ref()
            .and_then(|f| ImageFormat::from_extension(f));

        image_io::generate_output_path(&args.input, format)
            .context("Failed to generate output path")?
    };

    // Build processing config
    let config = ProcessingConfig {
        kernel_size: args.kernel_size,
        refinement_rounds: args.refinement_rounds,
        fmm_radius: args.fmm_radius,
        show_progress: !args.quiet,
        benchmark: args.benchmark,
    };

    // Manual grid pattern if specified
    let manual_grid = if args.x_spacing.is_some() || args.y_spacing.is_some() {
        Some(GridPattern::new(
            args.x_spacing.unwrap_or(0),
            args.y_spacing.unwrap_or(0),
        ))
    } else if args.no_auto_detect {
        anyhow::bail!("--no-auto-detect requires --x-spacing and --y-spacing");
    } else {
        None
    };

    // Process image
    if !args.quiet {
        println!("GridlineForge v{}", env!("CARGO_PKG_VERSION"));
        println!("Input:  {}", args.input.display());
        println!("Output: {}\n", output_path.display());
    }

    match process_image(&args.input, &output_path, manual_grid, &config) {
        Ok(pattern) => {
            if !args.quiet {
                println!("\nSuccess! Grid pattern used:");
                println!("  X spacing: {} pixels", pattern.x_spacing);
                println!("  Y spacing: {} pixels", pattern.y_spacing);
                println!("  Confidence: {:.2}", pattern.confidence);
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            anyhow::bail!("Processing failed");
        }
    }
}
