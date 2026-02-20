//! GridlineForge CLI

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gridline_forge::{
    duplicate_finder::{tsv_io, DuplicateFinderConfig},
    image_io, process_image,
    types::{GridPattern, ImageFormat, ProcessingConfig},
};
use std::path::PathBuf;

/// GridlineForge: Remove grid lines from panorama images and find duplicate transcripts
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Remove and inpaint grid lines in panorama images
    Process {
        /// Input image file (TIFF, PNG, JPEG)
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output file (default: INPUT_gridfilled.EXT)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Gaussian kernel size (default: 5, must be odd)
        #[arg(short = 'k', long, default_value = "5")]
        kernel_size: usize,

        /// Gaussian blur iterations (default: 40, matching MindaGap)
        #[arg(short = 'r', long, default_value = "40")]
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
    },

    /// Find duplicate gene transcripts at grid line boundaries
    DuplicateFinder {
        /// Input XYZ gene coordinates file (tab-separated: x y z gene)
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output file (default: INPUT_markedDups.txt)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Tile size on X axis — distance between vertical gridlines (default: 2144)
        #[arg(long, default_value = "2144")]
        x_tilesize: u32,

        /// Tile size on Y axis (default: same as x-tilesize)
        #[arg(long)]
        y_tilesize: Option<u32>,

        /// Window size around gridlines to search for duplicates (pixels each side, default: 30)
        #[arg(short = 'w', long, default_value = "30")]
        window_size: u32,

        /// Maximum per-gene transcript count for shift calculation (default: 400)
        #[arg(long, default_value = "400")]
        max_freq: usize,

        /// Minimum occurrences of a ~XYZ shift to consider it valid (default: 10)
        #[arg(long, default_value = "10")]
        min_mode: usize,

        /// Suppress progress output
        #[arg(short, long)]
        quiet: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Process {
            input,
            output,
            kernel_size,
            refinement_rounds,
            x_spacing,
            y_spacing,
            no_auto_detect,
            format,
            quiet,
            benchmark,
        } => run_process(
            input,
            output,
            kernel_size,
            refinement_rounds,
            x_spacing,
            y_spacing,
            no_auto_detect,
            format,
            quiet,
            benchmark,
        ),

        Command::DuplicateFinder {
            input,
            output,
            x_tilesize,
            y_tilesize,
            window_size,
            max_freq,
            min_mode,
            quiet,
        } => run_duplicate_finder(
            input,
            output,
            x_tilesize,
            y_tilesize,
            window_size,
            max_freq,
            min_mode,
            quiet,
        ),
    }
}

fn run_process(
    input: PathBuf,
    output: Option<PathBuf>,
    kernel_size: usize,
    refinement_rounds: usize,
    x_spacing: Option<u32>,
    y_spacing: Option<u32>,
    no_auto_detect: bool,
    format: Option<String>,
    quiet: bool,
    benchmark: bool,
) -> Result<()> {
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }
    if kernel_size % 2 == 0 {
        anyhow::bail!("Kernel size must be odd, got {}", kernel_size);
    }

    let output_path = if let Some(out) = output {
        out
    } else {
        let fmt = format
            .as_ref()
            .and_then(|f| ImageFormat::from_extension(f));
        image_io::generate_output_path(&input, fmt).context("Failed to generate output path")?
    };

    let config = ProcessingConfig {
        kernel_size,
        refinement_rounds,
        show_progress: !quiet,
        benchmark,
    };

    let manual_grid = if x_spacing.is_some() || y_spacing.is_some() {
        Some(GridPattern::new(
            x_spacing.unwrap_or(0),
            y_spacing.unwrap_or(0),
        ))
    } else if no_auto_detect {
        anyhow::bail!("--no-auto-detect requires --x-spacing and --y-spacing");
    } else {
        None
    };

    if !quiet {
        println!("GridlineForge v{}", env!("CARGO_PKG_VERSION"));
        println!("Input:  {}", input.display());
        println!("Output: {}\n", output_path.display());
    }

    match process_image(&input, &output_path, manual_grid, &config) {
        Ok(pattern) => {
            if !quiet {
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

fn run_duplicate_finder(
    input: PathBuf,
    output: Option<PathBuf>,
    x_tilesize: u32,
    y_tilesize: Option<u32>,
    window_size: u32,
    max_freq: usize,
    min_mode: usize,
    quiet: bool,
) -> Result<()> {
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    let output_path = if let Some(out) = output {
        out
    } else {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .context("Invalid input filename")?;
        let parent = input.parent().unwrap_or(std::path::Path::new("."));
        parent.join(format!("{}_markedDups.txt", stem))
    };

    if !quiet {
        println!("GridlineForge v{} — duplicate-finder", env!("CARGO_PKG_VERSION"));
        println!("Input:  {}", input.display());
        println!("Output: {}", output_path.display());
        println!(
            "Config: x_tilesize={}, y_tilesize={}, window={}, max_freq={}, min_mode={}\n",
            x_tilesize,
            y_tilesize.unwrap_or(x_tilesize),
            window_size,
            max_freq,
            min_mode
        );
    }

    let records = tsv_io::read_tsv(&input).context("Failed to read input TSV")?;
    if !quiet {
        println!("Read {} transcript records", records.len());
    }

    let config = DuplicateFinderConfig {
        x_tilesize,
        y_tilesize: y_tilesize.unwrap_or(x_tilesize),
        window_size,
        max_freq,
        min_mode,
        show_progress: !quiet,
    };

    let (duplicates, stats) = gridline_forge::duplicate_finder::find_duplicates(&records, &config);

    tsv_io::write_tsv(&output_path, &records, &duplicates)
        .context("Failed to write output TSV")?;

    println!("TilePairs, Tile Overlaps, Total duplicated transcripts");
    println!(
        "{} {} {}",
        stats.tile_pairs, stats.tile_overlaps, stats.total_dups
    );

    Ok(())
}
