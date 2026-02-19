# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GridlineForge is a Rust CLI tool and library for detecting and removing periodic grid lines from panoramic microscopy images. It is a high-performance replacement for the Python-based MindaGap tool, targeting 10-100x speedup. Grid lines are zero-valued pixel rows/columns at regular intervals in stitched panorama TIFF images.

## Build & Test Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build (optimized)
cargo test                     # Run all tests (unit + integration)
cargo test <test_name>         # Run a single test
cargo test -- --ignored        # Run ignored tests (MindaGap comparison, requires ../MindaGap/)
cargo bench                    # Run benchmarks (criterion)
cargo clippy                   # Lint
cargo fmt                      # Format
```

**Note:** Two binaries exist, so use `--bin gridline-forge` when running:
```bash
cargo run --release --bin gridline-forge -- <input.tiff> -o <output.tiff>
```

Integration tests generate synthetic TIFF images in `tests/fixtures/` (auto-cleaned).

## Architecture

Two-stage processing pipeline, orchestrated by `process_image()` in `src/lib.rs`:

1. **Detection** (`src/detection/`) — FFT-based frequency analysis on row/column projections to find grid spacing. Falls back to direct zero-pixel counting if FFT confidence is low. Only operates on `Array2<u16>` (8-bit is upcast).

2. **MindaGap-style Gaussian blur** (`src/refinement/gaussian.rs` + `src/parallel/mod.rs`) — Matches MindaGap's algorithm exactly:
   - Mask zero-valued pixels
   - Initialize grid pixels to min non-zero value
   - Iteratively blur full image with separable Gaussian, copying back only grid pixels
   - Default: 40 iterations, kernel size 5 (matching MindaGap defaults)
   - Uses OpenCV-matching fixed binomial kernels for ksize 1/3/5/7
   - Per-iteration rounding to match uint16 behavior
   - Verified to match OpenCV's `cv2.GaussianBlur` within MAE ~20 on 16-bit scale

### Key modules

- `src/types.rs` — All core types: `GridPattern`, `ProcessingConfig` (kernel_size, refinement_rounds), `DetectionConfig`, `ImageData` enum (Gray8/Gray16/Multi8/Multi16 via ndarray), custom `Error`/`Result`
- `src/parallel/mod.rs` — Layer processing orchestration. `process_single_layer()` creates mask, initializes grid pixels to min_nonzero, calls `refine_grid()`. Multi-layer processing is currently sequential (rayon integration TODO)
- `src/refinement/gaussian.rs` — Core blur loop using flat `Vec<f32>` buffers for cache-friendly separable convolution. Horizontal pass on all pixels, vertical pass writes only to grid pixels
- `src/image_io/mod.rs` — Load/save via `image` crate. Supports TIFF/PNG/JPEG. Multi-layer TIFF stack saving not yet implemented
- `src/main.rs` — CLI with clap derive. Parses args into `ProcessingConfig` + optional `GridPattern`
- `src/inpainting/` — Contains Fast Marching Method and directional interpolation (legacy, not used in current pipeline)

### Data flow

`image file → ImageData (ndarray) → detect GridPattern → create zero-pixel mask → init grid to min_nonzero → iterative Gaussian blur (40x) → save`

All image data uses ndarray arrays (`Array2<u16>` for single-layer, `Array3<u16>` for multi-layer) with (row, col) / (y, x) indexing convention.

## CLI Usage

```bash
# Basic (auto-detect grid, MindaGap defaults)
cargo run --release --bin gridline-forge -- input.tiff -o output.tiff

# With options
cargo run --release --bin gridline-forge -- input.tiff -o output.tiff \
  -r 40           # refinement rounds (default: 40)
  -k 5            # kernel size (default: 5, must be odd)
  --benchmark     # show timing breakdown
  -q              # quiet mode
  --x-spacing 2144 --y-spacing 2144  # manual grid spacing
```

## Key Dependencies

- `ndarray` — N-dimensional array storage for image data
- `rustfft` — FFT for grid detection frequency analysis
- `image` — Image I/O (TIFF, PNG, JPEG)
- `rayon` — Parallelism (declared but multi-layer parallel not yet working)
- `clap` (derive) — CLI argument parsing
- `criterion` — Benchmarks (dev-dependency)

## Important Implementation Notes

- **Gaussian kernel**: For ksize <= 7, uses OpenCV's fixed binomial kernels (e.g., `[1/16, 4/16, 6/16, 4/16, 1/16]` for ksize=5), NOT the calculated Gaussian with sigma formula. This matches OpenCV's `getGaussianKernelBitExact` behavior when sigma=0.
- **Boundary handling**: Uses clamp/replicate (vs OpenCV's reflect101). Negligible difference for grid pixels far from image edges.
- **MindaGap comparison reference**: The file `test_data/mindagap.*_gridfilled.tiff` may have been generated with unknown parameters. For fair comparison, regenerate with `pixi run python mindagap.py <input> -s 5 -r 40` from the `../MindaGap/` directory.

## Known Limitations / TODOs

- Multi-layer parallel processing is sequential (rayon + ndarray mutable iteration issue)
- Multi-layer TIFF stack saving not implemented
- 8-bit multi-layer images (`Multi8`) unsupported
- Benchmarks are placeholder only
- Inpainting modules (FMM, directional) are legacy code not used in current pipeline
