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
cargo test -- --ignored        # Run ignored tests (MindaGap comparison, requires ../mindagap.py)
cargo bench                    # Run benchmarks (criterion)
cargo clippy                   # Lint
cargo fmt                      # Format
```

Integration tests generate synthetic TIFF images in `tests/fixtures/` (auto-cleaned).

## Architecture

Three-stage processing pipeline, orchestrated by `process_image()` in `src/lib.rs`:

1. **Detection** (`src/detection/`) — FFT-based frequency analysis on row/column projections to find grid spacing. Falls back to direct zero-pixel counting if FFT confidence is low. Only operates on `Array2<u16>` (8-bit is upcast).

2. **Inpainting** (`src/inpainting/fast_march.rs`) — Fast Marching Method fills grid pixels from boundary inward using a priority queue ordered by distance. Weighted interpolation with Gaussian-like kernel. Single-pass O(n log n).

3. **Refinement** (`src/refinement/gaussian.rs`) — Separable Gaussian blur applied only to an expanded mask around grid pixels. Iterates 5-10 times (vs 40 in MindaGap) since FMM provides good initial values.

### Key modules

- `src/types.rs` — All core types: `GridPattern`, `ProcessingConfig`, `DetectionConfig`, `ImageData` enum (Gray8/Gray16/Multi8/Multi16 via ndarray), custom `Error`/`Result`
- `src/parallel/mod.rs` — Layer processing orchestration. `process_single_layer()` creates mask → inpaints → refines. Multi-layer processing is currently sequential (rayon integration TODO)
- `src/image_io/mod.rs` — Load/save via `image` crate. Supports TIFF/PNG/JPEG. Multi-layer TIFF stack saving not yet implemented
- `src/main.rs` — CLI with clap derive. Parses args into `ProcessingConfig` + optional `GridPattern`

### Data flow

`image file → ImageData (ndarray) → detect GridPattern → create bool mask → FMM inpaint → Gaussian refine → save`

All image data uses ndarray arrays (`Array2<u16>` for single-layer, `Array3<u16>` for multi-layer) with (row, col) / (y, x) indexing convention.

## Key Dependencies

- `ndarray` — N-dimensional array storage for image data
- `rustfft` — FFT for grid detection frequency analysis
- `image` — Image I/O (TIFF, PNG, JPEG)
- `rayon` — Parallelism (declared but multi-layer parallel not yet working)
- `clap` (derive) — CLI argument parsing
- `criterion` — Benchmarks (dev-dependency)

## Known Limitations / TODOs

- Multi-layer parallel processing is sequential (rayon + ndarray mutable iteration issue)
- Multi-layer TIFF stack saving not implemented
- 8-bit multi-layer images (`Multi8`) unsupported
- Benchmarks are placeholder only
