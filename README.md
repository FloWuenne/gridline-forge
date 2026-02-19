# GridlineForge

High-performance grid line removal for panorama microscopy images, written in Rust.

## Overview

GridlineForge is a fast, efficient tool for detecting and filling gridlines in panoramic images. It implements the same iterative Gaussian blur algorithm as [MindaGap](https://github.com/remisalmon/MindaGap) but in optimized Rust, offering significant performance improvements.

### Key Features

- **MindaGap-compatible**: Same algorithm, same defaults (ksize=5, 40 rounds), matching output quality
- **Auto-detection**: FFT-based frequency analysis automatically detects grid spacing
- **Fast processing**: Optimized separable convolution with cache-friendly memory access
- **Multi-format support**: TIFF (8/16-bit), PNG, JPEG
- **Multi-layer support**: Processes multi-layer TIFF stacks
- **CLI interface**: Easy-to-use command-line tool

## Installation

### From Source

Requires [Rust](https://www.rust-lang.org/tools/install) (1.70+).

```bash
git clone https://github.com/FloWuenne/gridline-forge.git
cd gridline-forge
cargo install --path .
```

The binary will be available as `gridline-forge`.

### Docker

Pre-built images for `linux/amd64` and `linux/arm64` are available on GHCR:

```bash
docker pull ghcr.io/FloWuenne/gridline-forge:latest
```

Run with a volume mount to access your images:

```bash
docker run --rm -v $(pwd):/data ghcr.io/FloWuenne/gridline-forge:latest /data/input.tif -o /data/output.tif
```

## Usage

### Basic Usage (Auto-detection)

```bash
gridline-forge input.tif
```

This will:
1. Auto-detect the grid pattern using FFT analysis
2. Mask zero-valued pixels and initialize them to the minimum non-zero value
3. Apply 40 rounds of iterative Gaussian blur (matching MindaGap)
4. Save output as `input_gridfilled.tif`

### Manual Grid Spacing

If auto-detection fails or you want to override it:

```bash
gridline-forge input.tif --x-spacing 2144 --y-spacing 2144
```

### Advanced Options

```bash
gridline-forge input.tif \
  --output output.tif \
  --kernel-size 5 \
  --refinement-rounds 40 \
  --format png \
  --benchmark
```

### Command-Line Options

```
USAGE:
    gridline-forge [OPTIONS] <INPUT>

ARGS:
    <INPUT>    Input image file (TIFF, PNG, JPEG)

OPTIONS:
    -o, --output <FILE>           Output file (default: INPUT_gridfilled.EXT)
    -k, --kernel-size <SIZE>      Gaussian kernel size (default: 5, must be odd)
    -r, --refinement-rounds <N>   Gaussian blur iterations (default: 40)
        --x-spacing <PIXELS>      Manual X grid spacing (auto-detect if not set)
        --y-spacing <PIXELS>      Manual Y grid spacing (auto-detect if not set)
        --no-auto-detect          Disable auto-detection, use zero-valued pixels
    -f, --format <FORMAT>         Output format: tiff, png, jpeg
    -q, --quiet                   Suppress progress output
        --benchmark               Show detailed timing information
    -h, --help                    Print help
    -V, --version                 Print version
```

## Algorithm

GridlineForge uses the same algorithm as MindaGap, implemented in optimized Rust:

### 1. Grid Detection (FFT-based)

- Projects image along rows and columns
- Applies FFT to find dominant periodic frequencies
- Validates detected pattern against actual zero-valued pixels
- Falls back to direct zero-detection if confidence is low

### 2. Iterative Gaussian Blur (MindaGap-compatible)

Matches MindaGap's `fill_grids()` algorithm exactly:

1. **Mask**: Identify all zero-valued pixels as grid pixels
2. **Initialize**: Set grid pixels to the minimum non-zero value in the image
3. **Iterate** (40 rounds by default):
   - Blur the entire image with a Gaussian kernel (separable convolution)
   - Copy blurred values back only to grid pixels (non-grid pixels stay unchanged)

Implementation details:
- **Separable convolution**: Horizontal + vertical passes instead of 2D kernel (same result, faster)
- **OpenCV-matching kernels**: Uses fixed binomial kernels for ksize <= 7 (e.g., `[1/16, 4/16, 6/16, 4/16, 1/16]` for ksize=5), matching OpenCV's `getGaussianKernel` behavior when sigma=0
- **Cache-friendly**: Flat `Vec<f32>` row-major buffers for sequential memory access
- **Per-iteration rounding**: Matches MindaGap's uint16 storage behavior

## Limitations

- Multi-layer TIFF stack saving not yet implemented (single-layer works)
- Multi-layer processing is currently sequential (rayon integration TODO)
- 8-bit multi-layer images (`Multi8`) not yet supported
- Grid lines must be reasonably regular and periodic

## Roadmap

- [ ] Fix parallel layer processing with Rayon
- [ ] Implement multi-layer TIFF stack saving
- [ ] Batch processing mode for multiple files
- [ ] Python bindings (PyO3)

## Comparison with MindaGap

GridlineForge implements the same algorithm as MindaGap with matching defaults. Migration is straightforward:

```bash
# MindaGap
python mindagap.py input.tif -s 5 -r 40 -xt 2144 -yt 2144

# GridlineForge equivalent (same defaults, so flags are optional)
gridline-forge input.tif --x-spacing 2144 --y-spacing 2144
```

## Development

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo bench
```

### Building Documentation

```bash
cargo doc --open
```

## License

MIT License - see [LICENSE](../LICENSE) file for details.

## Contributing

Contributions welcome! Please open an issue or pull request.

## Credits

- Original MindaGap implementation: [MindaGap](https://github.com/remisalmon/MindaGap) by Ricardo Guerreiro (Resolve Biosciences)
- Rust image processing: [image-rs](https://github.com/image-rs/image)

## Citation

If you use GridlineForge in your research, please cite:

```
@software{gridlineforge,
  title = {GridlineForge: High-performance grid line removal for panorama images},
  author = {GridlineForge Contributors},
  year = {2026},
  url = {https://github.com/FloWuenne/gridline-forge}
}
```
