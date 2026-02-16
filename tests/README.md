# GridlineForge Integration Tests

This directory contains integration tests that validate GridlineForge's correctness and compare it with the original MindaGap implementation.

## Test Suite Overview

### 1. `test_synthetic_image_basic`
**What it tests:** Basic grid filling functionality

- Generates a synthetic 1000×1000 image with a 100-pixel grid
- Processes it with GridlineForge
- Verifies that 95%+ of grid pixels are filled (no longer zero)
- Validates output dimensions match input

**Run with:**
```bash
cargo test test_synthetic_image_basic
```

### 2. `test_auto_detection`
**What it tests:** FFT-based grid auto-detection

- Generates an 800×800 image with 80-pixel grid spacing
- Runs GridlineForge WITHOUT manual spacing (auto-detect)
- Verifies detected spacing is within ±5 pixels of actual spacing
- Tests the core frequency analysis algorithm

**Run with:**
```bash
cargo test test_auto_detection
```

### 3. `test_compare_with_mindagap` ⚠️
**What it tests:** Output parity with MindaGap

- Generates a 1000×1000 test image
- Processes with BOTH GridlineForge and MindaGap
- Compares outputs using:
  - **RMSE** (Root Mean Square Error): Should be < 50 on 16-bit scale
  - **PSNR** (Peak Signal-to-Noise Ratio): Should be > 40 dB
- Validates that both implementations produce similar results

**Requirements:**
- MindaGap must be available at `../mindagap.py`
- Python 3 with required dependencies (numpy, scipy, etc.)

**Run with:**
```bash
cargo test test_compare_with_mindagap -- --ignored
```

*Note: This test is marked `#[ignore]` because it requires MindaGap to be installed. Use `-- --ignored` to run it explicitly.*

### 4. `test_edge_cases`
**What it tests:** Robustness on unusual inputs

- **Small images**: 100×100 pixels with 10-pixel grid
- **Irregular spacing**: 500×500 pixels with 73-pixel grid (not a nice round number)
- Ensures GridlineForge doesn't crash or fail on edge cases

**Run with:**
```bash
cargo test test_edge_cases
```

## Running All Tests

### Standard Tests (No MindaGap Required)
```bash
cargo test
```

Output:
```
running 3 tests
test test_auto_detection ... ok
test test_edge_cases ... ok
test test_synthetic_image_basic ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Including MindaGap Comparison
```bash
cargo test -- --ignored
```

This runs ONLY the comparison test (requires MindaGap).

### All Tests (Standard + Comparison)
```bash
cargo test -- --include-ignored
```

## Understanding Metrics

### RMSE (Root Mean Square Error)
- Measures pixel-by-pixel difference between images
- **Scale**: 0-65535 (for 16-bit images)
- **Lower is better**: 0 = identical
- **Threshold**: < 50 is considered good match

Formula: `sqrt(mean((img1 - img2)²))`

### PSNR (Peak Signal-to-Noise Ratio)
- Measures quality similarity between images
- **Scale**: 0-∞ dB (decibels)
- **Higher is better**: ∞ = identical
- **Threshold**: > 40 dB is considered good match

Formula: `20 * log10(MAX_VALUE / RMSE)`

### Typical Results

**Expected comparison metrics:**
```
=== Comparison Metrics ===
RMSE: 15.34
PSNR: 64.28 dB
```

These indicate GridlineForge and MindaGap produce **very similar** outputs (PSNR > 60 dB is excellent).

## Adding Your Own Tests

To test with your own panorama images:

1. **Place test image:**
   ```bash
   cp /path/to/your_panorama.tif tests/fixtures/
   ```

2. **Create test function:**
   ```rust
   #[test]
   fn test_my_panorama() {
       let input = PathBuf::from("tests/fixtures/your_panorama.tif");
       let output = PathBuf::from("tests/fixtures/your_panorama_gridfilled.tif");

       let pattern = GridPattern::new(2144, 2144); // Your grid spacing
       let config = ProcessingConfig::default();

       let result = process_image(&input, &output, Some(pattern), &config);
       assert!(result.is_ok());

       // Add your validation checks here
   }
   ```

3. **Run test:**
   ```bash
   cargo test test_my_panorama
   ```

## Test Fixtures

The `tests/fixtures/` directory contains:
- Synthetic test images (generated during tests)
- Optional: Your own test panoramas
- Test outputs (cleaned up after tests)

**Note:** All test files are automatically cleaned up after each test completes.

## Troubleshooting

### Test fails with "mindagap.py not found"
The comparison test requires MindaGap. Either:
- Install MindaGap in the parent directory
- Skip comparison: `cargo test` (without `--ignored`)

### Test fails with RMSE too high
This indicates outputs differ significantly. Check:
- Both use same grid spacing
- Both use compatible parameters (kernel size, iterations)
- Test image isn't corrupted

### Tests are slow
Integration tests process real images. For faster iteration:
- Run specific tests: `cargo test test_synthetic_image_basic`
- Use smaller test images
- Run in release mode: `cargo test --release` (faster execution)

## Performance Testing

To measure GridlineForge vs MindaGap speed:

```bash
# Time GridlineForge
time cargo run --release -- tests/fixtures/test_comparison.tif

# Time MindaGap
time python3 ../mindagap.py tests/fixtures/test_comparison.tif 5 40 -xt 100 -yt 100
```

Expected speedup: **10-100× faster** depending on image size.
