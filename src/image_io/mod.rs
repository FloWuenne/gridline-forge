//! Multi-format image I/O with bit depth preservation

use crate::types::{Error, ImageData, ImageFormat, Result};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Array2;
use std::path::Path;

/// Load an image from a file, preserving bit depth
pub fn load_image(path: &Path) -> Result<ImageData> {
    let img = image::open(path)?;

    match img {
        DynamicImage::ImageLuma8(buf) => {
            let (width, height) = buf.dimensions();
            let mut array = Array2::zeros((height as usize, width as usize));

            for y in 0..height as usize {
                for x in 0..width as usize {
                    array[(y, x)] = buf.get_pixel(x as u32, y as u32)[0];
                }
            }

            Ok(ImageData::Gray8(array))
        }
        DynamicImage::ImageLuma16(buf) => {
            let (width, height) = buf.dimensions();
            let mut array = Array2::zeros((height as usize, width as usize));

            for y in 0..height as usize {
                for x in 0..width as usize {
                    array[(y, x)] = buf.get_pixel(x as u32, y as u32)[0];
                }
            }

            Ok(ImageData::Gray16(array))
        }
        DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_) => {
            // Convert to grayscale
            let gray = img.to_luma8();
            let (width, height) = gray.dimensions();
            let mut array = Array2::zeros((height as usize, width as usize));

            for y in 0..height as usize {
                for x in 0..width as usize {
                    array[(y, x)] = gray.get_pixel(x as u32, y as u32)[0];
                }
            }

            Ok(ImageData::Gray8(array))
        }
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) => {
            // Convert to grayscale 16-bit
            let gray = img.to_luma16();
            let (width, height) = gray.dimensions();
            let mut array = Array2::zeros((height as usize, width as usize));

            for y in 0..height as usize {
                for x in 0..width as usize {
                    array[(y, x)] = gray.get_pixel(x as u32, y as u32)[0];
                }
            }

            Ok(ImageData::Gray16(array))
        }
        _ => Err(Error::UnsupportedFormat(format!(
            "Unsupported image format: {:?}",
            img.color()
        ))),
    }
}

/// Save an image to a file with the specified format
pub fn save_image(path: &Path, data: &ImageData, format: ImageFormat) -> Result<()> {
    match data {
        ImageData::Gray8(array) => {
            let (height, width) = (array.nrows() as u32, array.ncols() as u32);
            let mut buf = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);

            for (y, row) in array.rows().into_iter().enumerate() {
                for (x, &pixel) in row.iter().enumerate() {
                    buf.put_pixel(x as u32, y as u32, Luma([pixel]));
                }
            }

            save_buffer_with_format(path, &buf, format)?;
        }
        ImageData::Gray16(array) => {
            let (height, width) = (array.nrows() as u32, array.ncols() as u32);
            let mut buf = ImageBuffer::<Luma<u16>, Vec<u16>>::new(width, height);

            for (y, row) in array.rows().into_iter().enumerate() {
                for (x, &pixel) in row.iter().enumerate() {
                    buf.put_pixel(x as u32, y as u32, Luma([pixel]));
                }
            }

            save_buffer_u16(path, &buf)?;
        }
        ImageData::Multi8(_) => {
            return Err(Error::UnsupportedFormat(
                "Multi-layer 8-bit images not yet supported for saving".to_string(),
            ));
        }
        ImageData::Multi16(_) => {
            return Err(Error::UnsupportedFormat(
                "Multi-layer 16-bit images not yet supported for saving".to_string(),
            ));
        }
    }

    Ok(())
}

/// Helper to save 8-bit image buffer
fn save_buffer_with_format(
    path: &Path,
    buf: &ImageBuffer<Luma<u8>, Vec<u8>>,
    _format: ImageFormat,
) -> Result<()> {
    let dynamic = DynamicImage::ImageLuma8(buf.clone());
    dynamic.save(path)?;
    Ok(())
}

/// Helper to save 16-bit image buffer
fn save_buffer_u16(path: &Path, buf: &ImageBuffer<Luma<u16>, Vec<u16>>) -> Result<()> {
    let dynamic = DynamicImage::ImageLuma16(buf.clone());
    dynamic.save(path)?;
    Ok(())
}

/// Generate output filename based on input and format
pub fn generate_output_path(input: &Path, format: Option<ImageFormat>) -> Result<std::path::PathBuf> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid input filename",
        )))?;

    let parent = input.parent().unwrap_or_else(|| Path::new("."));

    let extension = if let Some(fmt) = format {
        fmt.extension()
    } else {
        input
            .extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No file extension",
            )))?
    };

    Ok(parent.join(format!("{}_gridfilled.{}", stem, extension)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_output_path() {
        let input = Path::new("/tmp/test_image.tif");
        let output = generate_output_path(input, None).unwrap();
        assert_eq!(output, Path::new("/tmp/test_image_gridfilled.tif"));
    }

    #[test]
    fn test_generate_output_path_with_format() {
        let input = Path::new("/tmp/test_image.tif");
        let output = generate_output_path(input, Some(ImageFormat::Png)).unwrap();
        assert_eq!(output, Path::new("/tmp/test_image_gridfilled.png"));
    }
}
