//! Core types and structures for GridlineForge

use ndarray::{Array2, Array3};
use thiserror::Error;

/// Grid pattern detection result
#[derive(Debug, Clone)]
pub struct GridPattern {
    /// Distance between vertical grid lines (pixels)
    pub x_spacing: u32,
    /// Distance between horizontal grid lines (pixels)
    pub y_spacing: u32,
    /// Offset of first vertical line from left edge
    pub x_offset: u32,
    /// Offset of first horizontal line from top edge
    pub y_offset: u32,
    /// Width of grid lines (typically 1-3 pixels)
    pub grid_width: u32,
    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
}

impl GridPattern {
    /// Create a new grid pattern with default values
    pub fn new(x_spacing: u32, y_spacing: u32) -> Self {
        Self {
            x_spacing,
            y_spacing,
            x_offset: 0,
            y_offset: 0,
            grid_width: 1,
            confidence: 1.0,
        }
    }

    /// Check if a pixel coordinate is on a grid line
    pub fn is_grid_pixel(&self, x: u32, y: u32) -> bool {
        let x_on_grid = if self.x_spacing > 0 {
            let x_rel = (x + self.x_spacing - self.x_offset) % self.x_spacing;
            x_rel < self.grid_width
        } else {
            false
        };

        let y_on_grid = if self.y_spacing > 0 {
            let y_rel = (y + self.y_spacing - self.y_offset) % self.y_spacing;
            y_rel < self.grid_width
        } else {
            false
        };

        x_on_grid || y_on_grid
    }
}

/// Grid detection configuration
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Minimum expected grid spacing (pixels)
    pub min_spacing: u32,
    /// Maximum expected grid spacing (pixels)
    pub max_spacing: u32,
    /// Minimum confidence threshold for accepting detection
    pub confidence_threshold: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_spacing: 50,
            max_spacing: 10000,
            confidence_threshold: 0.7,
        }
    }
}

/// Processing configuration
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Gaussian kernel size for refinement (must be odd)
    pub kernel_size: usize,
    /// Number of refinement iterations
    pub refinement_rounds: usize,
    /// Radius for fast marching method
    pub fmm_radius: usize,
    /// Show progress bars
    pub show_progress: bool,
    /// Enable detailed timing information
    pub benchmark: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            kernel_size: 5,
            refinement_rounds: 5,
            fmm_radius: 3,
            show_progress: true,
            benchmark: false,
        }
    }
}

/// Multi-format image data representation
#[derive(Debug)]
pub enum ImageData {
    /// 8-bit grayscale
    Gray8(Array2<u8>),
    /// 16-bit grayscale
    Gray16(Array2<u16>),
    /// Multi-layer 8-bit (e.g., TIFF stack)
    Multi8(Array3<u8>),
    /// Multi-layer 16-bit (e.g., TIFF stack)
    Multi16(Array3<u16>),
}

impl ImageData {
    /// Get the dimensions (width, height) of the image
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            ImageData::Gray8(arr) => (arr.ncols() as u32, arr.nrows() as u32),
            ImageData::Gray16(arr) => (arr.ncols() as u32, arr.nrows() as u32),
            ImageData::Multi8(arr) => (arr.shape()[2] as u32, arr.shape()[1] as u32),
            ImageData::Multi16(arr) => (arr.shape()[2] as u32, arr.shape()[1] as u32),
        }
    }

    /// Get the number of layers (1 for single-layer images)
    pub fn num_layers(&self) -> usize {
        match self {
            ImageData::Gray8(_) | ImageData::Gray16(_) => 1,
            ImageData::Multi8(arr) => arr.shape()[0],
            ImageData::Multi16(arr) => arr.shape()[0],
        }
    }

    /// Check if this is a 16-bit image
    pub fn is_16bit(&self) -> bool {
        matches!(self, ImageData::Gray16(_) | ImageData::Multi16(_))
    }
}

/// Image output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Tiff,
    Png,
    Jpeg,
}

impl ImageFormat {
    /// Parse format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "tif" | "tiff" => Some(ImageFormat::Tiff),
            "png" => Some(ImageFormat::Png),
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            _ => None,
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Tiff => "tif",
            ImageFormat::Png => "png",
            ImageFormat::Jpeg => "jpg",
        }
    }
}

/// GridlineForge error types
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Grid detection failed: {0}")]
    Detection(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

pub type Result<T> = std::result::Result<T, Error>;
