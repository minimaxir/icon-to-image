//! Image encoding utilities for PNG and WebP output.
//!
//! Provides optimized encoding for icon images with configurable quality.

use crate::error::{IconFontError, Result};
use image::{ImageBuffer, ImageEncoder, Rgba};
use std::io::Cursor;
use std::path::Path;

/// Image output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    /// PNG format (lossless)
    #[default]
    Png,
    /// WebP format (can be lossy or lossless)
    WebP,
}

impl ImageFormat {
    /// Get the file extension for this format.
    pub const fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::WebP => "webp",
        }
    }

    /// Get the MIME type for this format.
    pub const fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::WebP => "image/webp",
        }
    }
}

/// Encode RGBA pixels to PNG format.
///
/// # Arguments
///
/// * `pixels` - RGBA pixel data
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// Encoded PNG data as bytes.
///
/// # Errors
///
/// Returns error if encoding fails.
pub fn encode_png(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut output = Cursor::new(Vec::new());

    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        &mut output,
        image::codecs::png::CompressionType::Best,
        image::codecs::png::FilterType::Adaptive,
    );

    encoder
        .write_image(pixels, width, height, image::ExtendedColorType::Rgba8)
        .map_err(|e| IconFontError::ImageEncodingError(format!("PNG encoding failed: {}", e)))?;

    Ok(output.into_inner())
}

/// Encode RGBA pixels to WebP format.
///
/// # Arguments
///
/// * `pixels` - RGBA pixel data
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// Encoded WebP data as bytes.
///
/// # Errors
///
/// Returns error if encoding fails.
pub fn encode_webp(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, pixels.to_vec())
        .ok_or_else(|| {
            IconFontError::ImageEncodingError("Failed to create image buffer".to_string())
        })?;

    let mut output = Cursor::new(Vec::new());

    // Use lossless WebP for icons (better quality for vector-like graphics)
    let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut output);

    encoder
        .write_image(img.as_raw(), width, height, image::ExtendedColorType::Rgba8)
        .map_err(|e| IconFontError::ImageEncodingError(format!("WebP encoding failed: {}", e)))?;

    Ok(output.into_inner())
}

/// Encode pixels to the specified format.
///
/// # Arguments
///
/// * `pixels` - RGBA pixel data
/// * `width` - Image width
/// * `height` - Image height
/// * `format` - Output format
///
/// # Returns
///
/// Encoded image data as bytes.
pub fn encode(pixels: &[u8], width: u32, height: u32, format: ImageFormat) -> Result<Vec<u8>> {
    match format {
        ImageFormat::Png => encode_png(pixels, width, height),
        ImageFormat::WebP => encode_webp(pixels, width, height),
    }
}

/// Save pixels to a file with format determined by extension.
///
/// # Arguments
///
/// * `pixels` - RGBA pixel data
/// * `width` - Image width
/// * `height` - Image height
/// * `path` - Output file path (extension determines format)
///
/// # Returns
///
/// Ok if saved successfully.
///
/// # Errors
///
/// Returns error if encoding or file writing fails.
pub fn save_to_file<P: AsRef<Path>>(pixels: &[u8], width: u32, height: u32, path: P) -> Result<()> {
    let path = path.as_ref();
    let format = match path.extension().and_then(|e| e.to_str()) {
        Some("png") => ImageFormat::Png,
        Some("webp") => ImageFormat::WebP,
        Some(ext) => {
            return Err(IconFontError::ImageEncodingError(format!(
                "Unsupported format: {}",
                ext
            )))
        }
        None => ImageFormat::Png, // Default to PNG
    };

    let data = encode(pixels, width, height, format)?;
    std::fs::write(path, data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_extension() {
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::WebP.extension(), "webp");
    }

    #[test]
    fn test_format_mime_type() {
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::WebP.mime_type(), "image/webp");
    }

    #[test]
    fn test_encode_small_image() {
        // Create a 2x2 red image
        let pixels = vec![
            255, 0, 0, 255, // red
            255, 0, 0, 255, // red
            255, 0, 0, 255, // red
            255, 0, 0, 255, // red
        ];

        let png_data = encode_png(&pixels, 2, 2).unwrap();
        assert!(!png_data.is_empty());
        // PNG magic bytes
        assert_eq!(&png_data[0..4], &[0x89, 0x50, 0x4E, 0x47]);

        let webp_data = encode_webp(&pixels, 2, 2).unwrap();
        assert!(!webp_data.is_empty());
        // WebP magic bytes (RIFF....WEBP)
        assert_eq!(&webp_data[0..4], b"RIFF");
    }
}
