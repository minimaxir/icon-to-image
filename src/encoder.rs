//! Image encoding utilities for PNG and WebP output.

use crate::error::{IconFontError, Result};
use image::ImageEncoder;
use std::io::Cursor;
use std::path::Path;

/// Image output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    #[default]
    Png,
    WebP,
}

impl ImageFormat {
    pub const fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::WebP => "webp",
        }
    }

    pub const fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::WebP => "image/webp",
        }
    }
}

/// PNG compression presets for balancing encode speed and output size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PngCompression {
    Fast,
    Default,
    #[default]
    Best,
}

impl PngCompression {
    #[inline]
    const fn settings(
        self,
    ) -> (
        image::codecs::png::CompressionType,
        image::codecs::png::FilterType,
    ) {
        match self {
            PngCompression::Fast => (
                image::codecs::png::CompressionType::Fast,
                image::codecs::png::FilterType::NoFilter,
            ),
            PngCompression::Default => (
                image::codecs::png::CompressionType::Default,
                image::codecs::png::FilterType::Adaptive,
            ),
            PngCompression::Best => (
                image::codecs::png::CompressionType::Best,
                image::codecs::png::FilterType::Adaptive,
            ),
        }
    }
}

/// Encode RGBA pixels to PNG with best compression.
pub fn encode_png(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    encode_png_with_compression(pixels, width, height, PngCompression::Best)
}

/// Encode RGBA pixels to PNG with a configurable compression preset.
pub fn encode_png_with_compression(
    pixels: &[u8],
    width: u32,
    height: u32,
    compression: PngCompression,
) -> Result<Vec<u8>> {
    let mut output = Cursor::new(Vec::new());
    let (compression_type, filter_type) = compression.settings();

    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        &mut output,
        compression_type,
        filter_type,
    );

    encoder
        .write_image(pixels, width, height, image::ExtendedColorType::Rgba8)
        .map_err(|e| IconFontError::ImageEncodingError(format!("PNG encoding failed: {}", e)))?;

    Ok(output.into_inner())
}

/// Encode RGBA pixels to lossless WebP.
pub fn encode_webp(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut output = Cursor::new(Vec::new());
    let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut output);

    encoder
        .write_image(pixels, width, height, image::ExtendedColorType::Rgba8)
        .map_err(|e| IconFontError::ImageEncodingError(format!("WebP encoding failed: {}", e)))?;

    Ok(output.into_inner())
}

/// Encode pixels to the specified format.
pub fn encode(pixels: &[u8], width: u32, height: u32, format: ImageFormat) -> Result<Vec<u8>> {
    match format {
        ImageFormat::Png => encode_png(pixels, width, height),
        ImageFormat::WebP => encode_webp(pixels, width, height),
    }
}

/// Save pixels to a file (format determined by extension, defaults to PNG).
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
        None => ImageFormat::Png,
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
    fn test_png_compression_settings() {
        let (fast_c, fast_f) = PngCompression::Fast.settings();
        let (default_c, default_f) = PngCompression::Default.settings();
        let (best_c, best_f) = PngCompression::Best.settings();

        assert_eq!(fast_c, image::codecs::png::CompressionType::Fast);
        assert_eq!(fast_f, image::codecs::png::FilterType::NoFilter);
        assert_eq!(default_c, image::codecs::png::CompressionType::Default);
        assert_eq!(default_f, image::codecs::png::FilterType::Adaptive);
        assert_eq!(best_c, image::codecs::png::CompressionType::Best);
        assert_eq!(best_f, image::codecs::png::FilterType::Adaptive);
    }

    #[test]
    fn test_png_compression_default_is_best() {
        assert_eq!(PngCompression::default(), PngCompression::Best);
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
