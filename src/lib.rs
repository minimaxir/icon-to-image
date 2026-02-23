//! # Icon to Image
//!
//! A high-performance library for rendering Font Awesome icons to images.
//!
//! This library provides fast icon rendering using the `ab_glyph` rasterizer
//! with support for supersampling antialiasing, customizable colors and sizes,
//! and output to PNG and WebP formats.
//!
//! ## Features
//!
//! - Embedded Font Awesome assets (no external files needed)
//! - Fast glyph rasterization with `ab_glyph`
//! - Supersampling for high-quality antialiased output
//! - PNG and WebP output formats
//! - Customizable icon and background colors (hex and RGB)
//! - Flexible positioning with anchors and offsets
//! - Separate icon and canvas sizes
//! - Python bindings via PyO3
//!
//! ## Example (Rust)
//!
//! ```no_run
//! use icon_to_image::{IconRenderer, RenderConfig, Color, ImageFormat, encode};
//!
//! // Use embedded assets (recommended) - no external files needed
//! let renderer = IconRenderer::new()?;
//!
//! // Or load from a custom path if you have different font versions:
//! // let renderer = IconRenderer::from_path("./assets")?;
//!
//! let config = RenderConfig::new()
//!     .canvas_size(1024, 1024)
//!     .icon_size(800)
//!     .icon_color(Color::from_hex("#FF5733")?)
//!     .background_color(Color::transparent());
//!
//! let (width, height, pixels) = renderer.render("heart", &config)?;
//! let png_data = encode(&pixels, width, height, ImageFormat::Png)?;
//! std::fs::write("heart.png", png_data)?;
//! # Ok::<(), icon_to_image::IconFontError>(())
//! ```

mod color;
mod css_parser;
pub mod embedded;
mod encoder;
mod error;
mod renderer;

// Re-export public API
pub use color::Color;
pub use css_parser::{CssParser, FontStyle, IconMapping};
pub use encoder::{
    encode, encode_png, encode_png_with_compression, encode_webp, save_to_file, ImageFormat,
    PngCompression,
};
pub use error::{IconFontError, Result};
pub use renderer::{HorizontalAnchor, IconRenderer, RenderConfig, VerticalAnchor};

// Python bindings module (only compiled when the "python" feature is enabled)
#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::*;
