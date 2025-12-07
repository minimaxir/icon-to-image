//! Error types for icon font rendering.
//!
//! This module defines custom error types using `thiserror` for clear,
//! type-safe error handling throughout the library.

use thiserror::Error;

/// Errors that can occur during icon font rendering operations.
#[derive(Error, Debug)]
pub enum IconFontError {
    /// The requested icon name was not found in the CSS mappings.
    #[error("Icon '{0}' not found in font mappings")]
    IconNotFound(String),

    /// The font file could not be loaded.
    #[error("Failed to load font: {0}")]
    FontLoadError(String),

    /// The CSS file could not be parsed.
    #[error("Failed to parse CSS: {0}")]
    CssParseError(String),

    /// Invalid color format was provided.
    #[error("Invalid color format: {0}")]
    InvalidColor(String),

    /// Image encoding failed.
    #[error("Image encoding failed: {0}")]
    ImageEncodingError(String),

    /// Invalid dimensions were provided.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// IO error occurred.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias for operations that can fail with `IconFontError`.
pub type Result<T> = std::result::Result<T, IconFontError>;
