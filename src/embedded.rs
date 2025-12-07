//! Embedded Font Awesome assets.
//!
//! This module provides compile-time embedded font and CSS data for better
//! portability. The assets are baked directly into the binary using Rust's
//! `include_bytes!` and `include_str!` macros, eliminating the need to
//! distribute separate asset files.

/// Font Awesome Solid font data (fa-solid.otf).
///
/// Contains solid-style icons with weight 900.
pub static FONT_SOLID: &[u8] = include_bytes!("../assets/fa-solid.otf");

/// Font Awesome Regular font data (fa-regular.otf).
///
/// Contains regular-style icons with weight 400.
pub static FONT_REGULAR: &[u8] = include_bytes!("../assets/fa-regular.otf");

/// Font Awesome Brands font data (fa-brands.otf).
///
/// Contains brand/logo icons with weight 400.
pub static FONT_BRANDS: &[u8] = include_bytes!("../assets/fa-brands.otf");

/// Font Awesome CSS containing icon name to codepoint mappings.
///
/// This CSS file is parsed at runtime to build a lookup table from
/// icon names (like "heart", "github") to their Unicode codepoints.
pub static FONTAWESOME_CSS: &str = include_str!("../assets/fontawesome.css");
