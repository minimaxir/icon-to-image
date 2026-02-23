//! Color parsing and representation.
//!
//! Supports hex colors (#RGB, #RGBA, #RRGGBB, #RRGGBBAA) and RGB tuples.

use crate::error::{IconFontError, Result};

/// RGBA color with values 0-255.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    #[inline]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    #[inline]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[inline]
    pub const fn transparent() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }
    }

    #[inline]
    pub const fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    #[inline]
    pub const fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    /// Parse a color from a hex string (with or without '#').
    ///
    /// # Examples
    ///
    /// ```
    /// use icon_to_image::Color;
    /// let red = Color::from_hex("#FF0000").unwrap();
    /// let blue = Color::from_hex("0000FF").unwrap();
    /// let semi_transparent = Color::from_hex("#FF000080").unwrap();
    /// ```
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.strip_prefix('#').unwrap_or(hex);

        match hex.len() {
            3 => {
                let r = parse_hex_digit(hex.chars().next().unwrap())?;
                let g = parse_hex_digit(hex.chars().nth(1).unwrap())?;
                let b = parse_hex_digit(hex.chars().nth(2).unwrap())?;
                Ok(Self::rgb(r * 17, g * 17, b * 17))
            }
            4 => {
                let r = parse_hex_digit(hex.chars().next().unwrap())?;
                let g = parse_hex_digit(hex.chars().nth(1).unwrap())?;
                let b = parse_hex_digit(hex.chars().nth(2).unwrap())?;
                let a = parse_hex_digit(hex.chars().nth(3).unwrap())?;
                Ok(Self::rgba(r * 17, g * 17, b * 17, a * 17))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| IconFontError::InvalidColor(hex.to_string()))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| IconFontError::InvalidColor(hex.to_string()))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| IconFontError::InvalidColor(hex.to_string()))?;
                Ok(Self::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| IconFontError::InvalidColor(hex.to_string()))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| IconFontError::InvalidColor(hex.to_string()))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| IconFontError::InvalidColor(hex.to_string()))?;
                let a = u8::from_str_radix(&hex[6..8], 16)
                    .map_err(|_| IconFontError::InvalidColor(hex.to_string()))?;
                Ok(Self::rgba(r, g, b, a))
            }
            _ => Err(IconFontError::InvalidColor(format!(
                "Invalid hex length: {}. Expected 3, 4, 6, or 8 characters.",
                hex.len()
            ))),
        }
    }

    #[inline]
    pub const fn to_rgba(&self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    #[inline]
    pub const fn is_transparent(&self) -> bool {
        self.a == 0
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::black()
    }
}

fn parse_hex_digit(c: char) -> Result<u8> {
    match c {
        '0'..='9' => Ok(c as u8 - b'0'),
        'a'..='f' => Ok(c as u8 - b'a' + 10),
        'A'..='F' => Ok(c as u8 - b'A' + 10),
        _ => Err(IconFontError::InvalidColor(format!(
            "Invalid hex digit: {}",
            c
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_parsing() {
        assert_eq!(Color::from_hex("#FF0000").unwrap(), Color::rgb(255, 0, 0));
        assert_eq!(Color::from_hex("00FF00").unwrap(), Color::rgb(0, 255, 0));
        assert_eq!(Color::from_hex("#F00").unwrap(), Color::rgb(255, 0, 0));
        assert_eq!(
            Color::from_hex("#FF000080").unwrap(),
            Color::rgba(255, 0, 0, 128)
        );
    }

    #[test]
    fn test_rgb_constructor() {
        let c = Color::rgb(100, 150, 200);
        assert_eq!(c.r, 100);
        assert_eq!(c.g, 150);
        assert_eq!(c.b, 200);
        assert_eq!(c.a, 255);
    }
}
