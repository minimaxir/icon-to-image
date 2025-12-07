//! CSS parser for Font Awesome icon mappings.
//!
//! Extracts icon name to Unicode codepoint mappings from Font Awesome CSS files.
//! The CSS format uses CSS custom properties: `.fa-{name}{--fa:"\xxxx"}`

use crate::error::{IconFontError, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

/// Represents the font style/weight for an icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FontStyle {
    /// Solid icons (weight 900) - uses fa-solid.otf
    #[default]
    Solid,
    /// Regular icons (weight 400) - uses fa-regular.otf
    Regular,
    /// Brand icons (weight 400) - uses fa-brands.otf
    Brands,
}

/// Icon mapping containing the Unicode codepoint and font style.
#[derive(Debug, Clone)]
pub struct IconMapping {
    /// The Unicode codepoint for this icon (e.g., 0xf004 for heart)
    pub codepoint: char,
    /// The font style to use for rendering
    pub style: FontStyle,
}

/// Parser for Font Awesome CSS files.
///
/// Extracts icon mappings from the CSS custom property declarations.
#[derive(Debug)]
pub struct CssParser {
    /// Map of icon names (without "fa-" prefix) to their mappings
    icons: HashMap<String, IconMapping>,
}

// Regex to match icon rule blocks with their codepoint value.
// Captures the entire selector group and the codepoint value.
// Example: ".fa-circle-xmark,.fa-times-circle{--fa:"\f057"}"
static RULE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"([^{}]+)\{--fa:"([^"]+)"\}"#).expect("Invalid rule regex pattern"));

// Regex to extract individual icon names from a selector group.
// Matches ".fa-NAME" patterns and captures just the NAME part.
static NAME_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"\.fa-([a-zA-Z0-9_-]+)"#).expect("Invalid name regex pattern"));

impl CssParser {
    /// Parse a CSS string and extract icon mappings.
    ///
    /// # Arguments
    ///
    /// * `css_content` - The CSS file content as a string
    ///
    /// # Returns
    ///
    /// A `CssParser` instance with all parsed icon mappings.
    ///
    /// # Errors
    ///
    /// Returns `CssParseError` if the CSS cannot be parsed.
    pub fn parse(css_content: &str) -> Result<Self> {
        let mut icons = HashMap::new();
        let mut brand_icons = std::collections::HashSet::new();

        // First pass: identify brand icons by finding the brands section
        // Brand icons are declared after `.fa-brands,.fa-classic.fa-brands,.fab{`
        let brand_section_start = css_content.find(".fa-brands,.fa-classic.fa-brands,.fab{");
        let brand_section_end = css_content
            .find(":host,:root{--fa-font-regular")
            .or_else(|| css_content.find(".far{"));

        if let (Some(start), Some(end)) = (brand_section_start, brand_section_end) {
            let brand_section = &css_content[start..end];
            // Extract all icon names from the brands section
            for rule_cap in RULE_REGEX.captures_iter(brand_section) {
                let selectors = rule_cap.get(1).map(|m| m.as_str()).unwrap_or_default();
                // Find all .fa-NAME patterns in the selector group
                for name_cap in NAME_REGEX.captures_iter(selectors) {
                    let name = name_cap.get(1).map(|m| m.as_str()).unwrap_or_default();
                    if !is_utility_class(name) {
                        brand_icons.insert(name.to_lowercase());
                    }
                }
            }
        }

        // Second pass: parse all icon definitions, capturing all aliases
        for rule_cap in RULE_REGEX.captures_iter(css_content) {
            let selectors = rule_cap.get(1).map(|m| m.as_str()).unwrap_or_default();
            let value = rule_cap.get(2).map(|m| m.as_str()).unwrap_or_default();

            // Parse the codepoint from the value
            if let Some(codepoint) = parse_codepoint(value) {
                // Extract all icon names from this selector group (handles aliases)
                for name_cap in NAME_REGEX.captures_iter(selectors) {
                    let name = name_cap.get(1).map(|m| m.as_str()).unwrap_or_default();

                    // Skip utility classes (sizes, animations, etc.)
                    if is_utility_class(name) {
                        continue;
                    }

                    let name_lower = name.to_lowercase();
                    let style = if brand_icons.contains(&name_lower) {
                        FontStyle::Brands
                    } else {
                        // Default to Solid for non-brand icons
                        FontStyle::Solid
                    };

                    icons.insert(name_lower, IconMapping { codepoint, style });
                }
            }
        }

        if icons.is_empty() {
            return Err(IconFontError::CssParseError(
                "No icon mappings found in CSS".to_string(),
            ));
        }

        Ok(Self { icons })
    }

    /// Look up an icon by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The icon name (with or without "fa-" prefix)
    ///
    /// # Returns
    ///
    /// The `IconMapping` if found, or `None`.
    pub fn get_icon(&self, name: &str) -> Option<&IconMapping> {
        // Strip "fa-" prefix if present
        let name = name.strip_prefix("fa-").unwrap_or(name);
        self.icons.get(&name.to_lowercase())
    }

    /// Look up an icon by name with explicit style override.
    ///
    /// # Arguments
    ///
    /// * `name` - The icon name (with or without "fa-" prefix)
    /// * `style` - The font style to use (overrides default)
    ///
    /// # Returns
    ///
    /// A modified `IconMapping` with the specified style, or `None` if icon not found.
    pub fn get_icon_with_style(&self, name: &str, style: FontStyle) -> Option<IconMapping> {
        self.get_icon(name).map(|mapping| IconMapping {
            codepoint: mapping.codepoint,
            style,
        })
    }

    /// Get the total number of parsed icons.
    pub fn icon_count(&self) -> usize {
        self.icons.len()
    }

    /// Check if an icon exists.
    pub fn has_icon(&self, name: &str) -> bool {
        self.get_icon(name).is_some()
    }

    /// List all available icon names.
    pub fn list_icons(&self) -> Vec<&str> {
        self.icons.keys().map(|s| s.as_str()).collect()
    }
}

/// Check if a class name is a utility class (not an icon).
fn is_utility_class(name: &str) -> bool {
    // Size classes
    if name.ends_with('x')
        && name
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
    {
        return true;
    }

    matches!(
        name,
        "1" | "2"
            | "3"
            | "4"
            | "5"
            | "6"
            | "7"
            | "8"
            | "9"
            | "10"
            | "2xs"
            | "xs"
            | "sm"
            | "lg"
            | "xl"
            | "2xl"
            | "fw"
            | "ul"
            | "li"
            | "border"
            | "inverse"
            | "pull-left"
            | "pull-right"
            | "pull-start"
            | "pull-end"
            | "beat"
            | "bounce"
            | "fade"
            | "beat-fade"
            | "flip"
            | "shake"
            | "spin"
            | "spin-pulse"
            | "pulse"
            | "spin-reverse"
            | "rotate-90"
            | "rotate-180"
            | "rotate-270"
            | "rotate-by"
            | "flip-horizontal"
            | "flip-vertical"
            | "flip-both"
            | "stack"
            | "stack-1x"
            | "stack-2x"
            | "width-auto"
            | "width-fixed"
            | "brands"
            | "classic"
            | "regular"
            | "solid"
    )
}

/// Parse a Unicode codepoint from a CSS value.
///
/// Handles formats like:
/// - `\f004` - hex escape
/// - `\e005` - hex escape
/// - `A` - literal character
/// - `\!` - escaped literal
fn parse_codepoint(value: &str) -> Option<char> {
    let value = value.trim();

    if value.is_empty() {
        return None;
    }

    // Handle escaped hex codepoints (e.g., "\f004")
    if let Some(hex_part) = value.strip_prefix('\\') {
        // Check if it's a hex codepoint or an escaped literal
        if hex_part.len() >= 2 && hex_part.chars().all(|c| c.is_ascii_hexdigit() || c == ' ') {
            // It's a hex codepoint, possibly with trailing space
            let hex_clean = hex_part.trim();
            if let Ok(code) = u32::from_str_radix(hex_clean, 16) {
                return char::from_u32(code);
            }
        }
        // It's an escaped literal (e.g., "\!" -> "!")
        return hex_part.chars().next();
    }

    // Handle literal characters (e.g., "A", "!")
    if value.len() == 1 {
        return value.chars().next();
    }

    // Handle multi-byte Unicode characters
    value.chars().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_codepoint() {
        assert_eq!(parse_codepoint("\\f004"), Some('\u{f004}'));
        assert_eq!(parse_codepoint("\\e005"), Some('\u{e005}'));
        assert_eq!(parse_codepoint("A"), Some('A'));
        assert_eq!(parse_codepoint("\\!"), Some('!'));
        assert_eq!(parse_codepoint("\\30 "), Some('0')); // "0" character
    }

    #[test]
    fn test_is_utility_class() {
        assert!(is_utility_class("1x"));
        assert!(is_utility_class("2xl"));
        assert!(is_utility_class("spin"));
        assert!(is_utility_class("brands"));
        assert!(!is_utility_class("heart"));
        assert!(!is_utility_class("github"));
    }

    #[test]
    fn test_parse_icon_aliases() {
        // Test CSS with aliased icon names (multiple selectors pointing to same codepoint)
        let css = r#".fa-circle-xmark,.fa-times-circle,.fa-xmark-circle{--fa:"\f057"}"#;
        let parser = CssParser::parse(css).unwrap();

        // All three aliases should resolve to the same codepoint
        assert!(parser.has_icon("circle-xmark"));
        assert!(parser.has_icon("times-circle"));
        assert!(parser.has_icon("xmark-circle"));

        let mapping1 = parser.get_icon("circle-xmark").unwrap();
        let mapping2 = parser.get_icon("times-circle").unwrap();
        let mapping3 = parser.get_icon("xmark-circle").unwrap();

        assert_eq!(mapping1.codepoint, '\u{f057}');
        assert_eq!(mapping2.codepoint, '\u{f057}');
        assert_eq!(mapping3.codepoint, '\u{f057}');
    }
}
