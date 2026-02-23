//! CSS parser for Font Awesome icon mappings.
//!
//! Extracts icon name to Unicode codepoint mappings from Font Awesome CSS files.

use crate::error::{IconFontError, Result};
use rustc_hash::FxHashMap;
use std::collections::HashSet;

/// Represents the font style/weight for an icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FontStyle {
    #[default]
    Solid,
    Regular,
    Brands,
}

/// Icon mapping containing the Unicode codepoint and font style.
#[derive(Debug, Clone)]
pub struct IconMapping {
    pub codepoint: char,
    pub style: FontStyle,
}

/// Parser for Font Awesome CSS files.
#[derive(Debug)]
pub struct CssParser {
    icons: FxHashMap<String, IconMapping>,
    icon_names: Vec<String>,
}

const ICON_RULE_MARKER: &str = "{--fa:\"";

impl CssParser {
    /// Parse a CSS string and extract icon mappings.
    ///
    /// # Errors
    ///
    /// Returns `CssParseError` if no icons found.
    pub fn parse(css_content: &str) -> Result<Self> {
        let icon_rule_count = css_content.matches(ICON_RULE_MARKER).count();
        let mut icons = FxHashMap::with_capacity_and_hasher(
            icon_rule_count.saturating_mul(2),
            Default::default(),
        );
        let mut brand_icons = HashSet::with_capacity(icon_rule_count / 4);

        // First pass: identify brand icons from the brands CSS section
        let brand_section_start = css_content.find(".fa-brands,.fa-classic.fa-brands,.fab{");
        let brand_section_end = css_content
            .find(":host,:root{--fa-font-regular")
            .or_else(|| css_content.find(".far{"));

        if let (Some(start), Some(end)) = (brand_section_start, brand_section_end) {
            let brand_section = &css_content[start..end];
            scan_icon_rules(brand_section, |selectors, _value| {
                extract_fa_names(selectors, |name| {
                    if !is_utility_class(name) {
                        brand_icons.insert(normalize_ascii_lower(name));
                    }
                });
            });
        }

        // Second pass: parse all icon definitions
        scan_icon_rules(css_content, |selectors, value| {
            if let Some(codepoint) = parse_codepoint(value) {
                extract_fa_names(selectors, |name| {
                    if is_utility_class(name) {
                        return;
                    }

                    let name_lower = normalize_ascii_lower(name);
                    let style = if brand_icons.contains(&name_lower) {
                        FontStyle::Brands
                    } else {
                        FontStyle::Solid
                    };

                    icons.insert(name_lower, IconMapping { codepoint, style });
                });
            }
        });

        if icons.is_empty() {
            return Err(IconFontError::CssParseError(
                "No icon mappings found in CSS".to_string(),
            ));
        }

        let icon_names = icons.keys().cloned().collect();

        Ok(Self { icons, icon_names })
    }

    /// Look up an icon by name (with or without "fa-" prefix).
    #[inline]
    pub fn get_icon(&self, name: &str) -> Option<&IconMapping> {
        let name = name.strip_prefix("fa-").unwrap_or(name);
        if name.bytes().all(|byte| !byte.is_ascii_uppercase()) {
            self.icons.get(name)
        } else {
            let lower = name.to_ascii_lowercase();
            self.icons.get(lower.as_str())
        }
    }

    /// Look up an icon with explicit style override.
    pub fn get_icon_with_style(&self, name: &str, style: FontStyle) -> Option<IconMapping> {
        self.get_icon(name).map(|mapping| IconMapping {
            codepoint: mapping.codepoint,
            style,
        })
    }

    pub fn icon_count(&self) -> usize {
        self.icons.len()
    }

    pub fn has_icon(&self, name: &str) -> bool {
        self.get_icon(name).is_some()
    }

    pub fn list_icons(&self) -> Vec<&str> {
        self.icon_names.iter().map(String::as_str).collect()
    }
}

/// Scan icon rules matching `.fa-name{--fa:"value"}` and call `on_rule(selectors, value)`.
fn scan_icon_rules(css: &str, mut on_rule: impl FnMut(&str, &str)) {
    let mut scan_start = 0usize;
    let marker_len = ICON_RULE_MARKER.len();

    while let Some(marker_rel) = css[scan_start..].find(ICON_RULE_MARKER) {
        let marker = scan_start + marker_rel;
        let selector_start = css[scan_start..marker]
            .rfind('}')
            .map(|idx| scan_start + idx + 1)
            .unwrap_or(scan_start);
        let selectors = css[selector_start..marker].trim();

        let value_start = marker + marker_len;
        let Some(quote_rel) = css[value_start..].find('"') else {
            break;
        };
        let value_end = value_start + quote_rel;
        let value = &css[value_start..value_end];

        on_rule(selectors, value);

        let close_rel = css[value_end..].find('}').unwrap_or(0);
        scan_start = value_end + close_rel + 1;
    }
}

/// Extract `.fa-*` selector names from a selector list.
fn extract_fa_names(selectors: &str, mut on_name: impl FnMut(&str)) {
    let bytes = selectors.as_bytes();
    let mut idx = 0usize;

    while let Some(rel) = selectors[idx..].find(".fa-") {
        idx += rel + 4;
        let start = idx;
        while idx < bytes.len() {
            let byte = bytes[idx];
            if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' {
                idx += 1;
            } else {
                break;
            }
        }

        if idx > start {
            on_name(&selectors[start..idx]);
        }
    }
}

#[inline]
fn normalize_ascii_lower(name: &str) -> String {
    if name.bytes().any(|byte| byte.is_ascii_uppercase()) {
        name.to_ascii_lowercase()
    } else {
        name.to_string()
    }
}

/// Filter out Font Awesome utility classes (sizes, animations, transforms).
fn is_utility_class(name: &str) -> bool {
    matches!(
        name,
        // Size multiplier classes (1x through 10x)
        "1x" | "2x" | "3x" | "4x" | "5x" | "6x" | "7x" | "8x" | "9x" | "10x"
            // Relative size classes
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

/// Parse a Unicode codepoint from a CSS value (hex escape, escaped literal, or literal).
fn parse_codepoint(value: &str) -> Option<char> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }

    if let Some(hex_part) = value.strip_prefix('\\') {
        if hex_part.len() >= 2 && hex_part.chars().all(|c| c.is_ascii_hexdigit() || c == ' ') {
            let hex_clean = hex_part.trim();
            if let Ok(code) = u32::from_str_radix(hex_clean, 16) {
                return char::from_u32(code);
            }
        }
        return hex_part.chars().next();
    }

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
    fn test_numeric_icons_not_utility_classes() {
        // Single digit icons 0-9 are valid Font Awesome icons, not utility classes
        for digit in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] {
            assert!(
                !is_utility_class(digit),
                "Digit '{}' should NOT be a utility class",
                digit
            );
        }
        // But size classes like 1x, 2x, 10x ARE utility classes
        assert!(is_utility_class("1x"));
        assert!(is_utility_class("10x"));

        // Brand icons ending in 'x' that start with digits should NOT be filtered
        assert!(
            !is_utility_class("500px"),
            "500px brand icon should NOT be a utility class"
        );
    }

    #[test]
    fn test_parse_500px_brand_icon() {
        // 500px is a brand icon that starts with digit and ends with 'x'
        let css = r#".fa-500px{--fa:"\f26e"}"#;
        let parser = CssParser::parse(css).unwrap();

        assert!(parser.has_icon("500px"), "Icon '500px' should be parsed");
        assert_eq!(parser.get_icon("500px").unwrap().codepoint, '\u{f26e}');
    }

    #[test]
    fn test_parse_numeric_icons() {
        // Test that numeric icons 1-9 are correctly parsed
        let css = r#".fa-1{--fa:"\31 "}.fa-2{--fa:"\32 "}.fa-9{--fa:"\39 "}"#;
        let parser = CssParser::parse(css).unwrap();

        assert!(parser.has_icon("1"), "Icon '1' should be parsed");
        assert!(parser.has_icon("2"), "Icon '2' should be parsed");
        assert!(parser.has_icon("9"), "Icon '9' should be parsed");

        // Verify correct codepoints (ASCII digits)
        assert_eq!(parser.get_icon("1").unwrap().codepoint, '1');
        assert_eq!(parser.get_icon("2").unwrap().codepoint, '2');
        assert_eq!(parser.get_icon("9").unwrap().codepoint, '9');
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
