//! Icon rendering engine using ab_glyph.
//!
//! This module handles rasterizing icon glyphs from font files with
//! supersampling for high-quality antialiased output. Uses ab_glyph
//! for superior curve rendering compared to fontdue.

use crate::color::Color;
use crate::css_parser::{CssParser, FontStyle, IconMapping};
use crate::embedded;
use crate::error::{IconFontError, Result};
use ab_glyph::{Font, FontRef, GlyphId, OutlinedGlyph, PxScale};
use rayon::prelude::*;
use std::borrow::Cow;
use std::path::Path;

/// Horizontal anchor position for icon placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HorizontalAnchor {
    Left,
    #[default]
    Center,
    Right,
}

/// Vertical anchor position for icon placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerticalAnchor {
    Top,
    #[default]
    Center,
    Bottom,
}

/// Configuration for rendering an icon.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Canvas width in pixels
    pub canvas_width: u32,
    /// Canvas height in pixels
    pub canvas_height: u32,
    /// Icon size in pixels (height of the icon)
    pub icon_size: u32,
    /// Supersampling factor for antialiasing (default: 2)
    pub supersample_factor: u32,
    /// Icon foreground color
    pub icon_color: Color,
    /// Background color (use transparent for no background)
    pub background_color: Color,
    /// Horizontal anchor position
    pub horizontal_anchor: HorizontalAnchor,
    /// Vertical anchor position
    pub vertical_anchor: VerticalAnchor,
    /// Horizontal pixel offset from anchor
    pub offset_x: i32,
    /// Vertical pixel offset from anchor
    pub offset_y: i32,
    /// Rotation angle in degrees (positive = clockwise, negative = counter-clockwise)
    pub rotate: f64,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            canvas_width: 512,
            canvas_height: 512,
            // Default to 486px (about 95% of 512px canvas) for margin between icon and edges
            icon_size: 486,
            supersample_factor: 2,
            icon_color: Color::black(),
            background_color: Color::white(),
            horizontal_anchor: HorizontalAnchor::Center,
            vertical_anchor: VerticalAnchor::Center,
            offset_x: 0,
            offset_y: 0,
            rotate: 0.0,
        }
    }
}

impl RenderConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set canvas dimensions.
    pub fn canvas_size(mut self, width: u32, height: u32) -> Self {
        self.canvas_width = width;
        self.canvas_height = height;
        self
    }

    /// Set icon size.
    pub fn icon_size(mut self, size: u32) -> Self {
        self.icon_size = size;
        self
    }

    /// Set supersampling factor.
    pub fn supersample(mut self, factor: u32) -> Self {
        self.supersample_factor = factor.max(1);
        self
    }

    /// Set icon color.
    pub fn icon_color(mut self, color: Color) -> Self {
        self.icon_color = color;
        self
    }

    /// Set background color.
    pub fn background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Set anchor positions.
    pub fn anchor(mut self, horizontal: HorizontalAnchor, vertical: VerticalAnchor) -> Self {
        self.horizontal_anchor = horizontal;
        self.vertical_anchor = vertical;
        self
    }

    /// Set pixel offset from anchor.
    pub fn offset(mut self, x: i32, y: i32) -> Self {
        self.offset_x = x;
        self.offset_y = y;
        self
    }

    /// Set rotation angle in degrees.
    ///
    /// Positive values rotate clockwise, negative values rotate counter-clockwise.
    /// The rotation is applied around the center of the icon before compositing.
    ///
    /// # Arguments
    ///
    /// * `degrees` - Rotation angle in degrees
    ///
    /// # Examples
    ///
    /// ```
    /// use icon_to_image::RenderConfig;
    ///
    /// // Rotate 45 degrees clockwise
    /// let config = RenderConfig::new().rotate(45.0);
    ///
    /// // Rotate 90 degrees counter-clockwise
    /// let config = RenderConfig::new().rotate(-90.0);
    /// ```
    pub fn rotate(mut self, degrees: f64) -> Self {
        self.rotate = degrees;
        self
    }

    /// Apply sanity check to icon_size, clamping to 95% of smaller canvas dimension
    /// if it exceeds either canvas dimension.
    ///
    /// This prevents icons from being larger than the canvas, which would cause
    /// rendering issues.
    pub fn sanitize_icon_size(mut self) -> Self {
        let smaller_dim = self.canvas_width.min(self.canvas_height);
        if self.icon_size > smaller_dim {
            // Clamp to 95% of smaller dimension
            self.icon_size = ((smaller_dim as f64) * 0.95) as u32;
            self.icon_size = self.icon_size.max(1);
        }
        self
    }
}

/// Icon font renderer that loads fonts and renders icons to images.
///
/// Uses ab_glyph for high-quality curve rendering. Font data is stored
/// using `Cow` to support both embedded static data (borrowed) and
/// dynamically loaded data (owned). This allows zero-copy usage of
/// compile-time embedded assets while still supporting runtime loading.
pub struct IconRenderer {
    /// Parsed CSS icon mappings
    css_parser: CssParser,
    /// Solid font data (fa-solid.otf) - borrowed from embedded or owned from file
    font_solid_data: Cow<'static, [u8]>,
    /// Regular font data (fa-regular.otf) - borrowed from embedded or owned from file
    font_regular_data: Cow<'static, [u8]>,
    /// Brands font data (fa-brands.otf) - borrowed from embedded or owned from file
    font_brands_data: Cow<'static, [u8]>,
}

impl IconRenderer {
    /// Create a new renderer using embedded Font Awesome assets.
    ///
    /// This is the recommended constructor for most use cases. The Font Awesome
    /// font files and CSS are compiled directly into the binary, so no external
    /// asset files are needed. This provides better portability at the cost of
    /// increased binary size (~700KB).
    ///
    /// # Returns
    ///
    /// A configured `IconRenderer` ready to render icons.
    ///
    /// # Errors
    ///
    /// Returns error if the embedded fonts cannot be parsed (should never happen
    /// with valid embedded data).
    ///
    /// # Examples
    ///
    /// ```
    /// use icon_to_image::IconRenderer;
    ///
    /// let renderer = IconRenderer::new()?;
    /// # Ok::<(), icon_to_image::IconFontError>(())
    /// ```
    pub fn new() -> Result<Self> {
        // Use static references to embedded data (zero-copy, Cow::Borrowed)
        let font_solid_data = Cow::Borrowed(embedded::FONT_SOLID);
        let font_regular_data = Cow::Borrowed(embedded::FONT_REGULAR);
        let font_brands_data = Cow::Borrowed(embedded::FONT_BRANDS);

        // Validate that fonts can be parsed
        Self::validate_font(&font_solid_data, "embedded fa-solid.otf")?;
        Self::validate_font(&font_regular_data, "embedded fa-regular.otf")?;
        Self::validate_font(&font_brands_data, "embedded fa-brands.otf")?;

        // Parse embedded CSS
        let css_parser = CssParser::parse(embedded::FONTAWESOME_CSS)?;

        Ok(Self {
            css_parser,
            font_solid_data,
            font_regular_data,
            font_brands_data,
        })
    }

    /// Create a new renderer by loading fonts and CSS from a directory.
    ///
    /// Use this constructor when you need to use custom or updated Font Awesome
    /// assets instead of the embedded ones. The directory must contain:
    /// - `fa-solid.otf`
    /// - `fa-regular.otf`
    /// - `fa-brands.otf`
    /// - `fontawesome.css`
    ///
    /// # Arguments
    ///
    /// * `assets_dir` - Path to directory containing font files and CSS
    ///
    /// # Returns
    ///
    /// A configured `IconRenderer` ready to render icons.
    ///
    /// # Errors
    ///
    /// Returns error if fonts or CSS cannot be loaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use icon_to_image::IconRenderer;
    ///
    /// let renderer = IconRenderer::from_path("./assets")?;
    /// # Ok::<(), icon_to_image::IconFontError>(())
    /// ```
    pub fn from_path<P: AsRef<Path>>(assets_dir: P) -> Result<Self> {
        let assets_dir = assets_dir.as_ref();

        // Load font data into owned Vec (Cow::Owned)
        let font_solid_data = Cow::Owned(Self::load_font_data(assets_dir.join("fa-solid.otf"))?);
        let font_regular_data =
            Cow::Owned(Self::load_font_data(assets_dir.join("fa-regular.otf"))?);
        let font_brands_data = Cow::Owned(Self::load_font_data(assets_dir.join("fa-brands.otf"))?);

        // Validate that fonts can be parsed
        Self::validate_font(&font_solid_data, "fa-solid.otf")?;
        Self::validate_font(&font_regular_data, "fa-regular.otf")?;
        Self::validate_font(&font_brands_data, "fa-brands.otf")?;

        // Parse CSS
        let css_path = assets_dir.join("fontawesome.css");
        let css_content = std::fs::read_to_string(&css_path)
            .map_err(|e| IconFontError::CssParseError(format!("Failed to read CSS file: {}", e)))?;
        let css_parser = CssParser::parse(&css_content)?;

        Ok(Self {
            css_parser,
            font_solid_data,
            font_regular_data,
            font_brands_data,
        })
    }

    /// Load font file data as bytes.
    fn load_font_data<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
        let path = path.as_ref();
        std::fs::read(path).map_err(|e| {
            IconFontError::FontLoadError(format!(
                "Failed to read font file {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Validate that font data can be parsed by ab_glyph.
    fn validate_font(data: &[u8], name: &str) -> Result<()> {
        FontRef::try_from_slice(data).map_err(|e| {
            IconFontError::FontLoadError(format!("Failed to parse font {}: {}", name, e))
        })?;
        Ok(())
    }

    /// Get a FontRef for the given style.
    ///
    /// Creates a FontRef that borrows from the underlying font data (whether
    /// embedded or loaded from disk).
    fn get_font(&self, style: FontStyle) -> FontRef<'_> {
        // Cow::as_ref() returns &[u8] regardless of whether data is borrowed or owned
        let data: &[u8] = match style {
            FontStyle::Solid => self.font_solid_data.as_ref(),
            FontStyle::Regular => self.font_regular_data.as_ref(),
            FontStyle::Brands => self.font_brands_data.as_ref(),
        };
        // Safe: we validated at construction
        FontRef::try_from_slice(data).expect("Font was validated at construction")
    }

    /// Check if an icon exists.
    pub fn has_icon(&self, name: &str) -> bool {
        self.css_parser.has_icon(name)
    }

    /// Get the number of available icons.
    pub fn icon_count(&self) -> usize {
        self.css_parser.icon_count()
    }

    /// List all available icon names.
    pub fn list_icons(&self) -> Vec<&str> {
        self.css_parser.list_icons()
    }

    /// Render an icon to an RGBA pixel buffer.
    ///
    /// # Arguments
    ///
    /// * `icon_name` - Name of the icon (with or without "fa-" prefix)
    /// * `config` - Rendering configuration
    ///
    /// # Returns
    ///
    /// A tuple of (width, height, rgba_pixels) where rgba_pixels is a Vec<u8>
    /// containing width * height * 4 bytes in RGBA order.
    ///
    /// # Errors
    ///
    /// Returns error if icon not found or rendering fails.
    pub fn render(&self, icon_name: &str, config: &RenderConfig) -> Result<(u32, u32, Vec<u8>)> {
        // Look up the icon
        let mapping = self
            .css_parser
            .get_icon(icon_name)
            .ok_or_else(|| IconFontError::IconNotFound(icon_name.to_string()))?;

        self.render_mapping(mapping, config)
    }

    /// Render an icon with explicit style override.
    ///
    /// # Arguments
    ///
    /// * `icon_name` - Name of the icon
    /// * `style` - Font style to use
    /// * `config` - Rendering configuration
    ///
    /// # Returns
    ///
    /// Rendered pixel buffer as (width, height, rgba_pixels).
    pub fn render_with_style(
        &self,
        icon_name: &str,
        style: FontStyle,
        config: &RenderConfig,
    ) -> Result<(u32, u32, Vec<u8>)> {
        let mapping = self
            .css_parser
            .get_icon_with_style(icon_name, style)
            .ok_or_else(|| IconFontError::IconNotFound(icon_name.to_string()))?;

        self.render_mapping(&mapping, config)
    }

    /// Render an icon mapping to pixels.
    fn render_mapping(
        &self,
        mapping: &IconMapping,
        config: &RenderConfig,
    ) -> Result<(u32, u32, Vec<u8>)> {
        let font = self.get_font(mapping.style);
        let ss_factor = config.supersample_factor.max(1);

        // Calculate supersampled dimensions
        let ss_canvas_width = config.canvas_width * ss_factor;
        let ss_canvas_height = config.canvas_height * ss_factor;
        let ss_icon_size = config.icon_size * ss_factor;

        // Get the glyph ID for the codepoint
        let glyph_id = font.glyph_id(mapping.codepoint);

        // Check if glyph exists (glyph_id 0 is the .notdef glyph)
        if glyph_id == GlyphId(0) {
            return Ok(self.create_background_only(config));
        }

        // Scale the font to the desired size
        let scale = PxScale::from(ss_icon_size as f32);

        // Get the glyph with position at origin
        let glyph = glyph_id.with_scale_and_position(scale, ab_glyph::point(0.0, 0.0));

        // Outline the glyph for rasterization
        let outlined = match font.outline_glyph(glyph) {
            Some(o) => o,
            None => return Ok(self.create_background_only(config)),
        };

        // Get glyph bounds
        let bounds = outlined.px_bounds();
        let glyph_width = bounds.width() as u32;
        let glyph_height = bounds.height() as u32;

        if glyph_width == 0 || glyph_height == 0 {
            return Ok(self.create_background_only(config));
        }

        // Create supersampled canvas
        let mut ss_canvas = self.create_canvas(ss_canvas_width, ss_canvas_height, config);

        // Check if rotation is needed (non-zero angle)
        // Use small epsilon to handle floating-point comparison
        let needs_rotation = config.rotate.abs() > 0.001;

        if needs_rotation {
            // For rotation: render glyph to intermediate buffer, rotate, then composite
            let (rotated_buffer, rotated_width, rotated_height) = self.render_and_rotate_glyph(
                &outlined,
                glyph_width,
                glyph_height,
                config.rotate,
                &config.icon_color,
            );

            // Calculate position based on anchors using rotated dimensions
            let (x_pos, y_pos) = self.calculate_position(
                ss_canvas_width,
                ss_canvas_height,
                rotated_width,
                rotated_height,
                config,
                ss_factor,
            );

            // Composite rotated buffer onto canvas
            self.composite_buffer(
                &mut ss_canvas,
                ss_canvas_width,
                &rotated_buffer,
                rotated_width,
                rotated_height,
                x_pos,
                y_pos,
            );
        } else {
            // No rotation: direct compositing (original path)
            let (x_pos, y_pos) = self.calculate_position(
                ss_canvas_width,
                ss_canvas_height,
                glyph_width,
                glyph_height,
                config,
                ss_factor,
            );

            self.composite_outlined_glyph(
                &mut ss_canvas,
                ss_canvas_width,
                &outlined,
                x_pos,
                y_pos,
                &config.icon_color,
            );
        }

        // Downsample if needed
        let final_pixels = if ss_factor > 1 {
            self.downsample(&ss_canvas, ss_canvas_width, ss_canvas_height, ss_factor)
        } else {
            ss_canvas
        };

        Ok((config.canvas_width, config.canvas_height, final_pixels))
    }

    /// Render a glyph to a buffer and rotate it.
    ///
    /// Returns the rotated RGBA buffer along with its new dimensions.
    /// The rotation uses bilinear interpolation for smooth results.
    fn render_and_rotate_glyph(
        &self,
        outlined: &OutlinedGlyph,
        glyph_width: u32,
        glyph_height: u32,
        degrees: f64,
        color: &Color,
    ) -> (Vec<u8>, u32, u32) {
        // First, render the glyph to an intermediate buffer with alpha
        let mut glyph_buffer = vec![0u8; (glyph_width * glyph_height * 4) as usize];

        outlined.draw(|gx, gy, coverage| {
            let coverage_u8 = (coverage * 255.0) as u8;
            if coverage_u8 == 0 {
                return;
            }

            let idx = ((gy * glyph_width + gx) * 4) as usize;
            if idx + 3 < glyph_buffer.len() {
                // Store color with premultiplied alpha based on coverage
                let alpha = (coverage_u8 as u32 * color.a as u32) / 255;
                glyph_buffer[idx] = color.r;
                glyph_buffer[idx + 1] = color.g;
                glyph_buffer[idx + 2] = color.b;
                glyph_buffer[idx + 3] = alpha as u8;
            }
        });

        // Convert degrees to radians
        let radians = degrees.to_radians();
        let cos_angle = radians.cos();
        let sin_angle = radians.sin();

        // Calculate the bounding box of the rotated image
        // The corners of the original image after rotation determine the new bounds
        let w = glyph_width as f64;
        let h = glyph_height as f64;

        // Calculate rotated corner positions relative to center
        let half_w = w / 2.0;
        let half_h = h / 2.0;

        // Compute the absolute width and height after rotation
        // For a rectangle rotated by angle theta:
        // new_width = |w * cos(theta)| + |h * sin(theta)|
        // new_height = |w * sin(theta)| + |h * cos(theta)|
        let new_width = (w * cos_angle.abs() + h * sin_angle.abs()).ceil() as u32;
        let new_height = (w * sin_angle.abs() + h * cos_angle.abs()).ceil() as u32;

        // Ensure minimum size of 1
        let new_width = new_width.max(1);
        let new_height = new_height.max(1);

        let new_half_w = new_width as f64 / 2.0;
        let new_half_h = new_height as f64 / 2.0;

        // Create output buffer (transparent background)
        let mut rotated = vec![0u8; (new_width * new_height * 4) as usize];

        // Perform inverse rotation sampling with bilinear interpolation
        // For each pixel in the output, find the corresponding source pixel
        for out_y in 0..new_height {
            for out_x in 0..new_width {
                // Convert to coordinates centered on the output image
                let dx = out_x as f64 - new_half_w;
                let dy = out_y as f64 - new_half_h;

                // Apply inverse rotation to find source coordinates
                // Inverse rotation is rotation by -angle
                let src_x = dx * cos_angle + dy * sin_angle + half_w;
                let src_y = -dx * sin_angle + dy * cos_angle + half_h;

                // Bilinear interpolation
                if src_x >= 0.0 && src_x < w - 1.0 && src_y >= 0.0 && src_y < h - 1.0 {
                    let out_idx = ((out_y * new_width + out_x) * 4) as usize;
                    let (r, g, b, a) = self.bilinear_sample(
                        &glyph_buffer,
                        glyph_width,
                        glyph_height,
                        src_x,
                        src_y,
                    );
                    rotated[out_idx] = r;
                    rotated[out_idx + 1] = g;
                    rotated[out_idx + 2] = b;
                    rotated[out_idx + 3] = a;
                }
            }
        }

        (rotated, new_width, new_height)
    }

    /// Sample a pixel from an RGBA buffer using bilinear interpolation.
    ///
    /// This produces smooth results when rotating by handling sub-pixel positions.
    fn bilinear_sample(
        &self,
        buffer: &[u8],
        width: u32,
        height: u32,
        x: f64,
        y: f64,
    ) -> (u8, u8, u8, u8) {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x.floor();
        let fy = y - y.floor();

        // Get the four surrounding pixels
        let get_pixel = |px: u32, py: u32| -> (f64, f64, f64, f64) {
            let idx = ((py * width + px) * 4) as usize;
            (
                buffer[idx] as f64,
                buffer[idx + 1] as f64,
                buffer[idx + 2] as f64,
                buffer[idx + 3] as f64,
            )
        };

        let p00 = get_pixel(x0, y0);
        let p10 = get_pixel(x1, y0);
        let p01 = get_pixel(x0, y1);
        let p11 = get_pixel(x1, y1);

        // Bilinear interpolation for each channel
        let lerp = |a: f64, b: f64, t: f64| a * (1.0 - t) + b * t;

        let r = lerp(lerp(p00.0, p10.0, fx), lerp(p01.0, p11.0, fx), fy);
        let g = lerp(lerp(p00.1, p10.1, fx), lerp(p01.1, p11.1, fx), fy);
        let b = lerp(lerp(p00.2, p10.2, fx), lerp(p01.2, p11.2, fx), fy);
        let a = lerp(lerp(p00.3, p10.3, fx), lerp(p01.3, p11.3, fx), fy);

        (
            r.round() as u8,
            g.round() as u8,
            b.round() as u8,
            a.round() as u8,
        )
    }

    /// Composite an RGBA buffer onto the canvas with alpha blending.
    #[allow(clippy::too_many_arguments)]
    fn composite_buffer(
        &self,
        canvas: &mut [u8],
        canvas_width: u32,
        buffer: &[u8],
        buffer_width: u32,
        buffer_height: u32,
        x_offset: i32,
        y_offset: i32,
    ) {
        for by in 0..buffer_height {
            for bx in 0..buffer_width {
                let buf_idx = ((by * buffer_width + bx) * 4) as usize;
                let src_a = buffer[buf_idx + 3] as u32;

                if src_a == 0 {
                    continue;
                }

                let cx = x_offset + bx as i32;
                let cy = y_offset + by as i32;

                if cx < 0 || cy < 0 || cx >= canvas_width as i32 {
                    continue;
                }

                let canvas_idx = (cy as u32 * canvas_width + cx as u32) as usize * 4;
                if canvas_idx + 3 >= canvas.len() {
                    continue;
                }

                // Porter-Duff "over" compositing
                let inv_src_alpha = 255 - src_a;

                let dst_r = canvas[canvas_idx] as u32;
                let dst_g = canvas[canvas_idx + 1] as u32;
                let dst_b = canvas[canvas_idx + 2] as u32;
                let dst_a = canvas[canvas_idx + 3] as u32;

                canvas[canvas_idx] =
                    ((buffer[buf_idx] as u32 * src_a + dst_r * inv_src_alpha) / 255) as u8;
                canvas[canvas_idx + 1] =
                    ((buffer[buf_idx + 1] as u32 * src_a + dst_g * inv_src_alpha) / 255) as u8;
                canvas[canvas_idx + 2] =
                    ((buffer[buf_idx + 2] as u32 * src_a + dst_b * inv_src_alpha) / 255) as u8;
                canvas[canvas_idx + 3] = (src_a + (dst_a * inv_src_alpha) / 255) as u8;
            }
        }
    }

    /// Create a canvas filled with background color.
    fn create_canvas(&self, width: u32, height: u32, config: &RenderConfig) -> Vec<u8> {
        let pixel_count = (width * height) as usize;
        let bg = config.background_color.to_rgba();

        let mut canvas = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            canvas.extend_from_slice(&bg);
        }
        canvas
    }

    /// Create an image with only background (for missing glyphs).
    fn create_background_only(&self, config: &RenderConfig) -> (u32, u32, Vec<u8>) {
        let pixels = self.create_canvas(config.canvas_width, config.canvas_height, config);
        (config.canvas_width, config.canvas_height, pixels)
    }

    /// Calculate icon position based on anchors and offsets.
    fn calculate_position(
        &self,
        canvas_width: u32,
        canvas_height: u32,
        glyph_width: u32,
        glyph_height: u32,
        config: &RenderConfig,
        ss_factor: u32,
    ) -> (i32, i32) {
        let ss_offset_x = config.offset_x * ss_factor as i32;
        let ss_offset_y = config.offset_y * ss_factor as i32;

        let x = match config.horizontal_anchor {
            HorizontalAnchor::Left => ss_offset_x,
            HorizontalAnchor::Center => {
                (canvas_width as i32 - glyph_width as i32) / 2 + ss_offset_x
            }
            HorizontalAnchor::Right => canvas_width as i32 - glyph_width as i32 + ss_offset_x,
        };

        let y = match config.vertical_anchor {
            VerticalAnchor::Top => ss_offset_y,
            VerticalAnchor::Center => {
                (canvas_height as i32 - glyph_height as i32) / 2 + ss_offset_y
            }
            VerticalAnchor::Bottom => canvas_height as i32 - glyph_height as i32 + ss_offset_y,
        };

        (x, y)
    }

    /// Composite an outlined glyph onto the canvas with alpha blending.
    ///
    /// ab_glyph's draw method calls the provided closure for each pixel
    /// with coverage values, giving us high-quality antialiased output.
    fn composite_outlined_glyph(
        &self,
        canvas: &mut [u8],
        canvas_width: u32,
        outlined: &OutlinedGlyph,
        x_offset: i32,
        y_offset: i32,
        color: &Color,
    ) {
        // ab_glyph's draw method iterates over each pixel in the glyph's bounds
        // and provides coverage as a float from 0.0 to 1.0
        outlined.draw(|gx, gy, coverage| {
            // Convert coverage (0.0-1.0) to u8 (0-255)
            let coverage_u8 = (coverage * 255.0) as u8;

            if coverage_u8 == 0 {
                return;
            }

            // Calculate canvas position (glyph coords are relative to bounds.min)
            let cx = x_offset + gx as i32;
            let cy = y_offset + gy as i32;

            // Skip pixels outside canvas bounds
            if cx < 0 || cy < 0 || cx >= canvas_width as i32 {
                return;
            }

            let canvas_idx = (cy as u32 * canvas_width + cx as u32) as usize * 4;
            if canvas_idx + 3 >= canvas.len() {
                return;
            }

            // Alpha blend the glyph pixel onto the canvas
            // Using premultiplied alpha for correct blending
            let src_alpha = (coverage_u8 as u32 * color.a as u32) / 255;
            let inv_src_alpha = 255 - src_alpha;

            let dst_r = canvas[canvas_idx] as u32;
            let dst_g = canvas[canvas_idx + 1] as u32;
            let dst_b = canvas[canvas_idx + 2] as u32;
            let dst_a = canvas[canvas_idx + 3] as u32;

            // Porter-Duff "over" compositing
            canvas[canvas_idx] = ((color.r as u32 * src_alpha + dst_r * inv_src_alpha) / 255) as u8;
            canvas[canvas_idx + 1] =
                ((color.g as u32 * src_alpha + dst_g * inv_src_alpha) / 255) as u8;
            canvas[canvas_idx + 2] =
                ((color.b as u32 * src_alpha + dst_b * inv_src_alpha) / 255) as u8;
            canvas[canvas_idx + 3] = (src_alpha + (dst_a * inv_src_alpha) / 255) as u8;
        });
    }

    /// Downsample a supersampled image using box filtering.
    ///
    /// Uses parallel processing for better performance on large images.
    fn downsample(&self, ss_pixels: &[u8], ss_width: u32, ss_height: u32, factor: u32) -> Vec<u8> {
        let out_width = ss_width / factor;
        let out_height = ss_height / factor;
        let out_size = (out_width * out_height) as usize;
        let factor_sq = factor * factor;

        // Process rows in parallel for better performance
        let rows: Vec<Vec<u8>> = (0..out_height)
            .into_par_iter()
            .map(|out_y| {
                let mut row = Vec::with_capacity(out_width as usize * 4);
                for out_x in 0..out_width {
                    // Sum all pixels in the supersampled block
                    let mut r_sum: u32 = 0;
                    let mut g_sum: u32 = 0;
                    let mut b_sum: u32 = 0;
                    let mut a_sum: u32 = 0;

                    let base_x = out_x * factor;
                    let base_y = out_y * factor;

                    for dy in 0..factor {
                        for dx in 0..factor {
                            let ss_idx = ((base_y + dy) * ss_width + (base_x + dx)) as usize * 4;
                            r_sum += ss_pixels[ss_idx] as u32;
                            g_sum += ss_pixels[ss_idx + 1] as u32;
                            b_sum += ss_pixels[ss_idx + 2] as u32;
                            a_sum += ss_pixels[ss_idx + 3] as u32;
                        }
                    }

                    // Average the samples
                    row.push((r_sum / factor_sq) as u8);
                    row.push((g_sum / factor_sq) as u8);
                    row.push((b_sum / factor_sq) as u8);
                    row.push((a_sum / factor_sq) as u8);
                }
                row
            })
            .collect();

        // Flatten rows into single buffer
        let mut result = Vec::with_capacity(out_size * 4);
        for row in rows {
            result.extend(row);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_config_builder() {
        let config = RenderConfig::new()
            .canvas_size(512, 512)
            .icon_size(256)
            .supersample(4)
            .icon_color(Color::rgb(255, 0, 0))
            .background_color(Color::transparent())
            .anchor(HorizontalAnchor::Left, VerticalAnchor::Top)
            .offset(10, 20)
            .rotate(45.0);

        assert_eq!(config.canvas_width, 512);
        assert_eq!(config.canvas_height, 512);
        assert_eq!(config.icon_size, 256);
        assert_eq!(config.supersample_factor, 4);
        assert_eq!(config.icon_color, Color::rgb(255, 0, 0));
        assert!(config.background_color.is_transparent());
        assert_eq!(config.horizontal_anchor, HorizontalAnchor::Left);
        assert_eq!(config.vertical_anchor, VerticalAnchor::Top);
        assert_eq!(config.offset_x, 10);
        assert_eq!(config.offset_y, 20);
        assert!((config.rotate - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_render_config_rotate_default() {
        let config = RenderConfig::default();
        assert!((config.rotate - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_render_config_rotate_negative() {
        let config = RenderConfig::new().rotate(-90.0);
        assert!((config.rotate - (-90.0)).abs() < 0.001);
    }
}
