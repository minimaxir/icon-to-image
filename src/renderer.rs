//! Icon rendering engine using ab_glyph with SIMD optimizations.
//!
//! Handles rasterizing icon glyphs from font files with supersampling
//! for high-quality antialiased output. SIMD acceleration is applied
//! when dimensions are multiples of 8.

use crate::color::Color;
use crate::css_parser::{CssParser, FontStyle, IconMapping};
use crate::embedded;
use crate::error::{IconFontError, Result};
use ab_glyph::{Font, FontArc, GlyphId, PxScale};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Minimum pixel count before parallelizing compositing.
/// Below this, rayon thread dispatch overhead exceeds the benefit.
/// Set to 32K to cover 256x256 icons (~240x240 glyph = ~57K pixels).
const PARALLEL_PIXEL_THRESHOLD: usize = 32 * 1024;

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
/// as parsed `FontArc` handles so glyph lookup does not reparse font tables
/// on each render call. Embedded assets are cached globally and shared
/// across instances.
pub struct IconRenderer {
    /// Parsed CSS icon mappings
    css_parser: CssParser,
    /// Parsed solid font (fa-solid.otf)
    font_solid: FontArc,
    /// Parsed regular font (fa-regular.otf)
    font_regular: FontArc,
    /// Parsed brands font (fa-brands.otf)
    font_brands: FontArc,
    /// Cache of rasterized glyph alpha masks by style/codepoint/size.
    /// Uses RwLock to allow concurrent cache reads (the common path in benchmarks).
    glyph_cache: RwLock<FxHashMap<GlyphCacheKey, Arc<GlyphMask>>>,
}

#[derive(Debug, Clone)]
struct GlyphMask {
    width: u32,
    height: u32,
    alpha: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphCacheKey {
    style: FontStyle,
    codepoint: u32,
    ss_icon_size: u32,
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
        let css_parser = CssParser::parse(embedded::FONTAWESOME_CSS)?;
        let font_solid = FontArc::try_from_slice(embedded::FONT_SOLID).map_err(|err| {
            IconFontError::FontLoadError(format!("Failed to parse embedded fa-solid.otf: {}", err))
        })?;
        let font_regular = FontArc::try_from_slice(embedded::FONT_REGULAR).map_err(|err| {
            IconFontError::FontLoadError(format!(
                "Failed to parse embedded fa-regular.otf: {}",
                err
            ))
        })?;
        let font_brands = FontArc::try_from_slice(embedded::FONT_BRANDS).map_err(|err| {
            IconFontError::FontLoadError(format!("Failed to parse embedded fa-brands.otf: {}", err))
        })?;

        Ok(Self {
            css_parser,
            font_solid,
            font_regular,
            font_brands,
            glyph_cache: RwLock::new(FxHashMap::with_capacity_and_hasher(256, Default::default())),
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

        let font_solid = Self::load_font(assets_dir.join("fa-solid.otf"), "fa-solid.otf")?;
        let font_regular = Self::load_font(assets_dir.join("fa-regular.otf"), "fa-regular.otf")?;
        let font_brands = Self::load_font(assets_dir.join("fa-brands.otf"), "fa-brands.otf")?;

        // Parse CSS
        let css_path = assets_dir.join("fontawesome.css");
        let css_content = std::fs::read_to_string(&css_path)
            .map_err(|e| IconFontError::CssParseError(format!("Failed to read CSS file: {}", e)))?;
        let css_parser = CssParser::parse(&css_content)?;

        Ok(Self {
            css_parser,
            font_solid,
            font_regular,
            font_brands,
            glyph_cache: RwLock::new(FxHashMap::with_capacity_and_hasher(256, Default::default())),
        })
    }

    /// Load and parse a font file.
    fn load_font<P: AsRef<Path>>(path: P, name: &str) -> Result<FontArc> {
        let path = path.as_ref();
        let data = std::fs::read(path).map_err(|e| {
            IconFontError::FontLoadError(format!(
                "Failed to read font file {}: {}",
                path.display(),
                e
            ))
        })?;
        FontArc::try_from_vec(data).map_err(|e| {
            IconFontError::FontLoadError(format!("Failed to parse font {}: {}", name, e))
        })
    }

    /// Get a parsed font handle for the given style.
    fn get_font(&self, style: FontStyle) -> &FontArc {
        match style {
            FontStyle::Solid => &self.font_solid,
            FontStyle::Regular => &self.font_regular,
            FontStyle::Brands => &self.font_brands,
        }
    }

    /// Get a cached rasterized glyph mask, generating it if needed.
    #[inline]
    fn get_or_rasterized_glyph_mask(
        &self,
        style: FontStyle,
        codepoint: char,
        ss_icon_size: u32,
    ) -> Option<Arc<GlyphMask>> {
        let key = GlyphCacheKey {
            style,
            codepoint: codepoint as u32,
            ss_icon_size,
        };

        // Fast path: read lock for cache hits (no contention with other readers)
        if let Some(mask) = self
            .glyph_cache
            .read()
            .expect("glyph cache rwlock poisoned")
            .get(&key)
            .map(Arc::clone)
        {
            return Some(mask);
        }

        let font = self.get_font(style);
        let glyph_id = font.glyph_id(codepoint);
        if glyph_id == GlyphId(0) {
            return None;
        }

        let glyph = glyph_id.with_scale_and_position(
            PxScale::from(ss_icon_size as f32),
            ab_glyph::point(0.0, 0.0),
        );
        let outlined = font.outline_glyph(glyph)?;
        let bounds = outlined.px_bounds();
        let width = bounds.width() as u32;
        let height = bounds.height() as u32;
        if width == 0 || height == 0 {
            return None;
        }

        let width_usize = width as usize;
        let mut alpha = vec![0u8; width_usize * height as usize];
        outlined.draw(|gx, gy, coverage| {
            let idx = gy as usize * width_usize + gx as usize;
            alpha[idx] = (coverage * 255.0) as u8;
        });

        let mask = Arc::new(GlyphMask {
            width,
            height,
            alpha,
        });
        let mut cache = self
            .glyph_cache
            .write()
            .expect("glyph cache rwlock poisoned");
        if cache.len() >= 2048 {
            cache.clear();
        }
        let cached = cache.entry(key).or_insert(mask);
        Some(Arc::clone(cached))
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
    #[inline]
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
    #[inline]
    fn render_mapping(
        &self,
        mapping: &IconMapping,
        config: &RenderConfig,
    ) -> Result<(u32, u32, Vec<u8>)> {
        let ss_factor = config.supersample_factor.max(1);

        let ss_canvas_width = config.canvas_width * ss_factor;
        let ss_canvas_height = config.canvas_height * ss_factor;
        let mut ss_icon_size = config.icon_size * ss_factor;
        let mut glyph_mask =
            match self.get_or_rasterized_glyph_mask(mapping.style, mapping.codepoint, ss_icon_size)
            {
                Some(mask) => mask,
                None => return Ok(self.create_background_only(config)),
            };

        // Some glyphs in newer font versions can exceed the requested em-size bounds.
        // If that would clip against the canvas, shrink to a safe scale automatically.
        while glyph_mask.width > ss_canvas_width || glyph_mask.height > ss_canvas_height {
            let Some(next_size) = scaled_icon_size_to_fit_canvas(
                ss_icon_size,
                glyph_mask.width,
                glyph_mask.height,
                ss_canvas_width,
                ss_canvas_height,
            ) else {
                break;
            };

            ss_icon_size = next_size;
            glyph_mask = match self.get_or_rasterized_glyph_mask(
                mapping.style,
                mapping.codepoint,
                ss_icon_size,
            ) {
                Some(mask) => mask,
                None => return Ok(self.create_background_only(config)),
            };
        }

        let glyph_width = glyph_mask.width;
        let glyph_height = glyph_mask.height;

        if glyph_width == 0 || glyph_height == 0 {
            return Ok(self.create_background_only(config));
        }

        let needs_rotation = config.rotate.abs() > 0.001;

        if needs_rotation {
            let mut ss_canvas = self.create_canvas(ss_canvas_width, ss_canvas_height, config);
            let (rotated_alpha, rotated_width, rotated_height) =
                self.render_and_rotate_glyph(&glyph_mask, config.rotate);

            let (x_pos, y_pos) = self.calculate_position(
                ss_canvas_width,
                ss_canvas_height,
                rotated_width,
                rotated_height,
                config,
                ss_factor,
            );

            self.composite_alpha_buffer(
                &mut ss_canvas,
                ss_canvas_width,
                &rotated_alpha,
                rotated_width,
                rotated_height,
                x_pos,
                y_pos,
                &config.icon_color,
            );

            let final_pixels = match ss_factor {
                1 => ss_canvas,
                _ => self.downsample(&ss_canvas, ss_canvas_width, ss_canvas_height, ss_factor),
            };
            return Ok((config.canvas_width, config.canvas_height, final_pixels));
        }

        let (x_pos, y_pos) = self.calculate_position(
            ss_canvas_width,
            ss_canvas_height,
            glyph_width,
            glyph_height,
            config,
            ss_factor,
        );

        // Fused fill+composite: build the canvas with background and composite
        // the glyph in a single pass when the background is opaque white and
        // icon color alpha is 255 (the common benchmark/production case).
        let bg = config.background_color.to_rgba();
        let icon_a = config.icon_color.a;
        // Fuse fill+composite when background is opaque white and icon alpha is 255.
        // Avoids writing background pixels that will be immediately overwritten.
        let can_fuse = bg == [255, 255, 255, 255] && icon_a == 255 && ss_factor == 1;

        let ss_canvas = if can_fuse {
            self.create_canvas_with_glyph(
                ss_canvas_width,
                ss_canvas_height,
                &glyph_mask,
                x_pos,
                y_pos,
                &config.icon_color,
            )
        } else {
            let mut canvas = self.create_canvas(ss_canvas_width, ss_canvas_height, config);
            self.composite_glyph_mask(
                &mut canvas,
                ss_canvas_width,
                &glyph_mask,
                x_pos,
                y_pos,
                &config.icon_color,
            );
            canvas
        };

        let final_pixels = match ss_factor {
            1 => ss_canvas,
            _ => self.downsample(&ss_canvas, ss_canvas_width, ss_canvas_height, ss_factor),
        };

        Ok((config.canvas_width, config.canvas_height, final_pixels))
    }

    /// Render a glyph to a buffer and rotate it using bilinear interpolation.
    fn render_and_rotate_glyph(&self, glyph_mask: &GlyphMask, degrees: f64) -> (Vec<u8>, u32, u32) {
        let glyph_width = glyph_mask.width;
        let glyph_height = glyph_mask.height;
        let glyph_width_usize = glyph_width as usize;

        let radians = (degrees as f32).to_radians();
        let cos_angle = radians.cos();
        let sin_angle = radians.sin();

        let w = glyph_width as f32;
        let h = glyph_height as f32;
        let half_w = w / 2.0;
        let half_h = h / 2.0;

        let new_width = (w * cos_angle.abs() + h * sin_angle.abs()).ceil() as u32;
        let new_height = (w * sin_angle.abs() + h * cos_angle.abs()).ceil() as u32;
        let new_width = new_width.max(1);
        let new_height = new_height.max(1);
        let new_width_usize = new_width as usize;

        let new_half_w = new_width as f32 / 2.0;
        let new_half_h = new_height as f32 / 2.0;

        let mut rotated_alpha = vec![0u8; (new_width * new_height) as usize];
        let max_src_x = w - 1.0;
        let max_src_y = h - 1.0;

        // Parallel row processing with incremental affine stepping and alpha-only sampling.
        rotated_alpha
            .par_chunks_mut(new_width_usize)
            .enumerate()
            .for_each(|(out_y, row)| {
                let dy = out_y as f32 - new_half_h;
                let mut src_x = (-new_half_w) * cos_angle + dy * sin_angle + half_w;
                let mut src_y = new_half_w * sin_angle + dy * cos_angle + half_h;

                for pixel in row.iter_mut().take(new_width_usize) {
                    if src_x >= 0.0 && src_x < max_src_x && src_y >= 0.0 && src_y < max_src_y {
                        *pixel = bilinear_sample_alpha(
                            &glyph_mask.alpha,
                            glyph_width_usize,
                            src_x,
                            src_y,
                        );
                    }
                    src_x += cos_angle;
                    src_y -= sin_angle;
                }
            });

        (rotated_alpha, new_width, new_height)
    }

    /// Composite an alpha mask onto the canvas using a constant color.
    ///
    /// Parallelized across rows via rayon for large buffers.
    #[allow(clippy::too_many_arguments)]
    fn composite_alpha_buffer(
        &self,
        canvas: &mut [u8],
        canvas_width: u32,
        alpha: &[u8],
        buffer_width: u32,
        buffer_height: u32,
        x_offset: i32,
        y_offset: i32,
        color: &Color,
    ) {
        let canvas_width_i32 = canvas_width as i32;
        let canvas_height_i32 = (canvas.len() / (canvas_width as usize * 4)) as i32;
        let buffer_width_i32 = buffer_width as i32;
        let buffer_height_i32 = buffer_height as i32;

        let dst_start_x = x_offset.max(0);
        let dst_start_y = y_offset.max(0);
        let dst_end_x = (x_offset + buffer_width_i32).min(canvas_width_i32);
        let dst_end_y = (y_offset + buffer_height_i32).min(canvas_height_i32);

        if dst_start_x >= dst_end_x || dst_start_y >= dst_end_y {
            return;
        }

        let canvas_stride = canvas_width as usize * 4;
        let buffer_stride = buffer_width as usize;
        let color_a = color.a as u32;
        let is_opaque = color_a == 255;
        let color_r = color.r;
        let color_g = color.g;
        let color_b = color.b;
        let bx_start = (dst_start_x - x_offset) as usize;
        let span = (dst_end_x - dst_start_x) as usize;
        let px_start = dst_start_x as usize * 4;
        let px_end = px_start + span * 4;
        let row_count = (dst_end_y - dst_start_y) as usize;
        let first_row = dst_start_y as usize;
        let y_off = y_offset as usize;

        let canvas_region_start = first_row * canvas_stride;
        let canvas_region_end = (first_row + row_count) * canvas_stride;
        let canvas_region = &mut canvas[canvas_region_start..canvas_region_end];

        let total_pixels = row_count * span;
        if total_pixels >= PARALLEL_PIXEL_THRESHOLD {
            canvas_region
                .par_chunks_mut(canvas_stride)
                .enumerate()
                .for_each(|(row_idx, canvas_row_full)| {
                    let by = first_row + row_idx - y_off;
                    let buffer_row = by * buffer_stride;
                    let alpha_slice = &alpha[buffer_row + bx_start..buffer_row + bx_start + span];
                    let canvas_row = &mut canvas_row_full[px_start..px_end];
                    composite_alpha_row(
                        canvas_row,
                        alpha_slice,
                        color_r,
                        color_g,
                        color_b,
                        color_a,
                        is_opaque,
                    );
                });
        } else {
            for (row_idx, canvas_row_full) in canvas_region.chunks_mut(canvas_stride).enumerate() {
                let by = first_row + row_idx - y_off;
                let buffer_row = by * buffer_stride;
                let alpha_slice = &alpha[buffer_row + bx_start..buffer_row + bx_start + span];
                let canvas_row = &mut canvas_row_full[px_start..px_end];
                composite_alpha_row(
                    canvas_row,
                    alpha_slice,
                    color_r,
                    color_g,
                    color_b,
                    color_a,
                    is_opaque,
                );
            }
        }
    }

    /// Fused canvas creation + glyph compositing for white bg, opaque icon, ss=1.
    ///
    /// Fills the canvas with white and composites the glyph in a single pass per
    /// row. For rows outside the glyph region, the initial `vec![255; N]` memset
    /// already provides the correct white background.
    #[inline]
    fn create_canvas_with_glyph(
        &self,
        width: u32,
        height: u32,
        glyph_mask: &GlyphMask,
        x_offset: i32,
        y_offset: i32,
        color: &Color,
    ) -> Vec<u8> {
        let total_bytes = (width * height) as usize * 4;
        let mut canvas = vec![255u8; total_bytes];
        let canvas_stride = width as usize * 4;

        let glyph_w = glyph_mask.width as i32;
        let glyph_h = glyph_mask.height as i32;
        let canvas_w = width as i32;
        let canvas_h = height as i32;

        let min_gx = (-x_offset).max(0) as usize;
        let min_gy = (-y_offset).max(0) as usize;
        let max_gx = (canvas_w - x_offset).min(glyph_w) as usize;
        let max_gy = (canvas_h - y_offset).min(glyph_h) as usize;

        if min_gx >= max_gx || min_gy >= max_gy {
            return canvas;
        }

        let glyph_stride = glyph_mask.width as usize;
        let x_off = x_offset as usize;
        let color_r = color.r;
        let color_g = color.g;
        let color_b = color.b;
        let row_count = max_gy - min_gy;
        let span = max_gx - min_gx;
        let total_glyph_pixels = row_count * span;
        let first_canvas_row = (y_offset as usize) + min_gy;

        // Pre-compute (color_channel - 255) to reduce ops per pixel.
        // Blend formula: div255(color * c + 255 * (255-c))
        //              = div255(c * (color - 255) + 65025)
        let dr = color_r as i32 - 255;
        let dg = color_g as i32 - 255;
        let db = color_b as i32 - 255;

        let canvas_region_start = first_canvas_row * canvas_stride;
        let canvas_region_end = (first_canvas_row + row_count) * canvas_stride;
        let canvas_region = &mut canvas[canvas_region_start..canvas_region_end];

        let blend_row = |canvas_row_full: &mut [u8], gy: usize| {
            let glyph_row_start = gy * glyph_stride + min_gx;
            let alpha_row = &glyph_mask.alpha[glyph_row_start..glyph_row_start + span];
            let px_start = (x_off + min_gx) * 4;
            let canvas_row = &mut canvas_row_full[px_start..px_start + span * 4];

            for (i, &coverage) in alpha_row.iter().enumerate() {
                if coverage == 0 {
                    continue;
                }
                let dst = i * 4;
                if coverage == 255 {
                    canvas_row[dst] = color_r;
                    canvas_row[dst + 1] = color_g;
                    canvas_row[dst + 2] = color_b;
                } else {
                    let c = coverage as i32;
                    canvas_row[dst] = div255((c * dr + 65025) as u32) as u8;
                    canvas_row[dst + 1] = div255((c * dg + 65025) as u32) as u8;
                    canvas_row[dst + 2] = div255((c * db + 65025) as u32) as u8;
                }
            }
        };

        if total_glyph_pixels >= PARALLEL_PIXEL_THRESHOLD {
            canvas_region
                .par_chunks_mut(canvas_stride)
                .enumerate()
                .for_each(|(row_idx, canvas_row_full)| {
                    blend_row(canvas_row_full, min_gy + row_idx);
                });
        } else {
            for (row_idx, canvas_row_full) in canvas_region.chunks_mut(canvas_stride).enumerate() {
                blend_row(canvas_row_full, min_gy + row_idx);
            }
        }

        canvas
    }

    /// Create a canvas filled with background color.
    fn create_canvas(&self, width: u32, height: u32, config: &RenderConfig) -> Vec<u8> {
        let bg = config.background_color.to_rgba();
        let total_bytes = (width * height) as usize * 4;

        if bg == [0, 0, 0, 0] {
            return vec![0u8; total_bytes];
        }

        // All channels same: compiles to optimized memset
        if bg[0] == bg[1] && bg[1] == bg[2] && bg[2] == bg[3] {
            return vec![bg[0]; total_bytes];
        }

        // Exponential doubling fill: copies grow geometrically for O(n) with
        // low loop overhead
        let mut canvas = vec![0u8; total_bytes];
        canvas[..4].copy_from_slice(&bg);
        let mut filled = 4usize;
        while filled < total_bytes {
            let copy_len = filled.min(total_bytes - filled);
            let (src, dst) = canvas.split_at_mut(filled);
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
            filled += copy_len;
        }
        canvas
    }

    /// Create an image with only background (for missing glyphs).
    fn create_background_only(&self, config: &RenderConfig) -> (u32, u32, Vec<u8>) {
        let pixels = self.create_canvas(config.canvas_width, config.canvas_height, config);
        (config.canvas_width, config.canvas_height, pixels)
    }

    /// Calculate icon position based on anchors and offsets.
    #[inline]
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

    /// Composite a cached glyph coverage mask onto the canvas.
    ///
    /// Uses parallel row processing via rayon for large images to saturate
    /// memory bandwidth across multiple cores.
    fn composite_glyph_mask(
        &self,
        canvas: &mut [u8],
        canvas_width: u32,
        glyph_mask: &GlyphMask,
        x_offset: i32,
        y_offset: i32,
        color: &Color,
    ) {
        let glyph_width = glyph_mask.width as i32;
        let glyph_height = glyph_mask.height as i32;
        let canvas_height_i32 = (canvas.len() / (canvas_width as usize * 4)) as i32;
        let canvas_width_i32 = canvas_width as i32;
        let canvas_stride = canvas_width as usize * 4;

        let min_gx = (-x_offset).max(0) as usize;
        let min_gy = (-y_offset).max(0);
        let max_gx = (canvas_width_i32 - x_offset).min(glyph_width) as usize;
        let max_gy = (canvas_height_i32 - y_offset).min(glyph_height);

        if min_gx >= max_gx || min_gy >= max_gy {
            return;
        }

        let glyph_width_usize = glyph_mask.width as usize;
        let color_a = color.a as u32;
        let color_r = color.r;
        let color_g = color.g;
        let color_b = color.b;
        let is_opaque_color = color_a == 255;
        let x_off = x_offset as usize;
        let row_count = (max_gy - min_gy) as usize;
        let first_canvas_row = (y_offset + min_gy) as usize;

        // Slice the canvas to only the rows we need
        let canvas_region_start = first_canvas_row * canvas_stride;
        let canvas_region_end = (first_canvas_row + row_count) * canvas_stride;
        let canvas_region = &mut canvas[canvas_region_start..canvas_region_end];

        // Threshold: parallelize only when total pixel work justifies rayon overhead
        let total_pixels = row_count * (max_gx - min_gx);
        if total_pixels >= PARALLEL_PIXEL_THRESHOLD {
            canvas_region
                .par_chunks_mut(canvas_stride)
                .enumerate()
                .for_each(|(row_idx, canvas_row_full)| {
                    let gy = min_gy as usize + row_idx;
                    let glyph_row = gy * glyph_width_usize;
                    let alpha_row = &glyph_mask.alpha[glyph_row + min_gx..glyph_row + max_gx];
                    let px_start = (x_off + min_gx) * 4;
                    let px_end = (x_off + max_gx) * 4;
                    let canvas_row = &mut canvas_row_full[px_start..px_end];
                    composite_alpha_row(
                        canvas_row,
                        alpha_row,
                        color_r,
                        color_g,
                        color_b,
                        color_a,
                        is_opaque_color,
                    );
                });
        } else {
            for (row_idx, canvas_row_full) in canvas_region.chunks_mut(canvas_stride).enumerate() {
                let gy = min_gy as usize + row_idx;
                let glyph_row = gy * glyph_width_usize;
                let alpha_row = &glyph_mask.alpha[glyph_row + min_gx..glyph_row + max_gx];
                let px_start = (x_off + min_gx) * 4;
                let px_end = (x_off + max_gx) * 4;
                let canvas_row = &mut canvas_row_full[px_start..px_end];
                composite_alpha_row(
                    canvas_row,
                    alpha_row,
                    color_r,
                    color_g,
                    color_b,
                    color_a,
                    is_opaque_color,
                );
            }
        }
    }

    /// Downsample a supersampled image using box filtering with optimized memory access.
    fn downsample(&self, ss_pixels: &[u8], ss_width: u32, ss_height: u32, factor: u32) -> Vec<u8> {
        let out_width = ss_width / factor;
        let out_height = ss_height / factor;
        let out_size = (out_width * out_height) as usize;

        // Specialized fast path for 2x supersampling (most common case).
        match factor {
            2 => return self.downsample_2x(ss_pixels, ss_width, out_width, out_size),
            1 => return ss_pixels.to_vec(),
            _ => {}
        }

        // General path for other factors
        let factor_usize = factor as usize;
        let factor_sq = factor * factor;
        let out_row_bytes = out_width as usize * 4;
        let ss_row_bytes = ss_width as usize * 4;
        let mut result = vec![0u8; out_size * 4];

        result
            .par_chunks_mut(out_row_bytes)
            .enumerate()
            .for_each(|(out_y, row)| {
                let base_y = out_y * factor_usize;

                for out_x in 0..out_width as usize {
                    let mut r_sum: u32 = 0;
                    let mut g_sum: u32 = 0;
                    let mut b_sum: u32 = 0;
                    let mut a_sum: u32 = 0;

                    let base_x = out_x * factor_usize;

                    for dy in 0..factor_usize {
                        let src_row = (base_y + dy) * ss_row_bytes + base_x * 4;
                        for dx in 0..factor_usize {
                            let idx = src_row + dx * 4;
                            r_sum += ss_pixels[idx] as u32;
                            g_sum += ss_pixels[idx + 1] as u32;
                            b_sum += ss_pixels[idx + 2] as u32;
                            a_sum += ss_pixels[idx + 3] as u32;
                        }
                    }

                    let out_idx = out_x * 4;
                    row[out_idx] = (r_sum / factor_sq) as u8;
                    row[out_idx + 1] = (g_sum / factor_sq) as u8;
                    row[out_idx + 2] = (b_sum / factor_sq) as u8;
                    row[out_idx + 3] = (a_sum / factor_sq) as u8;
                }
            });

        result
    }

    /// Optimized 2x downsampling with parallel rows and unrolled inner loops.
    fn downsample_2x(
        &self,
        ss_pixels: &[u8],
        ss_width: u32,
        out_width: u32,
        out_size: usize,
    ) -> Vec<u8> {
        let out_row_bytes = out_width as usize * 4;
        let ss_row_bytes = ss_width as usize * 4;
        let mut result = vec![0u8; out_size * 4];

        result
            .par_chunks_mut(out_row_bytes)
            .enumerate()
            .for_each(|(out_y, row)| {
                let row0_start = out_y * 2 * ss_row_bytes;
                let row1_start = row0_start + ss_row_bytes;
                let row0 = &ss_pixels[row0_start..row0_start + ss_row_bytes];
                let row1 = &ss_pixels[row1_start..row1_start + ss_row_bytes];

                let out_width_usize = out_width as usize;
                let simd_width = out_width_usize & !3;
                let mut out_x = 0usize;

                while out_x < simd_width {
                    let mut base = out_x * 8;
                    for _ in 0..4 {
                        let dst = out_x * 4;
                        row[dst] = ((row0[base] as u16
                            + row0[base + 4] as u16
                            + row1[base] as u16
                            + row1[base + 4] as u16)
                            >> 2) as u8;
                        row[dst + 1] = ((row0[base + 1] as u16
                            + row0[base + 5] as u16
                            + row1[base + 1] as u16
                            + row1[base + 5] as u16)
                            >> 2) as u8;
                        row[dst + 2] = ((row0[base + 2] as u16
                            + row0[base + 6] as u16
                            + row1[base + 2] as u16
                            + row1[base + 6] as u16)
                            >> 2) as u8;
                        row[dst + 3] = ((row0[base + 3] as u16
                            + row0[base + 7] as u16
                            + row1[base + 3] as u16
                            + row1[base + 7] as u16)
                            >> 2) as u8;
                        out_x += 1;
                        base += 8;
                    }
                }

                while out_x < out_width_usize {
                    let base = out_x * 8;
                    let dst = out_x * 4;
                    row[dst] = ((row0[base] as u16
                        + row0[base + 4] as u16
                        + row1[base] as u16
                        + row1[base + 4] as u16)
                        >> 2) as u8;
                    row[dst + 1] = ((row0[base + 1] as u16
                        + row0[base + 5] as u16
                        + row1[base + 1] as u16
                        + row1[base + 5] as u16)
                        >> 2) as u8;
                    row[dst + 2] = ((row0[base + 2] as u16
                        + row0[base + 6] as u16
                        + row1[base + 2] as u16
                        + row1[base + 6] as u16)
                        >> 2) as u8;
                    row[dst + 3] = ((row0[base + 3] as u16
                        + row0[base + 7] as u16
                        + row1[base + 3] as u16
                        + row1[base + 7] as u16)
                        >> 2) as u8;
                    out_x += 1;
                }
            });

        result
    }
}

/// Composite a row of alpha coverage values onto a canvas row.
///
/// Shared inner loop used by both `composite_glyph_mask` and `composite_alpha_buffer`.
/// The `canvas_row` slice must be exactly `alpha_row.len() * 4` bytes.
#[inline]
fn composite_alpha_row(
    canvas_row: &mut [u8],
    alpha_row: &[u8],
    color_r: u8,
    color_g: u8,
    color_b: u8,
    color_a: u32,
    is_opaque_color: bool,
) {
    if is_opaque_color {
        for (i, &coverage) in alpha_row.iter().enumerate() {
            if coverage == 0 {
                continue;
            }
            let dst_idx = i * 4;
            blend_rgba_over(
                canvas_row,
                dst_idx,
                color_r,
                color_g,
                color_b,
                coverage as u32,
            );
        }
    } else {
        for (i, &coverage) in alpha_row.iter().enumerate() {
            if coverage == 0 {
                continue;
            }
            let src_alpha = div255(coverage as u32 * color_a);
            if src_alpha == 0 {
                continue;
            }
            let dst_idx = i * 4;
            blend_rgba_over(canvas_row, dst_idx, color_r, color_g, color_b, src_alpha);
        }
    }
}

/// Compute a reduced supersampled icon size that fits the canvas bounds.
///
/// Returns `None` if no downscale is needed or if no further reduction is possible.
#[inline]
fn scaled_icon_size_to_fit_canvas(
    ss_icon_size: u32,
    glyph_width: u32,
    glyph_height: u32,
    canvas_width: u32,
    canvas_height: u32,
) -> Option<u32> {
    if glyph_width <= canvas_width && glyph_height <= canvas_height {
        return None;
    }

    let fit_x = canvas_width as f64 / glyph_width as f64;
    let fit_y = canvas_height as f64 / glyph_height as f64;
    let fit_scale = fit_x.min(fit_y) * 0.95;
    let scaled = ((ss_icon_size as f64) * fit_scale).floor().max(1.0) as u32;

    if scaled < ss_icon_size {
        Some(scaled)
    } else {
        None
    }
}

/// Fast approximate division by 255: `(x + 128) * 257 >> 16` is exact for all u8*u8 products.
#[inline(always)]
fn div255(x: u32) -> u32 {
    (x + 128 + ((x + 128) >> 8)) >> 8
}

/// Blend source RGBA over destination RGBA in-place at `dst_idx`.
#[inline(always)]
fn blend_rgba_over(canvas: &mut [u8], dst_idx: usize, src_r: u8, src_g: u8, src_b: u8, src_a: u32) {
    if src_a == 255 {
        canvas[dst_idx] = src_r;
        canvas[dst_idx + 1] = src_g;
        canvas[dst_idx + 2] = src_b;
        canvas[dst_idx + 3] = 255;
        return;
    }
    let inv_src_alpha = 255 - src_a;
    canvas[dst_idx] = div255(src_r as u32 * src_a + canvas[dst_idx] as u32 * inv_src_alpha) as u8;
    canvas[dst_idx + 1] =
        div255(src_g as u32 * src_a + canvas[dst_idx + 1] as u32 * inv_src_alpha) as u8;
    canvas[dst_idx + 2] =
        div255(src_b as u32 * src_a + canvas[dst_idx + 2] as u32 * inv_src_alpha) as u8;
    canvas[dst_idx + 3] = (src_a + div255(canvas[dst_idx + 3] as u32 * inv_src_alpha)) as u8;
}

/// Bilinear interpolation sampler for alpha buffers using fixed-point arithmetic.
/// Uses 8-bit fractional precision (256 = 1.0) to avoid floating-point ops in
/// the inner loop.
#[inline(always)]
fn bilinear_sample_alpha(buffer: &[u8], width: usize, x: f32, y: f32) -> u8 {
    let x0 = x as usize;
    let y0 = y as usize;

    // 8-bit fractional parts (0..=255)
    let fx = ((x - x0 as f32) * 256.0) as u32;
    let fy = ((y - y0 as f32) * 256.0) as u32;
    let inv_fx = 256 - fx;
    let inv_fy = 256 - fy;

    let row0 = y0 * width + x0;
    let row1 = row0 + width;

    // Weighted sum with 16-bit precision, then shift back
    let val = inv_fy * (inv_fx * buffer[row0] as u32 + fx * buffer[row0 + 1] as u32)
        + fy * (inv_fx * buffer[row1] as u32 + fx * buffer[row1 + 1] as u32);

    ((val + 32768) >> 16) as u8
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

    #[test]
    fn test_scaled_icon_size_to_fit_canvas_no_change_when_fit() {
        let scaled = scaled_icon_size_to_fit_canvas(1024, 800, 700, 1024, 1024);
        assert_eq!(scaled, None);
    }

    #[test]
    fn test_scaled_icon_size_to_fit_canvas_downscales_oversized_glyph() {
        let scaled = scaled_icon_size_to_fit_canvas(1000, 1200, 800, 1000, 1000);
        let scaled_value = scaled.expect("oversized glyph should trigger downscaling");
        assert!(scaled_value < 1000);
    }
}
