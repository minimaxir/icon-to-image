//! Python bindings for icon_to_image using PyO3.
//!
//! This module provides a Python-friendly API for the icon rendering functionality.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Borrowed;

use crate::color::Color;
use crate::css_parser::FontStyle as RustFontStyle;
use crate::encoder::{encode, ImageFormat};
use crate::renderer::{
    HorizontalAnchor as RustHAnchor, IconRenderer as RustRenderer, RenderConfig as RustConfig,
    VerticalAnchor as RustVAnchor,
};

/// Convert a Rust error to a Python exception.
fn to_py_err(e: crate::IconFontError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Calculate safe icon size with sanity check.
///
/// If icon_size is None, defaults to 95% of the smaller canvas dimension.
/// If icon_size exceeds either canvas dimension, clamps to 95% of the smaller dimension.
fn calculate_safe_icon_size(icon_size: Option<u32>, canvas_width: u32, canvas_height: u32) -> u32 {
    let smaller_dim = canvas_width.min(canvas_height);
    // 95% of smaller dimension, ensuring at least 1px
    let max_safe_size = ((smaller_dim as f64) * 0.95) as u32;
    let max_safe_size = max_safe_size.max(1);

    match icon_size {
        Some(size) if size > smaller_dim => {
            // Icon size exceeds canvas, clamp to 95% of smaller dimension
            max_safe_size
        }
        Some(size) => size,
        None => {
            // Default: 95% of smaller dimension (for 512x512, this is ~486)
            max_safe_size
        }
    }
}

/// Common parameters for building a RenderConfig.
///
/// This struct consolidates the shared parameters used across render_icon,
/// render_icon_bytes, and save_icon methods to avoid code duplication.
struct RenderParams {
    canvas_width: u32,
    canvas_height: u32,
    icon_size: Option<u32>,
    supersample: u32,
    icon_color: Color,
    background_color: Color,
    horizontal_anchor: RustHAnchor,
    vertical_anchor: RustVAnchor,
    offset_x: i32,
    offset_y: i32,
    rotate: f64,
}

impl RenderParams {
    /// Build a RustConfig from these parameters.
    fn to_config(&self) -> RustConfig {
        let effective_icon_size =
            calculate_safe_icon_size(self.icon_size, self.canvas_width, self.canvas_height);

        RustConfig {
            canvas_width: self.canvas_width,
            canvas_height: self.canvas_height,
            icon_size: effective_icon_size,
            supersample_factor: self.supersample.max(1),
            icon_color: self.icon_color,
            background_color: self.background_color,
            horizontal_anchor: self.horizontal_anchor,
            vertical_anchor: self.vertical_anchor,
            offset_x: self.offset_x,
            offset_y: self.offset_y,
            rotate: self.rotate,
        }
    }
}

/// Render an icon with optional style override.
///
/// Consolidates the render dispatch logic used across multiple Python methods.
fn render_with_optional_style(
    renderer: &RustRenderer,
    name: &str,
    style: Option<RustFontStyle>,
    config: &RustConfig,
) -> PyResult<(u32, u32, Vec<u8>)> {
    match style {
        Some(font_style) => renderer
            .render_with_style(name, font_style, config)
            .map_err(to_py_err),
        None => renderer.render(name, config).map_err(to_py_err),
    }
}

/// Parse a color from a Python object (hex string or RGB/RGBA tuple).
///
/// Returns the parsed Color or a PyErr if the format is invalid.
fn parse_color_from_pyobject(ob: &Bound<'_, PyAny>) -> PyResult<Color> {
    // Try to extract as string (hex color)
    if let Ok(s) = ob.extract::<String>() {
        return Color::from_hex(&s).map_err(to_py_err);
    }
    // Try to extract as 3-tuple (RGB)
    if let Ok((r, g, b)) = ob.extract::<(u8, u8, u8)>() {
        return Ok(Color::rgb(r, g, b));
    }
    // Try to extract as 4-tuple (RGBA)
    if let Ok((r, g, b, a)) = ob.extract::<(u8, u8, u8, u8)>() {
        return Ok(Color::rgba(r, g, b, a));
    }
    Err(PyValueError::new_err(
        "color must be a hex string (e.g., '#FF0000') or RGB/RGBA tuple (e.g., (255, 0, 0) or (255, 0, 0, 128))",
    ))
}

/// Python-exposed horizontal anchor enum.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HorizontalAnchor {
    Left,
    Center,
    Right,
}

impl From<HorizontalAnchor> for RustHAnchor {
    fn from(val: HorizontalAnchor) -> Self {
        match val {
            HorizontalAnchor::Left => RustHAnchor::Left,
            HorizontalAnchor::Center => RustHAnchor::Center,
            HorizontalAnchor::Right => RustHAnchor::Right,
        }
    }
}

/// Flexible horizontal anchor that accepts either enum or string.
/// Strings accepted: "left", "center", "right" (case-insensitive)
#[derive(Clone, Copy)]
struct FlexHAnchor(RustHAnchor);

impl Default for FlexHAnchor {
    fn default() -> Self {
        Self(RustHAnchor::Center)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FlexHAnchor {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Try to extract as HorizontalAnchor enum first
        if let Ok(anchor) = ob.extract::<HorizontalAnchor>() {
            return Ok(Self(anchor.into()));
        }
        // Try to extract as string
        if let Ok(s) = ob.extract::<String>() {
            return match s.to_lowercase().as_str() {
                "left" => Ok(Self(RustHAnchor::Left)),
                "center" => Ok(Self(RustHAnchor::Center)),
                "right" => Ok(Self(RustHAnchor::Right)),
                _ => Err(PyValueError::new_err(format!(
                    "Invalid horizontal anchor: '{}'. Expected 'left', 'center', or 'right'",
                    s
                ))),
            };
        }
        Err(PyValueError::new_err(
            "horizontal_anchor must be a HorizontalAnchor enum or string ('left', 'center', 'right')",
        ))
    }
}

/// Python-exposed vertical anchor enum.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VerticalAnchor {
    Top,
    Center,
    Bottom,
}

impl From<VerticalAnchor> for RustVAnchor {
    fn from(val: VerticalAnchor) -> Self {
        match val {
            VerticalAnchor::Top => RustVAnchor::Top,
            VerticalAnchor::Center => RustVAnchor::Center,
            VerticalAnchor::Bottom => RustVAnchor::Bottom,
        }
    }
}

/// Flexible vertical anchor that accepts either enum or string.
/// Strings accepted: "top", "center", "bottom" (case-insensitive)
#[derive(Clone, Copy)]
struct FlexVAnchor(RustVAnchor);

impl Default for FlexVAnchor {
    fn default() -> Self {
        Self(RustVAnchor::Center)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FlexVAnchor {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Try to extract as VerticalAnchor enum first
        if let Ok(anchor) = ob.extract::<VerticalAnchor>() {
            return Ok(Self(anchor.into()));
        }
        // Try to extract as string
        if let Ok(s) = ob.extract::<String>() {
            return match s.to_lowercase().as_str() {
                "top" => Ok(Self(RustVAnchor::Top)),
                "center" => Ok(Self(RustVAnchor::Center)),
                "bottom" => Ok(Self(RustVAnchor::Bottom)),
                _ => Err(PyValueError::new_err(format!(
                    "Invalid vertical anchor: '{}'. Expected 'top', 'center', or 'bottom'",
                    s
                ))),
            };
        }
        Err(PyValueError::new_err(
            "vertical_anchor must be a VerticalAnchor enum or string ('top', 'center', 'bottom')",
        ))
    }
}

/// Python-exposed font style enum for selecting icon style.
///
/// Font Awesome icons come in different styles:
/// - Solid: Filled icons (most common)
/// - Regular: Outlined icons
/// - Brands: Company/brand logos
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FontStyle {
    Solid,
    Regular,
    Brands,
}

impl From<FontStyle> for RustFontStyle {
    fn from(val: FontStyle) -> Self {
        match val {
            FontStyle::Solid => RustFontStyle::Solid,
            FontStyle::Regular => RustFontStyle::Regular,
            FontStyle::Brands => RustFontStyle::Brands,
        }
    }
}

/// Flexible optional font style that accepts None, enum, or string.
/// Strings accepted: "solid", "regular", "brands" (case-insensitive)
/// None means use the default style for the icon (solid for most icons, brands for brand icons).
#[derive(Clone, Copy, Default)]
struct FlexOptionalFontStyle(Option<RustFontStyle>);

impl<'a, 'py> FromPyObject<'a, 'py> for FlexOptionalFontStyle {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Check for None first
        if ob.is_none() {
            return Ok(Self(None));
        }
        // Try to extract as FontStyle enum
        if let Ok(style) = ob.extract::<FontStyle>() {
            return Ok(Self(Some(style.into())));
        }
        // Try to extract as string
        if let Ok(s) = ob.extract::<String>() {
            return match s.to_lowercase().as_str() {
                "solid" => Ok(Self(Some(RustFontStyle::Solid))),
                "regular" => Ok(Self(Some(RustFontStyle::Regular))),
                "brands" => Ok(Self(Some(RustFontStyle::Brands))),
                _ => Err(PyValueError::new_err(format!(
                    "Invalid font style: '{}'. Expected 'solid', 'regular', or 'brands'",
                    s
                ))),
            };
        }
        Err(PyValueError::new_err(
            "style must be None, a FontStyle enum, or string ('solid', 'regular', 'brands')",
        ))
    }
}

/// Python-exposed image format enum.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Png,
    WebP,
}

impl From<OutputFormat> for ImageFormat {
    fn from(val: OutputFormat) -> Self {
        match val {
            OutputFormat::Png => ImageFormat::Png,
            OutputFormat::WebP => ImageFormat::WebP,
        }
    }
}

/// Flexible output format that accepts either enum or string.
/// Strings accepted: "png", "webp" (case-insensitive)
#[derive(Clone, Copy)]
struct FlexOutputFormat(ImageFormat);

impl Default for FlexOutputFormat {
    fn default() -> Self {
        Self(ImageFormat::Png)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FlexOutputFormat {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Try to extract as OutputFormat enum first
        if let Ok(fmt) = ob.extract::<OutputFormat>() {
            return Ok(Self(fmt.into()));
        }
        // Try to extract as string
        if let Ok(s) = ob.extract::<String>() {
            return match s.to_lowercase().as_str() {
                "png" => Ok(Self(ImageFormat::Png)),
                "webp" => Ok(Self(ImageFormat::WebP)),
                _ => Err(PyValueError::new_err(format!(
                    "Invalid output format: '{}'. Expected 'png' or 'webp'",
                    s
                ))),
            };
        }
        Err(PyValueError::new_err(
            "output_format must be an OutputFormat enum or string ('png', 'webp')",
        ))
    }
}

/// Flexible color that accepts either hex string or RGB/RGBA tuple.
/// - String: Hex color like "#FF0000" or "FF0000"
/// - 3-tuple: (R, G, B) with values 0-255
/// - 4-tuple: (R, G, B, A) with values 0-255
#[derive(Clone, Copy)]
struct FlexColor(Color);

impl<'a, 'py> FromPyObject<'a, 'py> for FlexColor {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        parse_color_from_pyobject(&ob).map(Self)
    }
}

/// Flexible optional color that accepts None, hex string, or RGB/RGBA tuple.
#[derive(Clone, Copy)]
struct FlexOptionalColor(Option<Color>);

impl<'a, 'py> FromPyObject<'a, 'py> for FlexOptionalColor {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if ob.is_none() {
            return Ok(Self(None));
        }
        parse_color_from_pyobject(&ob).map(|c| Self(Some(c)))
    }
}

/// Icon renderer that loads Font Awesome fonts and renders icons to images.
///
/// By default, uses embedded Font Awesome assets (no external files needed).
/// Optionally, you can provide a custom assets directory path.
///
/// Args:
///     assets_dir: Optional path to directory containing fa-solid.otf, fa-regular.otf,
///                 fa-brands.otf, and fontawesome.css. If not provided, uses embedded assets.
///
/// Example:
///     >>> from icon_to_image import IconRenderer
///     >>> # Use embedded assets (recommended)
///     >>> renderer = IconRenderer()
///     >>> # Or use custom assets from a directory
///     >>> renderer = IconRenderer("./custom_assets")
///     >>>
///     >>> # render_icon() returns a PIL.Image (requires Pillow)
///     >>> img = renderer.render_icon("heart", icon_color="#FF0000")
///     >>> img.save("heart.png")
///     >>>
///     >>> # render_icon_bytes() returns raw PNG/WebP bytes (no Pillow needed)
///     >>> png_data = renderer.render_icon_bytes("heart", icon_color="#FF0000")
///     >>> with open("heart.png", "wb") as f:
///     ...     f.write(png_data)
#[pyclass]
pub struct IconRenderer {
    inner: RustRenderer,
}

#[pymethods]
impl IconRenderer {
    /// Create a new IconRenderer.
    ///
    /// Args:
    ///     assets_dir: Optional path to the assets directory containing font files and CSS.
    ///                 If not provided, uses embedded Font Awesome assets.
    #[new]
    #[pyo3(signature = (assets_dir = None))]
    fn new(assets_dir: Option<&str>) -> PyResult<Self> {
        let inner = match assets_dir {
            Some(path) => RustRenderer::from_path(path).map_err(to_py_err)?,
            None => RustRenderer::new().map_err(to_py_err)?,
        };
        Ok(Self { inner })
    }

    /// Check if an icon exists by name.
    ///
    /// Args:
    ///     name: Icon name (e.g., "heart", "github", "fa-star")
    ///
    /// Returns:
    ///     True if the icon exists, False otherwise
    fn has_icon(&self, name: &str) -> bool {
        self.inner.has_icon(name)
    }

    /// Get the number of available icons.
    ///
    /// Returns:
    ///     Number of icons loaded from CSS
    fn icon_count(&self) -> usize {
        self.inner.icon_count()
    }

    /// List all available icon names.
    ///
    /// Returns:
    ///     List of icon names (without "fa-" prefix)
    fn list_icons(&self) -> Vec<String> {
        self.inner
            .list_icons()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Render an icon to a PIL.Image.
    ///
    /// This method requires the Pillow library to be installed. If you need raw
    /// bytes instead, use `render_icon_bytes()`.
    ///
    /// Args:
    ///     name: Icon name (e.g., "heart", "github")
    ///     canvas_width: Output image width in pixels (default: 512)
    ///     canvas_height: Output image height in pixels (default: 512)
    ///     icon_size: Icon size in pixels (default: 486, about 95% of 512 for margin)
    ///     supersample: Supersampling factor for antialiasing (default: 2)
    ///     icon_color: Icon color as hex string (e.g., "#FF0000") or RGB/RGBA tuple
    ///         (e.g., (255, 0, 0) or (255, 0, 0, 128)). Default: "#000000"
    ///     background_color: Background color as hex string or RGB/RGBA tuple,
    ///         or None for transparent. Default: "#FFFFFF" (white)
    ///     horizontal_anchor: Horizontal alignment - HorizontalAnchor enum or string
    ///         ("left", "center", "right"). Default: "center"
    ///     vertical_anchor: Vertical alignment - VerticalAnchor enum or string
    ///         ("top", "center", "bottom"). Default: "center"
    ///     offset_x: Horizontal pixel offset from anchor (default: 0)
    ///     offset_y: Vertical pixel offset from anchor (default: 0)
    ///     rotate: Rotation angle in degrees (default: 0). Positive values rotate
    ///         clockwise, negative values rotate counter-clockwise.
    ///     style: Font style - FontStyle enum or string ("solid", "regular", "brands").
    ///         Default: None (uses the icon's default style: solid for most icons,
    ///         brands for brand icons like "github"). Use "solid" to force filled icons,
    ///         "regular" for outlined icons.
    ///
    /// Returns:
    ///     PIL.Image.Image object in RGBA mode
    ///
    /// Raises:
    ///     ImportError: If Pillow is not installed
    ///     ValueError: If icon not found or invalid parameters
    #[pyo3(signature = (
        name,
        canvas_width = 512,
        canvas_height = 512,
        icon_size = None,
        supersample = 2,
        icon_color = FlexColor(Color::black()),
        background_color = FlexOptionalColor(Some(Color::white())),
        horizontal_anchor = FlexHAnchor::default(),
        vertical_anchor = FlexVAnchor::default(),
        offset_x = 0,
        offset_y = 0,
        rotate = 0.0,
        style = FlexOptionalFontStyle::default()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn render_icon<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        canvas_width: u32,
        canvas_height: u32,
        icon_size: Option<u32>,
        supersample: u32,
        icon_color: FlexColor,
        background_color: FlexOptionalColor,
        horizontal_anchor: FlexHAnchor,
        vertical_anchor: FlexVAnchor,
        offset_x: i32,
        offset_y: i32,
        rotate: f64,
        style: FlexOptionalFontStyle,
    ) -> PyResult<Py<PyAny>> {
        let params = RenderParams {
            canvas_width,
            canvas_height,
            icon_size,
            supersample,
            icon_color: icon_color.0,
            background_color: background_color.0.unwrap_or_else(Color::transparent),
            horizontal_anchor: horizontal_anchor.0,
            vertical_anchor: vertical_anchor.0,
            offset_x,
            offset_y,
            rotate,
        };
        let config = params.to_config();
        let (width, height, pixels) =
            render_with_optional_style(&self.inner, name, style.0, &config)?;

        // Import PIL.Image and create an image from raw RGBA bytes
        let pil_image = py.import("PIL.Image").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "Pillow is required for render_icon(). Install it with: pip install Pillow\n\
                 Alternatively, use render_icon_bytes() to get raw PNG/WebP bytes without Pillow.",
            )
        })?;

        // Create PIL Image from raw RGBA bytes using frombytes()
        // frombytes(mode, size, data) creates an image from raw pixel data
        let bytes = PyBytes::new(py, &pixels);
        let size = (width, height);
        let img = pil_image.call_method1("frombytes", ("RGBA", size, bytes))?;

        Ok(img.unbind())
    }

    /// Render an icon to encoded image bytes (PNG or WebP).
    ///
    /// This method returns raw encoded image bytes without requiring Pillow.
    /// Use `render_icon()` if you need a PIL.Image object instead.
    ///
    /// Args:
    ///     name: Icon name (e.g., "heart", "github")
    ///     canvas_width: Output image width in pixels (default: 512)
    ///     canvas_height: Output image height in pixels (default: 512)
    ///     icon_size: Icon size in pixels (default: 486, about 95% of 512 for margin)
    ///     supersample: Supersampling factor for antialiasing (default: 2)
    ///     icon_color: Icon color as hex string (e.g., "#FF0000") or RGB/RGBA tuple
    ///         (e.g., (255, 0, 0) or (255, 0, 0, 128)). Default: "#000000"
    ///     background_color: Background color as hex string or RGB/RGBA tuple,
    ///         or None for transparent. Default: "#FFFFFF" (white)
    ///     horizontal_anchor: Horizontal alignment - HorizontalAnchor enum or string
    ///         ("left", "center", "right"). Default: "center"
    ///     vertical_anchor: Vertical alignment - VerticalAnchor enum or string
    ///         ("top", "center", "bottom"). Default: "center"
    ///     offset_x: Horizontal pixel offset from anchor (default: 0)
    ///     offset_y: Vertical pixel offset from anchor (default: 0)
    ///     rotate: Rotation angle in degrees (default: 0). Positive values rotate
    ///         clockwise, negative values rotate counter-clockwise.
    ///     output_format: Output format - OutputFormat enum or string
    ///         ("png", "webp"). Default: "png"
    ///     style: Font style - FontStyle enum or string ("solid", "regular", "brands").
    ///         Default: None (uses the icon's default style).
    ///
    /// Returns:
    ///     Encoded image data as bytes
    ///
    /// Raises:
    ///     ValueError: If icon not found or invalid parameters
    #[pyo3(signature = (
        name,
        canvas_width = 512,
        canvas_height = 512,
        icon_size = None,
        supersample = 2,
        icon_color = FlexColor(Color::black()),
        background_color = FlexOptionalColor(Some(Color::white())),
        horizontal_anchor = FlexHAnchor::default(),
        vertical_anchor = FlexVAnchor::default(),
        offset_x = 0,
        offset_y = 0,
        rotate = 0.0,
        output_format = FlexOutputFormat::default(),
        style = FlexOptionalFontStyle::default()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn render_icon_bytes<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        canvas_width: u32,
        canvas_height: u32,
        icon_size: Option<u32>,
        supersample: u32,
        icon_color: FlexColor,
        background_color: FlexOptionalColor,
        horizontal_anchor: FlexHAnchor,
        vertical_anchor: FlexVAnchor,
        offset_x: i32,
        offset_y: i32,
        rotate: f64,
        output_format: FlexOutputFormat,
        style: FlexOptionalFontStyle,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let params = RenderParams {
            canvas_width,
            canvas_height,
            icon_size,
            supersample,
            icon_color: icon_color.0,
            background_color: background_color.0.unwrap_or_else(Color::transparent),
            horizontal_anchor: horizontal_anchor.0,
            vertical_anchor: vertical_anchor.0,
            offset_x,
            offset_y,
            rotate,
        };
        let config = params.to_config();
        let (width, height, pixels) =
            render_with_optional_style(&self.inner, name, style.0, &config)?;
        let encoded = encode(&pixels, width, height, output_format.0).map_err(to_py_err)?;

        Ok(PyBytes::new(py, &encoded))
    }

    /// Save an icon directly to a file.
    ///
    /// Args:
    ///     name: Icon name
    ///     path: Output file path (extension determines format: .png or .webp)
    ///     canvas_width: Output width (default: 512)
    ///     canvas_height: Output height (default: 512)
    ///     icon_size: Icon size (default: 486, about 95% of 512 for margin)
    ///     supersample: Supersampling factor (default: 2)
    ///     icon_color: Icon color as hex string or RGB/RGBA tuple (default: "#000000")
    ///     background_color: Background color as hex or RGB/RGBA tuple,
    ///         or None for transparent. Default: "#FFFFFF" (white)
    ///     rotate: Rotation angle in degrees (default: 0). Positive values rotate
    ///         clockwise, negative values rotate counter-clockwise.
    ///     style: Font style - FontStyle enum or string ("solid", "regular", "brands").
    ///         Default: None (uses the icon's default style).
    #[pyo3(signature = (
        name,
        path,
        canvas_width = 512,
        canvas_height = 512,
        icon_size = None,
        supersample = 2,
        icon_color = FlexColor(Color::black()),
        background_color = FlexOptionalColor(Some(Color::white())),
        rotate = 0.0,
        style = FlexOptionalFontStyle::default()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn save_icon(
        &self,
        name: &str,
        path: &str,
        canvas_width: u32,
        canvas_height: u32,
        icon_size: Option<u32>,
        supersample: u32,
        icon_color: FlexColor,
        background_color: FlexOptionalColor,
        rotate: f64,
        style: FlexOptionalFontStyle,
    ) -> PyResult<()> {
        let params = RenderParams {
            canvas_width,
            canvas_height,
            icon_size,
            supersample,
            icon_color: icon_color.0,
            background_color: background_color.0.unwrap_or_else(Color::transparent),
            horizontal_anchor: RustHAnchor::Center,
            vertical_anchor: RustVAnchor::Center,
            offset_x: 0,
            offset_y: 0,
            rotate,
        };
        let config = params.to_config();
        let (width, height, pixels) =
            render_with_optional_style(&self.inner, name, style.0, &config)?;
        crate::save_to_file(&pixels, width, height, path).map_err(to_py_err)?;

        Ok(())
    }
}

/// Python module definition.
#[pymodule]
fn icon_to_image(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IconRenderer>()?;
    m.add_class::<HorizontalAnchor>()?;
    m.add_class::<VerticalAnchor>()?;
    m.add_class::<OutputFormat>()?;
    m.add_class::<FontStyle>()?;
    Ok(())
}
