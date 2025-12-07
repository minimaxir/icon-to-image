# Icon to Image

A high-performance Rust library with Python bindings for rendering [Font Awesome](https://fontawesome.com) icons to images, with all the customizations you may want. This library natively includes the icons from Font Awesome 7.1.0 Free (solid, regular and brands).

Features:

- Extremely fast icon rendering due to `ab_glyph` (~10ms for a 512x512 image)
- Python bindings via PyO3
- No Python dependencies: `Pillow` is optional only if using an image object
- Supersampling for high-quality antialiased output (2x by default)
- PNG and WebP output formats
- Customizable icon and background colors (hex and RGB/RGBA), including transparency
- Flexible positioning in large canvases with anchors and offsets
- Command-line interface for both Rust and Python

_**Disclosure:** This library was mostly coded with the assistance of Claude 4.5 Opus. However, I personally have reviewed all code to ensure it is accurate, have added numerous tests and demo examples to ensure it works as both intended and advertised, and have edited documentation and comments to provide greater signal as to how the package operates. I have given this project the same care and attention as I would give a project I have written from scratch._

## Installation

### Python

```bash
pip install icon-to-image
```

### Rust

```bash
cargo install icon-to-image
```

Or, add to your `Cargo.toml`:

```toml
[dependencies]
icon-to-image = "0.1"
```

## Command-Line Interface

Both the Rust binary and Python package provide a CLI for rendering icons.

### Basic Usage

The list of included icons can be found on [Font Awesome's Icon Grid](https://fontawesome.com/search?ic=free-collection) and filtering to "Free" icons. The icon names do not use the `fa` prefix.

```bash
# View CLI documentation
icon-to-image --help

# Render a heart icon to PNG
icon-to-image heart heart.png

# Render with custom color and size
icon-to-image star star.png --color "#FFD700" --size 512

# Render with transparent background
icon-to-image github logo.png --color "#333333" --background transparent

# Output as WebP
icon-to-image rocket rocket.webp --color "#FF6B35"

# Render with rotation (45 degrees clockwise)
icon-to-image arrow-right rotated.png --rotate 45

# Render with counter-clockwise rotation
icon-to-image arrow-up arrow.png --rotate -90
```

### List and Search Icons

```bash
# List all available icons
icon-to-image list

# Search for icons matching a pattern
icon-to-image search arrow
```

## Python Usage

### Basic Rendering

To use `render_icon()`, you will also need to have `Pillow` installed.

```python
from icon_to_image import IconRenderer

renderer = IconRenderer()

# Render a heart icon with default settings (512x512, solid weight, black on white)
# render_icon() returns a PIL.Image directly
img = renderer.render_icon("heart")
img.save("heart.png")
```

![](docs/heart.webp)

### Custom Colors

```python
# Red heart on default white background
red_heart = renderer.render_icon(
    "heart",
    icon_color="#FF0000",
)

# Blue star on yellow background
blue_star = renderer.render_icon(
    "star",
    icon_color="#0066CC",
    background_color="#FFD700",
)

# Semitransparent gray ghost on transparent background
gray_ghost = renderer.render_icon(
    "ghost",
    icon_color=(128, 128, 128, 128),
    background_color=None,
)
```

![](docs/combined_icons.webp)

### Brand Icons

```python
github = renderer.render_icon("github", icon_color="#333333", background_color=None)
python = renderer.render_icon("python", icon_color="#3776AB", background_color=None)
```

![](docs/combined_icons_brands.webp)

### Custom Sizes

```python
# Small icon with padding (128px icon on 256px canvas)
small = renderer.render_icon(
    "rocket",
    canvas_width=256,
    canvas_height=256,
    icon_size=128,
    icon_color="#FF6B35",
)
```

![](docs/rocket_small.webp)

### Icon Positioning

```python
# Position icon at bottom-right with padding
badge = renderer.render_icon(
    "check",
    canvas_width=256,
    canvas_height=256,
    icon_size=48,
    horizontal_anchor="right",
    vertical_anchor="bottom",
    offset_x=-16,
    offset_y=-16,
    icon_color="#4CAF50",
    background_color="#E8F5E9",
)
```

![](docs/check_badge.webp)

### Icon Rotation

```python
# Rotate an arrow 45 degrees clockwise
rotated = renderer.render_icon(
    "arrow-right",
    rotate=45,  # Positive = clockwise
    icon_color="#E91E63",
)

# Rotate counter-clockwise
rotated_ccw = renderer.render_icon(
    "egg",
    rotate=-30,  # Negative = counter-clockwise
    icon_color="#2196F3",
)
```

![](docs/combined_icons_rotated.webp)

### Font Styles

Some Font Awesome icons have different styles. You can explicitly specify which style to use:

```python
# Solid style (filled) - default for most icons
solid_heart = renderer.render_icon("heart", style="solid")

# Regular style (outlined)
regular_heart = renderer.render_icon("heart", style="regular")
```

![](docs/combined_icons_styles.webp)

### render_icon() and render_icon_bytes()

`render_icon()` returns a `PIL.Image` directly. `render_icon_bytes()` returns raw encoded bytes (PNG or WebP) without requiring Pillow, which can be useful for tasks such as APIs.

```python
# render_icon() returns PIL.Image (requires Pillow)
img = renderer.render_icon("heart", icon_color="#FF0000")
img.save("heart.png")  # Use PIL's save method

# render_icon_bytes() returns raw bytes (no Pillow needed)
png_data = renderer.render_icon_bytes("heart", icon_color="#FF0000")
with open("heart.png", "wb") as f:
    f.write(png_data)

# render_icon_bytes() supports output format selection
webp_data = renderer.render_icon_bytes("heart", output_format="webp")
```

### Save Directly to File

The `save_icon()` method writes the image to disk directly from Rust for maximum performance.

```python
renderer.save_icon("check-circle", "check.png", icon_color="#4CAF50")
renderer.save_icon("times-circle", "times.webp", icon_color="#F44336")
```

### Query Available Icons

```python
# Get total icon count
print(f"Total icons: {renderer.icon_count()}")

# List all icons
all_icons = renderer.list_icons()

# Search for specific icons
arrow_icons = [name for name in all_icons if "arrow" in name]

# Check if an icon exists
if renderer.has_icon("heart"):
    print("Heart icon is available")
```

## Rust Usage

```rust
use icon_to_image::{IconRenderer, RenderConfig, Color, ImageFormat, encode};

fn main() -> Result<(), icon_to_image::IconFontError> {
    // Use embedded assets
    let renderer = IconRenderer::new()?;

    let config = RenderConfig::new()
        .canvas_size(1024, 1024)
        .icon_size(800)
        .icon_color(Color::from_hex("#FF5733")?)
        .background_color(Color::transparent());

    let (width, height, pixels) = renderer.render("heart", &config)?;
    let png_data = encode(&pixels, width, height, ImageFormat::Png)?;
    std::fs::write("heart.png", png_data)?;

    Ok(())
}
```

## API Reference

### Python

#### Parameters

Applies to both `render_icon()` and `render_icon_bytes()`:

| Parameter           | Type           | Default       | Description                         |
| ------------------- | -------------- | ------------- | ----------------------------------- |
| `name`              | str            | required      | Icon name                           |
| `canvas_width`      | int            | 512           | Output width in pixels              |
| `canvas_height`     | int            | 512           | Output height in pixels             |
| `icon_size`         | int            | 95% of canvas | Icon size in pixels                 |
| `supersample`       | int            | 2             | Antialiasing factor (1, 2, or 4)    |
| `icon_color`        | str/tuple      | "#000000"     | Hex or RGB(A) tuple                 |
| `background_color`  | str/tuple/None | "#FFFFFF"     | Hex, RGB(A), or None                |
| `horizontal_anchor` | str            | "center"      | "left", "center", "right"           |
| `vertical_anchor`   | str            | "center"      | "top", "center", "bottom"           |
| `offset_x`          | int            | 0             | Horizontal offset in pixels         |
| `offset_y`          | int            | 0             | Vertical offset in pixels           |
| `rotate`            | float          | 0             | Rotation in degrees (+ = clockwise) |
| `style`             | str/None       | None          | "solid", "regular", or "brands"     |

## Notes

- Normally I would use `fontdue` for rendering the glyphs as it's more performant, however `fontdue` does not have the ability to render curves which is very important for most icons.
- This library technically supports using provided Font Awesome files (e.g. the Pro variants) but I do not have access to Pro for testing.

## License

MIT

Font Awesome CSS included per its MIT License.

Font Awesome font files included per their SIL Open Font 1.1 License.
