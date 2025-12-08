"""Type stubs for icon_to_image."""

from enum import IntEnum
from typing import Optional, Union

# Type alias for flexible color input
ColorType = Union[str, tuple[int, int, int], tuple[int, int, int, int]]

# Type alias for flexible anchor input
HorizontalAnchorType = Union["HorizontalAnchor", str]
VerticalAnchorType = Union["VerticalAnchor", str]

# Type alias for flexible format input
OutputFormatType = Union["OutputFormat", str]

class HorizontalAnchor(IntEnum):
    """Horizontal anchor position for icon placement."""
    Left = ...
    Center = ...
    Right = ...

class VerticalAnchor(IntEnum):
    """Vertical anchor position for icon placement."""
    Top = ...
    Center = ...
    Bottom = ...

class OutputFormat(IntEnum):
    """Output image format."""
    Png = ...
    WebP = ...

class IconRenderer:
    """
    Icon renderer that loads Font Awesome fonts and renders icons to images.

    By default, uses embedded Font Awesome assets (no external files needed).
    Optionally, you can provide a custom assets directory path.

    Args:
        assets_dir: Optional path to directory containing fa-solid.otf, fa-regular.otf,
                    fa-brands.otf, and fontawesome.css. If not provided, uses embedded assets.

    Example:
        >>> from icon_to_image import IconRenderer
        >>> # Use embedded assets (recommended)
        >>> renderer = IconRenderer()
        >>> # Or use custom assets from a directory
        >>> renderer = IconRenderer("./custom_assets")
        >>> png_data = renderer.render_icon("heart", icon_color="#FF0000")
    """

    def __init__(self, assets_dir: Optional[str] = None) -> None: ...

    def has_icon(self, name: str) -> bool:
        """
        Check if an icon exists by name.

        Args:
            name: Icon name (e.g., "heart", "github", "fa-star")

        Returns:
            True if the icon exists, False otherwise
        """
        ...

    def icon_count(self) -> int:
        """
        Get the number of available icons.

        Returns:
            Number of icons loaded from CSS
        """
        ...

    def list_icons(self) -> list[str]:
        """
        List all available icon names.

        Returns:
            List of icon names (without "fa-" prefix)
        """
        ...

    def render_icon(
        self,
        name: str,
        canvas_width: int = 1024,
        canvas_height: int = 1024,
        icon_size: Optional[int] = None,
        supersample: int = 2,
        icon_color: ColorType = "#000000",
        background_color: Optional[ColorType] = "#FFFFFF",
        horizontal_anchor: HorizontalAnchorType = HorizontalAnchor.Center,
        vertical_anchor: VerticalAnchorType = VerticalAnchor.Center,
        offset_x: int = 0,
        offset_y: int = 0,
        output_format: OutputFormatType = OutputFormat.Png,
    ) -> bytes:
        """
        Render an icon to an image.

        Args:
            name: Icon name (e.g., "heart", "github")
            canvas_width: Output image width in pixels (default: 1024)
            canvas_height: Output image height in pixels (default: 1024)
            icon_size: Icon size in pixels (default: 95% of smaller canvas dimension)
            supersample: Supersampling factor for antialiasing (default: 2)
            icon_color: Icon color as hex string (e.g., "#FF0000") or RGB/RGBA tuple
                (e.g., (255, 0, 0) or (255, 0, 0, 128)). Default: "#000000"
            background_color: Background color as hex string or RGB/RGBA tuple,
                or None for transparent. Default: "#FFFFFF" (white)
            horizontal_anchor: Horizontal alignment - HorizontalAnchor enum or string
                ("left", "center", "right"). Default: "center"
            vertical_anchor: Vertical alignment - VerticalAnchor enum or string
                ("top", "center", "bottom"). Default: "center"
            offset_x: Horizontal pixel offset from anchor (default: 0)
            offset_y: Vertical pixel offset from anchor (default: 0)
            output_format: Output format - OutputFormat enum or string
                ("png", "webp"). Default: "png"

        Returns:
            Encoded image data as bytes

        Raises:
            ValueError: If icon not found or invalid parameters
        """
        ...

    def save_icon(
        self,
        name: str,
        path: str,
        canvas_width: int = 1024,
        canvas_height: int = 1024,
        icon_size: Optional[int] = None,
        supersample: int = 2,
        icon_color: ColorType = "#000000",
        background_color: Optional[ColorType] = "#FFFFFF",
    ) -> None:
        """
        Save an icon directly to a file.

        Args:
            name: Icon name
            path: Output file path (extension determines format: .png or .webp)
            canvas_width: Output width (default: 1024)
            canvas_height: Output height (default: 1024)
            icon_size: Icon size (default: 95% of smaller canvas dimension)
            supersample: Supersampling factor (default: 2)
            icon_color: Icon color as hex string or RGB/RGBA tuple (default: "#000000")
            background_color: Background color as hex or RGB/RGBA tuple,
                or None for transparent. Default: "#FFFFFF" (white)
        """
        ...
