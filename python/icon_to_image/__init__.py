"""
Icon to Image - Fast Font Awesome icon rendering.

This package provides high-performance icon rendering using Rust's ab_glyph
library with Python bindings via PyO3.

Example:
    >>> from icon_to_image import IconRenderer
    >>> renderer = IconRenderer()  # Uses embedded assets
    >>>
    >>> # render_icon() returns a PIL.Image (requires Pillow)
    >>> img = renderer.render_icon("heart", icon_color="#FF0000")
    >>> img.save("heart.png")
    >>>
    >>> # render_icon_bytes() returns raw bytes (no Pillow needed)
    >>> png_data = renderer.render_icon_bytes("heart", icon_color="#FF0000")
    >>> with open("heart.png", "wb") as f:
    ...     f.write(png_data)
"""

from ._native import (
    FontStyle,
    HorizontalAnchor,
    IconRenderer,
    OutputFormat,
    VerticalAnchor,
)

__all__ = [
    "FontStyle",
    "HorizontalAnchor",
    "IconRenderer",
    "OutputFormat",
    "VerticalAnchor",
    "main",
]

__version__ = "0.1.0"


def main() -> None:
    """Command-line interface entry point for icon-to-image."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="icon-to-image",
        description="Render Font Awesome icons to image files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render a heart icon to PNG
  icon-to-image heart output.png

  # Render with custom color and size
  icon-to-image star output.png --color "#FFD700" --size 512

  # Render with transparent background
  icon-to-image github logo.png --color "#333333" --background transparent

  # List available icons
  icon-to-image --list

  # Search for icons
  icon-to-image --search arrow
""",
    )

    # Positional arguments
    parser.add_argument(
        "icon",
        nargs="?",
        help="Icon name (e.g., 'heart', 'github', 'star')",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file path (e.g., 'icon.png' or 'icon.webp')",
    )

    # Optional arguments
    parser.add_argument(
        "-c", "--color",
        default="#000000",
        help="Icon color as hex (default: #000000)",
    )
    parser.add_argument(
        "-b", "--background",
        default="#FFFFFF",
        help="Background color as hex, or 'transparent' (default: #FFFFFF)",
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=1024,
        help="Canvas size in pixels (default: 1024)",
    )
    parser.add_argument(
        "--icon-size",
        type=int,
        default=None,
        help="Icon size in pixels (default: 95%% of canvas size)",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=2,
        choices=[1, 2, 4],
        help="Supersampling factor for antialiasing (default: 2)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available icon names",
    )
    parser.add_argument(
        "--search",
        metavar="PATTERN",
        help="Search for icons matching a pattern",
    )
    parser.add_argument(
        "--assets",
        metavar="DIR",
        help="Path to custom Font Awesome assets directory",
    )

    args = parser.parse_args()

    # Initialize renderer
    try:
        renderer = IconRenderer(args.assets)
    except ValueError as e:
        print(f"Error initializing renderer: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle --list
    if args.list:
        icons = renderer.list_icons()
        print(f"Available icons ({len(icons)}):\n")
        # Print in columns
        icons_sorted = sorted(icons)
        col_width = max(len(name) for name in icons_sorted) + 2
        cols = 80 // col_width
        for i in range(0, len(icons_sorted), cols):
            row = icons_sorted[i : i + cols]
            print("".join(name.ljust(col_width) for name in row))
        sys.exit(0)

    # Handle --search
    if args.search:
        pattern = args.search.lower()
        icons = renderer.list_icons()
        matches = [name for name in icons if pattern in name.lower()]
        if matches:
            print(f"Icons matching '{args.search}' ({len(matches)}):\n")
            for name in sorted(matches):
                print(f"  {name}")
        else:
            print(f"No icons found matching '{args.search}'")
        sys.exit(0)

    # Validate required arguments for rendering
    if not args.icon or not args.output:
        parser.error("icon and output arguments are required for rendering")

    # Check if icon exists
    if not renderer.has_icon(args.icon):
        print(f"Error: Icon '{args.icon}' not found", file=sys.stderr)
        # Suggest similar icons
        icons = renderer.list_icons()
        suggestions = [
            name for name in icons if args.icon.lower() in name.lower()
        ][:5]
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}?", file=sys.stderr)
        sys.exit(1)

    # Parse background color
    bg_color = None if args.background.lower() == "transparent" else args.background

    # Render and save
    try:
        renderer.save_icon(
            args.icon,
            args.output,
            canvas_width=args.size,
            canvas_height=args.size,
            icon_size=args.icon_size,
            supersample=args.supersample,
            icon_color=args.color,
            background_color=bg_color,
        )
        print(f"Saved {args.icon} to {args.output}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
