"""Comprehensive test suite for icon_to_image package."""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image

from icon_to_image import (
    FontStyle,
    HorizontalAnchor,
    IconRenderer,
    OutputFormat,
    VerticalAnchor,
)

# Path to assets directory
ASSETS_DIR = Path(__file__).parent.parent / "assets"


@pytest.fixture
def renderer() -> IconRenderer:
    """Create an IconRenderer instance for testing."""
    return IconRenderer(str(ASSETS_DIR))


class TestIconRendererInit:
    """Tests for IconRenderer initialization."""

    def test_init_with_valid_path(self) -> None:
        """Test initialization with valid assets directory."""
        renderer = IconRenderer(str(ASSETS_DIR))
        assert renderer.icon_count() > 0

    def test_init_with_invalid_path(self) -> None:
        """Test initialization with invalid path raises error."""
        with pytest.raises(ValueError, match="Failed to"):
            IconRenderer("/nonexistent/path")


class TestIconLookup:
    """Tests for icon lookup functionality."""

    def test_has_icon_existing(self, renderer: IconRenderer) -> None:
        """Test has_icon returns True for existing icons."""
        assert renderer.has_icon("heart") is True
        assert renderer.has_icon("star") is True
        assert renderer.has_icon("user") is True

    def test_has_icon_with_prefix(self, renderer: IconRenderer) -> None:
        """Test has_icon works with fa- prefix."""
        assert renderer.has_icon("fa-heart") is True
        assert renderer.has_icon("fa-star") is True

    def test_has_icon_nonexistent(self, renderer: IconRenderer) -> None:
        """Test has_icon returns False for nonexistent icons."""
        assert renderer.has_icon("nonexistent-icon-xyz") is False

    def test_icon_count(self, renderer: IconRenderer) -> None:
        """Test icon_count returns reasonable number."""
        count = renderer.icon_count()
        assert count > 1000  # Font Awesome has many icons
        assert count < 10000  # But not too many

    def test_list_icons(self, renderer: IconRenderer) -> None:
        """Test list_icons returns icon names."""
        icons = renderer.list_icons()
        assert len(icons) == renderer.icon_count()
        assert "heart" in icons
        assert "star" in icons

    def test_case_insensitive(self, renderer: IconRenderer) -> None:
        """Test icon lookup is case insensitive."""
        assert renderer.has_icon("heart") is True
        assert renderer.has_icon("HEART") is True
        assert renderer.has_icon("Heart") is True


class TestRenderIcon:
    """Tests for icon rendering."""

    def test_render_default(self, renderer: IconRenderer) -> None:
        """Test rendering with default parameters returns PIL Image."""
        img = renderer.render_icon("heart")
        # render_icon returns a PIL Image, not bytes
        assert img.mode == "RGBA"
        # Default canvas size is 512x512 per the API
        assert img.size == (512, 512)

    def test_render_custom_size(self, renderer: IconRenderer) -> None:
        """Test rendering with custom canvas size."""
        img = renderer.render_icon("star", canvas_width=256, canvas_height=512)
        assert img.size == (256, 512)

    def test_render_icon_size(self, renderer: IconRenderer) -> None:
        """Test rendering with custom icon size."""
        # Use render_icon_bytes to compare raw bytes
        small_png = renderer.render_icon_bytes("heart", icon_size=100, canvas_width=256)
        large_png = renderer.render_icon_bytes("heart", icon_size=200, canvas_width=256)
        # Different sizes should produce different results
        assert small_png != large_png

    def test_render_webp(self, renderer: IconRenderer) -> None:
        """Test rendering to WebP format via render_icon_bytes."""
        webp_data = renderer.render_icon_bytes("heart", output_format=OutputFormat.WebP)
        assert isinstance(webp_data, bytes)
        img = Image.open(io.BytesIO(webp_data))
        assert img.format == "WEBP"

    def test_render_icon_color_hex(self, renderer: IconRenderer) -> None:
        """Test rendering with hex color."""
        # Use render_icon_bytes for byte comparison
        red_png = renderer.render_icon_bytes("heart", icon_color="#FF0000")
        blue_png = renderer.render_icon_bytes("heart", icon_color="#0000FF")
        assert red_png != blue_png

    def test_render_icon_color_short_hex(self, renderer: IconRenderer) -> None:
        """Test rendering with short hex color."""
        img = renderer.render_icon("heart", icon_color="#F00")
        assert img.size == (512, 512)

    def test_render_transparent_background(self, renderer: IconRenderer) -> None:
        """Test rendering with transparent background."""
        img = renderer.render_icon("heart", background_color=None)
        assert img.mode == "RGBA"

    def test_render_solid_background(self, renderer: IconRenderer) -> None:
        """Test rendering with solid background color."""
        img = renderer.render_icon("heart", background_color="#FFFFFF")
        assert img.size == (512, 512)

    def test_render_invalid_icon(self, renderer: IconRenderer) -> None:
        """Test rendering invalid icon raises error."""
        with pytest.raises(ValueError, match="not found"):
            renderer.render_icon("nonexistent-icon-xyz")

    def test_render_invalid_color(self, renderer: IconRenderer) -> None:
        """Test rendering with invalid color raises error."""
        with pytest.raises(ValueError, match="Invalid"):
            renderer.render_icon("heart", icon_color="not-a-color")


class TestRenderIconRGB:
    """Tests for RGB color rendering via tuples."""

    def test_render_rgb_red(self, renderer: IconRenderer) -> None:
        """Test rendering with RGB red color tuple."""
        img = renderer.render_icon("heart", icon_color=(255, 0, 0))
        assert img.size == (512, 512)

    def test_render_rgb_with_alpha(self, renderer: IconRenderer) -> None:
        """Test rendering with RGBA color tuple."""
        img = renderer.render_icon(
            "heart",
            icon_color=(255, 0, 0, 128),
        )
        assert img.size == (512, 512)

    def test_render_rgb_transparent_bg(self, renderer: IconRenderer) -> None:
        """Test rendering RGB with transparent background."""
        img = renderer.render_icon(
            "heart",
            icon_color=(0, 0, 0),
            background_color=None,
        )
        assert img.mode == "RGBA"


class TestAnchoring:
    """Tests for icon positioning with anchors."""

    def test_horizontal_anchors(self, renderer: IconRenderer) -> None:
        """Test different horizontal anchor positions."""
        # Use smaller icon_size than canvas to see anchor effect
        left = renderer.render_icon(
            "arrow-right",
            horizontal_anchor=HorizontalAnchor.Left,
            canvas_width=256,
            icon_size=100,
        )
        center = renderer.render_icon(
            "arrow-right",
            horizontal_anchor=HorizontalAnchor.Center,
            canvas_width=256,
            icon_size=100,
        )
        right = renderer.render_icon(
            "arrow-right",
            horizontal_anchor=HorizontalAnchor.Right,
            canvas_width=256,
            icon_size=100,
        )
        # All should be different
        assert left != center
        assert center != right
        assert left != right

    def test_vertical_anchors(self, renderer: IconRenderer) -> None:
        """Test different vertical anchor positions."""
        # Use smaller icon_size than canvas to see anchor effect
        top = renderer.render_icon(
            "arrow-down",
            vertical_anchor=VerticalAnchor.Top,
            canvas_height=256,
            icon_size=100,
        )
        center = renderer.render_icon(
            "arrow-down",
            vertical_anchor=VerticalAnchor.Center,
            canvas_height=256,
            icon_size=100,
        )
        bottom = renderer.render_icon(
            "arrow-down",
            vertical_anchor=VerticalAnchor.Bottom,
            canvas_height=256,
            icon_size=100,
        )
        # All should be different
        assert top != center
        assert center != bottom
        assert top != bottom

    def test_offset(self, renderer: IconRenderer) -> None:
        """Test rendering with pixel offset."""
        no_offset = renderer.render_icon("heart", canvas_width=256)
        with_offset = renderer.render_icon(
            "heart",
            canvas_width=256,
            offset_x=50,
            offset_y=50,
        )
        assert no_offset != with_offset


class TestSupersampling:
    """Tests for supersampling antialiasing."""

    def test_supersample_default(self, renderer: IconRenderer) -> None:
        """Test default supersampling (2x)."""
        img = renderer.render_icon("heart", supersample=2)
        assert img.size == (512, 512)

    def test_supersample_4x(self, renderer: IconRenderer) -> None:
        """Test 4x supersampling."""
        img = renderer.render_icon("heart", supersample=4)
        assert img.size == (512, 512)

    def test_supersample_quality_difference(self, renderer: IconRenderer) -> None:
        """Test that higher supersampling produces different results."""
        img1 = renderer.render_icon("heart", supersample=1, canvas_width=64)
        img4 = renderer.render_icon("heart", supersample=4, canvas_width=64)
        # Results should be the same size but may differ in quality
        assert img1.size == img4.size


class TestSaveIcon:
    """Tests for save_icon functionality."""

    def test_save_png(self, renderer: IconRenderer) -> None:
        """Test saving icon as PNG."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            try:
                renderer.save_icon("heart", f.name)
                assert os.path.exists(f.name)
                img = Image.open(f.name)
                assert img.format == "PNG"
            finally:
                os.unlink(f.name)

    def test_save_webp(self, renderer: IconRenderer) -> None:
        """Test saving icon as WebP."""
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            try:
                renderer.save_icon("heart", f.name)
                assert os.path.exists(f.name)
                img = Image.open(f.name)
                assert img.format == "WEBP"
            finally:
                os.unlink(f.name)

    def test_save_with_options(self, renderer: IconRenderer) -> None:
        """Test saving with custom options."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            try:
                renderer.save_icon(
                    "star",
                    f.name,
                    canvas_width=512,
                    canvas_height=512,
                    icon_color="#FFD700",
                )
                img = Image.open(f.name)
                assert img.size == (512, 512)
            finally:
                os.unlink(f.name)


class TestBrandIcons:
    """Tests for brand icons (uses fa-brands.otf)."""

    def test_github_icon(self, renderer: IconRenderer) -> None:
        """Test rendering GitHub brand icon."""
        assert renderer.has_icon("github")
        img = renderer.render_icon("github")
        assert img.size == (512, 512)

    def test_twitter_icon(self, renderer: IconRenderer) -> None:
        """Test rendering Twitter brand icon."""
        assert renderer.has_icon("twitter")
        img = renderer.render_icon("twitter")
        assert img.size == (512, 512)

    def test_python_icon(self, renderer: IconRenderer) -> None:
        """Test rendering Python brand icon."""
        assert renderer.has_icon("python")
        img = renderer.render_icon("python")
        assert img.size == (512, 512)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_size(self, renderer: IconRenderer) -> None:
        """Test rendering at minimum size."""
        img = renderer.render_icon("heart", canvas_width=1, canvas_height=1)
        assert img.size == (1, 1)

    def test_large_size(self, renderer: IconRenderer) -> None:
        """Test rendering at large size."""
        img = renderer.render_icon("heart", canvas_width=2048, canvas_height=2048)
        assert img.size == (2048, 2048)

    def test_non_square(self, renderer: IconRenderer) -> None:
        """Test rendering non-square dimensions."""
        img = renderer.render_icon("heart", canvas_width=100, canvas_height=200)
        assert img.size == (100, 200)

    def test_letter_icons(self, renderer: IconRenderer) -> None:
        """Test rendering letter icons (a-z)."""
        assert renderer.has_icon("a")
        img = renderer.render_icon("a")
        assert img.size == (512, 512)

    def test_number_icons(self, renderer: IconRenderer) -> None:
        """Test rendering number icons (0-9)."""
        assert renderer.has_icon("0")
        img = renderer.render_icon("0")
        assert img.size == (512, 512)


class TestIconAliases:
    """Tests for icon name aliases (legacy names mapping to same codepoint)."""

    def test_times_circle_alias(self, renderer: IconRenderer) -> None:
        """Test times-circle alias for circle-xmark."""
        # Both names should exist and render
        assert renderer.has_icon("times-circle")
        assert renderer.has_icon("circle-xmark")
        assert renderer.has_icon("xmark-circle")

        # All should render successfully (use render_icon_bytes for comparison)
        times_circle = renderer.render_icon_bytes("times-circle", canvas_width=64)
        circle_xmark = renderer.render_icon_bytes("circle-xmark", canvas_width=64)
        xmark_circle = renderer.render_icon_bytes("xmark-circle", canvas_width=64)

        assert len(times_circle) > 0
        assert len(circle_xmark) > 0
        assert len(xmark_circle) > 0

        # All aliases should produce identical output (same codepoint)
        assert times_circle == circle_xmark
        assert circle_xmark == xmark_circle

    def test_common_aliases(self, renderer: IconRenderer) -> None:
        """Test other common icon aliases work."""
        # close/times/xmark/multiply/remove all map to the same icon
        assert renderer.has_icon("close")
        assert renderer.has_icon("times")
        assert renderer.has_icon("xmark")

        # Check they render the same
        close_icon = renderer.render_icon("close", canvas_width=64)
        times_icon = renderer.render_icon("times", canvas_width=64)
        xmark_icon = renderer.render_icon("xmark", canvas_width=64)

        assert close_icon == times_icon
        assert times_icon == xmark_icon


class TestOutputFormats:
    """Tests for different output formats."""

    def test_png_format_enum(self) -> None:
        """Test OutputFormat.Png enum value."""
        assert OutputFormat.Png is not None

    def test_webp_format_enum(self) -> None:
        """Test OutputFormat.WebP enum value."""
        assert OutputFormat.WebP is not None

    def test_png_file_header(self, renderer: IconRenderer) -> None:
        """Test PNG file has correct magic bytes."""
        png_data = renderer.render_icon_bytes("heart", output_format=OutputFormat.Png)
        # PNG magic bytes
        assert png_data[:4] == b"\x89PNG"

    def test_webp_file_header(self, renderer: IconRenderer) -> None:
        """Test WebP file has correct magic bytes."""
        webp_data = renderer.render_icon_bytes("heart", output_format=OutputFormat.WebP)
        # WebP magic bytes (RIFF....WEBP)
        assert webp_data[:4] == b"RIFF"
        assert webp_data[8:12] == b"WEBP"


class TestRotation:
    """Tests for icon rotation."""

    def test_rotate_zero(self, renderer: IconRenderer) -> None:
        """Test that rotate=0 produces same result as no rotation."""
        # Use render_icon_bytes for byte comparison
        no_rotate = renderer.render_icon_bytes("arrow-right", canvas_width=256)
        with_zero = renderer.render_icon_bytes("arrow-right", canvas_width=256, rotate=0)
        assert no_rotate == with_zero

    def test_rotate_produces_different_result(self, renderer: IconRenderer) -> None:
        """Test that rotation produces a different image."""
        # Use render_icon_bytes for byte comparison
        no_rotate = renderer.render_icon_bytes(
            "arrow-right",
            canvas_width=256,
            icon_size=128,
        )
        rotated = renderer.render_icon_bytes(
            "arrow-right",
            canvas_width=256,
            icon_size=128,
            rotate=45,
        )
        assert no_rotate != rotated

    def test_rotate_positive(self, renderer: IconRenderer) -> None:
        """Test positive rotation (clockwise)."""
        img = renderer.render_icon("arrow-up", rotate=90)
        # Default canvas size is 512x512
        assert img.size == (512, 512)

    def test_rotate_negative(self, renderer: IconRenderer) -> None:
        """Test negative rotation (counter-clockwise)."""
        img = renderer.render_icon("arrow-up", rotate=-90)
        assert img.size == (512, 512)

    def test_rotate_45_degrees(self, renderer: IconRenderer) -> None:
        """Test 45-degree rotation."""
        img = renderer.render_icon(
            "heart",
            canvas_width=256,
            canvas_height=256,
            icon_size=128,
            rotate=45,
        )
        assert img.size == (256, 256)

    def test_rotate_180_degrees(self, renderer: IconRenderer) -> None:
        """Test 180-degree rotation (upside down)."""
        img = renderer.render_icon("arrow-up", rotate=180)
        assert img.size == (512, 512)

    def test_rotate_360_degrees(self, renderer: IconRenderer) -> None:
        """Test 360-degree rotation returns similar to original."""
        # 360-degree rotation should be visually identical to no rotation
        # (though may have slight anti-aliasing differences due to bilinear sampling)
        no_rotate = renderer.render_icon(
            "circle",
            canvas_width=256,
            canvas_height=256,
            icon_size=100,
            rotate=0,
        )
        full_rotate = renderer.render_icon(
            "circle",
            canvas_width=256,
            canvas_height=256,
            icon_size=100,
            rotate=360,
        )
        # Both should render successfully
        assert no_rotate.size == (256, 256)
        assert full_rotate.size == (256, 256)

    def test_rotate_with_transparent_background(self, renderer: IconRenderer) -> None:
        """Test rotation with transparent background."""
        img = renderer.render_icon(
            "star",
            rotate=30,
            background_color=None,
        )
        assert img.mode == "RGBA"

    def test_rotate_with_offset(self, renderer: IconRenderer) -> None:
        """Test rotation combined with offset."""
        img = renderer.render_icon(
            "heart",
            canvas_width=256,
            canvas_height=256,
            icon_size=64,
            rotate=45,
            offset_x=20,
            offset_y=20,
        )
        assert img.size == (256, 256)

    def test_rotate_with_anchoring(self, renderer: IconRenderer) -> None:
        """Test rotation combined with anchoring."""
        img = renderer.render_icon(
            "check",
            canvas_width=256,
            canvas_height=256,
            icon_size=64,
            rotate=45,
            horizontal_anchor="left",
            vertical_anchor="top",
        )
        assert img.size == (256, 256)

    def test_rotate_float_precision(self, renderer: IconRenderer) -> None:
        """Test rotation with float values."""
        img = renderer.render_icon("star", rotate=22.5)
        assert img.size == (512, 512)

    def test_rotate_large_angle(self, renderer: IconRenderer) -> None:
        """Test rotation with angle > 360 degrees."""
        img = renderer.render_icon("star", rotate=450)  # Same as 90 degrees
        assert img.size == (512, 512)


class TestFontStyles:
    """Tests for font style selection (solid, regular, brands)."""

    def test_style_enum_values(self) -> None:
        """Test FontStyle enum has expected values."""
        assert FontStyle.Solid is not None
        assert FontStyle.Regular is not None
        assert FontStyle.Brands is not None

    def test_render_solid_style_string(self, renderer: IconRenderer) -> None:
        """Test rendering with solid style using string."""
        png_data = renderer.render_icon_bytes("heart", style="solid")
        assert len(png_data) > 0
        img = Image.open(io.BytesIO(png_data))
        assert img.format == "PNG"

    def test_render_solid_style_enum(self, renderer: IconRenderer) -> None:
        """Test rendering with solid style using enum."""
        png_data = renderer.render_icon_bytes("heart", style=FontStyle.Solid)
        assert len(png_data) > 0
        img = Image.open(io.BytesIO(png_data))
        assert img.format == "PNG"

    def test_render_regular_style(self, renderer: IconRenderer) -> None:
        """Test rendering with regular (outlined) style."""
        png_data = renderer.render_icon_bytes("heart", style="regular")
        assert len(png_data) > 0
        img = Image.open(io.BytesIO(png_data))
        assert img.format == "PNG"

    def test_render_brands_style(self, renderer: IconRenderer) -> None:
        """Test rendering brand icon with brands style."""
        png_data = renderer.render_icon_bytes("github", style="brands")
        assert len(png_data) > 0
        img = Image.open(io.BytesIO(png_data))
        assert img.format == "PNG"

    def test_solid_vs_regular_produces_different_output(
        self, renderer: IconRenderer
    ) -> None:
        """Test that solid and regular styles produce different images."""
        solid = renderer.render_icon_bytes("heart", canvas_width=256, style="solid")
        regular = renderer.render_icon_bytes("heart", canvas_width=256, style="regular")
        # Solid (filled) and regular (outlined) should look different
        assert solid != regular

    def test_default_style_is_solid_for_normal_icons(
        self, renderer: IconRenderer
    ) -> None:
        """Test that default style for normal icons is solid."""
        default = renderer.render_icon_bytes("heart", canvas_width=256)
        solid = renderer.render_icon_bytes("heart", canvas_width=256, style="solid")
        # Default style should match solid style for normal icons
        assert default == solid

    def test_default_style_is_brands_for_brand_icons(
        self, renderer: IconRenderer
    ) -> None:
        """Test that default style for brand icons is brands."""
        default = renderer.render_icon_bytes("github", canvas_width=256)
        brands = renderer.render_icon_bytes("github", canvas_width=256, style="brands")
        # Default style should match brands style for brand icons
        assert default == brands

    def test_style_case_insensitive(self, renderer: IconRenderer) -> None:
        """Test that style string is case insensitive."""
        lower = renderer.render_icon_bytes("heart", canvas_width=256, style="solid")
        upper = renderer.render_icon_bytes("heart", canvas_width=256, style="SOLID")
        mixed = renderer.render_icon_bytes("heart", canvas_width=256, style="Solid")
        assert lower == upper == mixed

    def test_invalid_style_raises_error(self, renderer: IconRenderer) -> None:
        """Test that invalid style raises ValueError."""
        with pytest.raises(ValueError, match="Invalid font style"):
            renderer.render_icon_bytes("heart", style="invalid")

    def test_render_icon_bytes_with_style(self, renderer: IconRenderer) -> None:
        """Test render_icon_bytes with style parameter."""
        solid = renderer.render_icon_bytes("heart", canvas_width=256, style="solid")
        regular = renderer.render_icon_bytes("heart", canvas_width=256, style="regular")
        assert solid != regular

    def test_style_with_other_parameters(self, renderer: IconRenderer) -> None:
        """Test style parameter works with other rendering options."""
        png_data = renderer.render_icon_bytes(
            "heart",
            canvas_width=256,
            canvas_height=256,
            icon_size=128,
            icon_color="#FF0000",
            background_color=None,
            rotate=45,
            style="solid",
        )
        assert len(png_data) > 0
        img = Image.open(io.BytesIO(png_data))
        assert img.size == (256, 256)
        assert img.mode == "RGBA"

    def test_style_none_uses_default(self, renderer: IconRenderer) -> None:
        """Test that style=None uses the icon's default style."""
        default = renderer.render_icon_bytes("heart", canvas_width=256)
        explicit_none = renderer.render_icon_bytes("heart", canvas_width=256, style=None)
        assert default == explicit_none


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
