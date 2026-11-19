"""Generate the repository's generic geometric application icon.

This artwork contains no uploaded image. Pillow is already a build dependency.
The higher-resolution drawing is downsampled for clean small Windows icons.
"""

from pathlib import Path

from PIL import Image, ImageDraw


def build_fallback_icon(destination: Path | None = None) -> None:
    destination = destination or Path(__file__).resolve().parents[1] / "assets"
    destination.mkdir(parents=True, exist_ok=True)
    factor = 4
    image = Image.new("RGBA", (256 * factor, 256 * factor), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, 255 * factor, 255 * factor),
                           radius=46 * factor, fill="#14283F")
    vertices = [(128, 37), (205, 82), (205, 174), (128, 219), (51, 174), (51, 82)]
    points = [(x * factor, y * factor) for x, y in vertices]
    draw.polygon(points, fill="#204F65")
    draw.line(points + [points[0]], fill="#65DFCA", width=8 * factor, joint="curve")
    for x, y in ((128, 91), (96, 110), (160, 110), (128, 128),
                 (96, 147), (160, 147), (128, 165)):
        radius = 8 * factor
        draw.ellipse((x * factor-radius, y * factor-radius,
                      x * factor+radius, y * factor+radius), fill="#F4F8FB")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    image.save(destination / "app-icon.png")
    image.save(destination / "app-icon.ico", sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])


if __name__ == "__main__":
    build_fallback_icon()
