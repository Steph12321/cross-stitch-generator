"""Image processing logic for cross-stitch pattern generation."""

import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from dmc_colors import DMC_COLORS

STITCH_SIZE = 10  # pixels per stitch cell in output image
SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Pre-compute LAB cache at import time
_DMC_LAB_CACHE = None
_DMC_INDICES = None


def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb):
    """Convert an (R, G, B) tuple or Nx3 array to CIE LAB."""
    arr = np.array(rgb, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]

    linear = _srgb_to_linear(arr)

    # D65 illuminant matrix (sRGB → XYZ)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ M.T

    # Normalise by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883])

    # f(t)
    delta = 6 / 29
    xyz = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3 * delta ** 2) + 4 / 29,
    )

    L = 116 * xyz[:, 1] - 16
    a = 500 * (xyz[:, 0] - xyz[:, 1])
    b = 200 * (xyz[:, 1] - xyz[:, 2])
    lab = np.stack([L, a, b], axis=1)

    return lab[0] if single else lab


def build_dmc_lab_cache():
    """Pre-compute CIE LAB values for every DMC colour."""
    global _DMC_LAB_CACHE
    rgbs = np.array([c["rgb"] for c in DMC_COLORS], dtype=float)
    _DMC_LAB_CACHE = rgb_to_lab(rgbs)


def find_nearest_dmc(rgb):
    """Return the DMC colour dict closest to *rgb* in CIE LAB space."""
    if _DMC_LAB_CACHE is None:
        build_dmc_lab_cache()
    lab = rgb_to_lab(np.array(rgb, dtype=float))
    diffs = _DMC_LAB_CACHE - lab
    distances = np.einsum("ij,ij->i", diffs, diffs)
    return DMC_COLORS[int(np.argmin(distances))]


def quantize_image(img, n_colors):
    """
    KMeans colour quantization.

    Returns
    -------
    label_grid : np.ndarray, shape (H, W), dtype int
    center_colors : np.ndarray, shape (n_colors, 3), dtype float
    """
    arr = np.array(img, dtype=float)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)

    km = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    km.fit(pixels)

    label_grid = km.labels_.reshape(h, w)
    center_colors = km.cluster_centers_
    return label_grid, center_colors


def map_colors_to_dmc(center_colors):
    """Map each cluster centroid to its nearest DMC colour dict."""
    return [find_nearest_dmc(c) for c in center_colors]


def assign_symbols(dmc_list):
    """
    Assign a unique letter/digit symbol to each distinct DMC number.

    Returns a dict mapping dmc_number → symbol string.
    """
    seen = {}
    symbol_idx = 0
    for entry in dmc_list:
        key = entry["dmc"]
        if key not in seen:
            seen[key] = SYMBOLS[symbol_idx % len(SYMBOLS)]
            symbol_idx += 1
    return seen


def get_contrast_color(rgb):
    """Return (0,0,0) or (255,255,255) for readable text on *rgb* background."""
    r, g, b = [x / 255.0 for x in rgb]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if luminance > 0.179 else (255, 255, 255)


def _load_font(size=7):
    """Try to load a small TrueType font; fall back to PIL default."""
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def generate_pattern_image(label_grid, dmc_list, dmc_symbols):
    """
    Draw the cross-stitch pattern as a PIL Image.

    Each stitch cell is STITCH_SIZE×STITCH_SIZE pixels, filled with the
    matched DMC colour and labelled with the assigned symbol.
    Grid lines (light gray) are drawn on top.
    """
    h, w = label_grid.shape
    img_w = w * STITCH_SIZE
    img_h = h * STITCH_SIZE

    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(7)

    for row in range(h):
        for col in range(w):
            cluster_idx = int(label_grid[row, col])
            dmc = dmc_list[cluster_idx]
            fill = dmc["rgb"]
            x0 = col * STITCH_SIZE
            y0 = row * STITCH_SIZE
            x1 = x0 + STITCH_SIZE - 1
            y1 = y0 + STITCH_SIZE - 1

            draw.rectangle([x0, y0, x1, y1], fill=fill)

            symbol = dmc_symbols[dmc["dmc"]]
            text_color = get_contrast_color(fill)

            # Centre the symbol in the cell
            try:
                bbox = font.getbbox(symbol)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except AttributeError:
                tw, th = font.getsize(symbol)
            tx = x0 + (STITCH_SIZE - tw) // 2
            ty = y0 + (STITCH_SIZE - th) // 2
            draw.text((tx, ty), symbol, fill=text_color, font=font)

    # Grid lines on top
    grid_color = (180, 180, 180)
    for col in range(w + 1):
        x = col * STITCH_SIZE
        draw.line([(x, 0), (x, img_h)], fill=grid_color, width=1)
    for row in range(h + 1):
        y = row * STITCH_SIZE
        draw.line([(0, y), (img_w, y)], fill=grid_color, width=1)

    return canvas


def generate_legend(dmc_list, dmc_symbols):
    """
    Build a list of dicts for the legend table.

    Each entry: {symbol, dmc, name, hex}
    Deduplicated and ordered by symbol.
    """
    seen = {}
    for entry in dmc_list:
        key = entry["dmc"]
        if key not in seen:
            r, g, b = entry["rgb"]
            seen[key] = {
                "symbol": dmc_symbols[key],
                "dmc": key,
                "name": entry["name"],
                "hex": f"#{r:02X}{g:02X}{b:02X}",
            }
    return sorted(seen.values(), key=lambda x: x["symbol"])


def process_image(file_obj, n_colors, grid_width):
    """
    Full pipeline: open image → quantize → match DMC → draw → base64.

    Parameters
    ----------
    file_obj   : file-like object (BytesIO or Werkzeug FileStorage stream)
    n_colors   : int, number of colours (8–24)
    grid_width : int, stitch columns (30–100)

    Returns
    -------
    dict with keys: image_b64, legend, grid_width, grid_height, n_colors_used
    """
    img = Image.open(file_obj).convert("RGB")

    # Resize preserving aspect ratio
    orig_w, orig_h = img.size
    grid_height = max(1, round(grid_width * orig_h / orig_w))
    img = img.resize((grid_width, grid_height), Image.LANCZOS)

    label_grid, center_colors = quantize_image(img, n_colors)
    dmc_list = map_colors_to_dmc(center_colors)
    dmc_symbols = assign_symbols(dmc_list)

    pattern_img = generate_pattern_image(label_grid, dmc_list, dmc_symbols)
    legend = generate_legend(dmc_list, dmc_symbols)

    buf = io.BytesIO()
    pattern_img.save(buf, format="PNG")
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("ascii")

    return {
        "image_b64": image_b64,
        "legend": legend,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "n_colors_used": len(legend),
    }


# Build the LAB cache once at import time
build_dmc_lab_cache()
