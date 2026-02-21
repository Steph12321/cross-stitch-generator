# Cross-Stitch Pattern Generator — Project Plan

## Overview
A Flask web app that converts uploaded photos into printable cross-stitch patterns with DMC thread colour codes.

## File Structure
```
cross-stitch-generator/
├── app.py               # Flask routes (thin layer)
├── pattern.py           # All image processing logic
├── dmc_colors.py        # DMC thread colour database (~88 colours)
├── requirements.txt
├── templates/
│   └── index.html       # Single Jinja2 page: form + result
├── static/
│   └── style.css
└── .gitignore
```

No file writes to disk — uploaded images and generated patterns are handled in-memory (BytesIO).

## Tech Stack
- **Flask** — web framework
- **Pillow (PIL)** — image manipulation and pattern rendering
- **scikit-learn** — KMeans colour quantization
- **numpy** — array operations and vectorised colour matching

## Core Algorithm
1. User uploads photo + selects number of colours (8–24) and grid width (30–100 stitches)
2. Resize image to stitch grid dimensions (preserving aspect ratio)
3. KMeans colour quantization → `n_colors` cluster centroids
4. Map each centroid to nearest DMC thread colour using CIE LAB distance (perceptually accurate)
5. Assign a unique symbol (A, B, C...) to each DMC colour used
6. Draw output: 10×10 px cells per stitch, filled with DMC colour, symbol drawn in contrasting text, grid lines on top
7. Encode result as base64 PNG for inline display + download

## Key Functions in `pattern.py`
| Function | Purpose |
|---|---|
| `rgb_to_lab(rgb)` | RGB → CIE LAB colour space |
| `build_dmc_lab_cache()` | Pre-compute LAB for all DMC colours at startup |
| `find_nearest_dmc(rgb)` | Vectorised nearest-DMC lookup via LAB distance |
| `quantize_image(img, n_colors)` | KMeans quantization → label_grid + centroids |
| `map_colors_to_dmc(center_colors)` | Map each centroid to nearest DMC dict |
| `assign_symbols(dmc_list)` | Assign A–Z symbols per unique DMC number |
| `get_contrast_color(rgb)` | WCAG luminance → black or white text colour |
| `generate_pattern_image(...)` | Draw all stitch cells + grid lines → PIL Image |
| `generate_legend(...)` | Build legend list with hex codes for template |
| `process_image(...)` | Top-level orchestrator: open → resize → quantize → draw → base64 |

## UI
- Single page: upload form always visible, result section appears below after processing
- Pattern displayed inline as base64 image
- Download button saves pattern as PNG
- Colour legend table: Symbol | Swatch | DMC Number | Colour Name
- `image-rendering: pixelated` CSS prevents browser from blurring the pixel-art pattern
- Print styles hide form and download button

## DMC Colour Database (`dmc_colors.py`)
~88 DMC stranded cotton colours covering:
- Blacks, whites, grays
- Reds, pinks, roses
- Purples, violets
- Blues, teals, aquas
- Greens
- Yellows, golds, oranges
- Browns, tans

## Running Locally
```bash
pip install Flask Pillow scikit-learn numpy
python app.py
# Visit http://127.0.0.1:5000
```
