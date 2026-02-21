"""Flask routes for the cross-stitch pattern generator."""

from flask import Flask, render_template, request, Response
from pattern import process_image, generate_pdf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}

# In-memory cache of the most recently generated pattern (for PDF download).
_pdf_cache = None


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    global _pdf_cache
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("photo")
        if not file or file.filename == "":
            error = "Please select an image file."
        elif not _allowed(file.filename):
            error = "Unsupported file type. Please upload a JPEG, PNG, GIF, BMP, or WebP image."
        else:
            try:
                n_colors = int(request.form.get("n_colors", 16))
                n_colors = max(8, min(24, n_colors))

                grid_width = int(request.form.get("grid_width", 60))
                grid_width = max(30, min(100, grid_width))

                full_result = process_image(file.stream, n_colors, grid_width)

                # Cache the full result (including PIL Image) for PDF download.
                _pdf_cache = full_result

                # Strip the PIL Image before passing to the template.
                result = {k: v for k, v in full_result.items() if k != "pattern_img"}
            except Exception as exc:
                error = f"Could not process image: {exc}"

    return render_template("index.html", result=result, error=error)


@app.route("/download-pdf")
def download_pdf():
    if _pdf_cache is None:
        return "No pattern has been generated yet. Please generate a pattern first.", 404
    pdf_bytes = generate_pdf(
        _pdf_cache["pattern_img"],
        _pdf_cache["legend"],
        _pdf_cache["grid_width"],
        _pdf_cache["grid_height"],
        _pdf_cache["n_colors_used"],
    )
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cross_stitch_pattern.pdf"},
    )


if __name__ == "__main__":
    app.run(debug=True)
