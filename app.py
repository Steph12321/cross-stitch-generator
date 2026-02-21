"""Flask routes for the cross-stitch pattern generator."""

from flask import Flask, render_template, request
from pattern import process_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
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

                result = process_image(file.stream, n_colors, grid_width)
            except Exception as exc:
                error = f"Could not process image: {exc}"

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
