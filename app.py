import base64
import os
import tempfile
from pathlib import Path

import cv2
from flask import Flask, render_template, request

from sight_reader_v2 import SightReaderV2


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "sight-reader-secret")
reader = SightReaderV2()


def image_to_data_uri(image):
    """Encode a BGR OpenCV image to a data URI for inline display."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode annotated image")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET", "POST"])
def index():
    context = {"image_data": None, "notes": None, "error": None}

    if request.method == "POST":
        upload = request.files.get("score")
        if not upload or upload.filename == "":
            context["error"] = "Please choose a sheet-music image before submitting."
            return render_template("index.html", **context)

        suffix = Path(upload.filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            upload.save(tmp.name)
            temp_path = tmp.name

        try:
            annotated, notes = reader.process_sheet_music(temp_path, return_details=True)
            context["image_data"] = image_to_data_uri(annotated)
            context["notes"] = notes
        except Exception as exc:
            context["error"] = f"Unable to analyze that file: {exc}"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
