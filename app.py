import io
import os
import uuid
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt

from face_recognition import add_labels_to_image

# Initialize Flask application
app = Flask(__name__, static_folder='static')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.route("/", methods=["GET"])
def index():
    """Render the main page with upload form."""
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def process_image():
    """Handle image upload, perform face recognition, and return result."""
    file = request.files.get("image")
    if not file:
        return render_template("index.html", error="No file uploaded.")

    img = Image.open(file.stream)
    result_fig = add_labels_to_image(img)

    if result_fig is None:
        return render_template("index.html", error="No face detected.")

    # Ensure 'static' directory exists
    os.makedirs('static', exist_ok=True)

    # Save the result image with a unique filename
    result_filename = f"result_{uuid.uuid4()}.jpg"
    output_path = os.path.join("static", result_filename)
    result_fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(result_fig)

    return render_template("index.html", result_image="/" + output_path)

if __name__ == "__main__":
    # For development only; use a production WSGI server for deployment
    app.run(debug=True)
