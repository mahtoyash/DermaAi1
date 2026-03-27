import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

from model.model_loader import load_model
from utils.predictor    import predict
from utils.gradcam      import generate_gradcam
from utils.report_gen   import generate_report

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB limit

ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp"}

# ── Model — startup pe ek baar load ───────────────────────────────────────────
print("⏳ Model load ho raha hai...")
model, CLASSES, VAL_ACC = load_model(MODEL_PATH)
print(f"✅ Ready | Classes: {CLASSES} | Val Acc: {VAL_ACC:.4f}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        classes=CLASSES,
        val_acc=round(VAL_ACC * 100, 2)
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Format supported nahi. PNG/JPG/JPEG/BMP/WEBP use karo"}), 400

    # ── Save uploaded image ────────────────────────────────────────────────────
    uid      = uuid.uuid4().hex[:8]
    fname    = uid + "_" + secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_DIR, fname)
    file.save(img_path)

    # ── 1. Predict ─────────────────────────────────────────────────────────────
    predictions = predict(model, CLASSES, img_path)
    top_class   = predictions[0]["class"]
    top_idx     = CLASSES.index(top_class)

    # ── 2. GradCAM ─────────────────────────────────────────────────────────────
    cam_fname = uid + "_gradcam.png"
    cam_path  = os.path.join(UPLOAD_DIR, cam_fname)
    generate_gradcam(model, img_path, top_idx, cam_path)

    # ── 3. Report ──────────────────────────────────────────────────────────────
    rep_fname = uid + "_report.json"
    rep_path  = os.path.join(REPORT_DIR, rep_fname)
    generate_report(fname, predictions, VAL_ACC, rep_path)

    return jsonify({
        "predictions" : predictions,
        "top_class"   : top_class,
        "confidence"  : predictions[0]["confidence"],
        "image_url"   : url_for("static", filename=f"uploads/{fname}"),
        "gradcam_url" : url_for("static", filename=f"uploads/{cam_fname}"),
        "report_url"  : url_for("download_report", filename=rep_fname),
        "val_acc"     : round(VAL_ACC * 100, 2),
    })


@app.route("/download/<filename>")
def download_report(filename):
    path = os.path.join(REPORT_DIR, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Report not found"}), 404


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too bada hai. Max 16 MB allowed."}), 413
