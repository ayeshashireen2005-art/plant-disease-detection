import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.h5")
model = load_model(MODEL_PATH)

IMG_SIZE = 224

# -----------------------------
# Load class names
# -----------------------------
CLASS_NAMES = sorted(os.listdir("PlantVillage/train"))

# -----------------------------
# Disease-wise cures
# -----------------------------
DISEASE_CURES = {
    "Apple scab": [
        "Remove infected leaves",
        "Apply fungicide",
        "Avoid overhead watering",
        "Ensure air circulation"
    ],
    "Black rot": [
        "Prune infected branches",
        "Use copper fungicide",
        "Keep orchard clean"
    ],
    "Cedar apple rust": [
        "Remove nearby cedar trees",
        "Use resistant varieties",
        "Apply fungicide in spring"
    ],
}

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = confidence = plant = disease = img_path = cures = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

            img = image.load_img(save_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            idx = int(np.argmax(preds))
            confidence = round(float(np.max(preds)) * 100, 2)

            full_class = CLASS_NAMES[idx]
            parts = full_class.split("___")

            plant = parts[0].replace("_", " ")

            if "healthy" in full_class.lower():
                prediction = "Healthy"
                disease = "Healthy"
                cures = ["No disease detected"]
            else:
                prediction = "Diseased"
                disease = parts[1].replace("_", " ")
                cures = DISEASE_CURES.get(
                    disease,
                    ["Consult agricultural expert"]
                )

            # 🔴 THIS LINE FIXES IMAGE
            img_path = f"/uploads/{file.filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        plant=plant,
        disease=disease,
        img_path=img_path,
        cures=cures
    )

# -----------------------------
# IMAGE SERVING ROUTE (MOST IMPORTANT)
# -----------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
