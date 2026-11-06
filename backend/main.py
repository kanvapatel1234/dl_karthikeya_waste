import io
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------
# PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
METRICS_PATH = BASE_DIR / "results" / "metrics.json"

IMG_SIZE = (224, 224)

# --------------------------------
# FASTAPI INITIALIZATION
# --------------------------------

app = FastAPI(
    title="Waste Classification API",
    description="MobileNetV2-based waste classification system",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# LOAD MODEL ONCE
# --------------------------------

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Load class names from evaluation results
with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

CLASS_NAMES = metrics["class_names"]


# --------------------------------
# IMAGE PREPROCESSING
# --------------------------------

def preprocess_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------
# ROOT ENDPOINT
# --------------------------------

@app.get("/")
def home():

    return {
        "message": "Waste Classification API is running",
        "model": "MobileNetV2",
        "classes": CLASS_NAMES
    }


# --------------------------------
# PREDICTION ENDPOINT
# --------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are supported."
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        image_array = preprocess_image(image_bytes)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(predictions[predicted_index])

        probabilities = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# --------------------------------
# EVALUATION ENDPOINT
# --------------------------------

@app.get("/evaluation")
def get_evaluation():

    return metrics