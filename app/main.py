from fastapi import FastAPI, UploadFile, File, Form
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import config

app = FastAPI(title="AgroVision Field Inference Engine")

# Load ONNX Session
session = ort.InferenceSession("agrovision.onnx")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(config.IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

@app.post("/predict")
async def predict(file: UploadFile = File(...), weather_data: str = Form(...)):
    img_tensor = preprocess_image(await file.read())
    
    # Parse weather array from request
    weather_seq = np.array(eval(weather_data), dtype=np.float32).reshape(1, 7, 7)

    inputs = {
        "image": img_tensor,
        "weather": weather_seq
    }
    logits, attn = session.run(None, inputs)
    pred_idx = int(np.argmax(logits, axis=1)[0])

    return {
        "disease_predicted": config.CLASS_LABELS[pred_idx],
        "confidence": float(np.max(np.exp(logits) / np.sum(np.exp(logits)))),
        "attention_distribution": attn.tolist()
    }