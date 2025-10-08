# # backend/leaf_api.py

import io
import torch
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import requests

# --- Initialize FastAPI ---
app = FastAPI(title="Leaf Disease Detection API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your LAN IP or Expo URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classes ---
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Load model ---
model = models.mobilenet_v2(num_classes=len(classes))
import os
model_path = os.path.join(os.path.dirname(__file__), "saved_models", "leaf_mobilenet.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    tensor = transform(image).unsqueeze(0).to(device)
    print("Tensor shape:", tensor.shape)  # DEBUG: check shape
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# --- Gemini API ---

# --- OpenRouter API ---
OPENROUTER_API_KEY = "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"  # <-- Replace with your actual OpenRouter API key
OPENROUTER_MODEL = "microsoft/mai-ds-r1:free"  # You can change to another supported model if desired


def get_openrouter_insights(disease_name: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        f"You are a plant disease expert.\n"
        f"For the disease: {disease_name}, provide ONLY the most useful and actionable treatment and care tips. "
        f"Format the answer with clear section headings (like 'Symptoms', 'Treatment', 'Prevention'), and under each heading, give concise bullet points. "
        f"Do NOT include any unnecessary information, introductions, or disclaimers. Only show practical, actionable insights."
    )
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a plant disease expert."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Optional: Post-process to keep only headings and bullet points
        import re
        useful_lines = []
        for line in text.splitlines():
            if re.match(r"^\s*([A-Z][A-Za-z ]+:|[-•*])", line):
                useful_lines.append(line.strip())
        formatted = "\n".join(useful_lines)
        return formatted or "No insights generated."
    except requests.exceptions.RequestException as e:
        return f"OpenRouter API request failed: {e}"

# --- API routes ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        disease = predict_image(contents)
        insights = get_openrouter_insights(disease)
        return {"disease": disease, "insights": insights}
    except Exception as e:
        # Always return status key for uniform frontend handling
        return {"error": str(e), "status": "error"}

@app.get("/")
def root():
    return {"message": "Leaf Disease Detection API is running!"}



