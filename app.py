from fastapi import FastAPI, File,  UploadFile
import requests
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask, redirect, url_for, request, render_template

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://your-vercel-app.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Model = tf.keras.layers.TFSMLayer("vgg16_fruit_disease_classifier\kaggle\working\vgg16_fruit_disease_classifier")

Model = tf.keras.models.load_model("vgg16_fruit_disease_classifier.h5")

# Model = tf.keras.models.load_model("VillageDataModellargemodel.h5")

CLASS_NAMES = [
    'Apple-Apple_scab', 'Apple-Black_rot', 'Apple-Cedar_apple_rust',
    'Apple-healthy', 'Blueberry-healthy',
    'Cherry-(including_sour)-Powdery_mildew', 'Cherry-(including_sour)-healthy',
    'Corn-(maize)-Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn-(maize)-Common_rust', 'Corn-(maize)-Northern_Leaf_Blight',
    'Corn-(maize)-healthy', 'Grape-Black_rot', 'Grape-Esca-(Black_Measles)',
    'Grape-Leaf_blight-(Isariopsis_Leaf_Spot)', 'Grape-healthy',
    'Orange-Haunglongbing-(Citrus_greening)', 'Peach-Bacterial_spot',
    'Peach-healthy', 'Pepper,_bell-Bacterial_spot',
    'Pepper,_bell-healthy', 'Potato-Early_blight', 'Potato-Late_blight',
    'Potato-healthy', 'Raspberry-healthy', 'Soybean-healthy',
    'Squash-Powdery_mildew', 'Strawberry-Leaf_scorch', 'Strawberry-healthy',
    'Tomato-Bacterial_spot', 'Tomato-Early_blight', 'Tomato-Late_blight',
    'Tomato-Leaf_Mold', 'Tomato-Septoria_leaf_spot',
    'Tomato-Spider_mites_Two-spotted_spider_mite', 'Tomato-Target_Spot',
    'Tomato-Tomato_Yellow_Leaf_Curl_Virus', 'Tomato-Tomato_mosaic_virus',
    'Tomato-healthy'
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image to (224, 224)
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, 0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

@app.post("/predict")
async def predict(file: UploadFile = File()):
    image = read_file_as_image(await file.read())
    predictions = Model.predict(image)
    predictions_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return {
        'class': predictions_class,
        'confidence' : float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

# if __name__ == '__main__':
#     app.run(debug=True)


# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)