from fastapi import FastAPI, File,  UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask, redirect, url_for, request, render_template

#
# app = FastAPI()
#
# origins = [
#     "https://your-vercel-app.vercel.app"
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# Model = tf.keras.models.load_model("VillageDataModellargemodel.h5")
#
#
# # ModelBeta = tf.keras.models.load_model("../Models/VillageDataModellargemodel.h5")
#
# # endpoint = "http://localhost:8501/v1/models/potatomodel:predict"
# # endpoint = "https://8515-2405-201-5c23-40c0-eca3-a625-9322-ddd2.ngrok-free.app"
#
#
# CLASS_NAMES = [
#                 'Tomato_Bacterial_spot',
#                 'Tomato_Early_blight',
#                 'Tomato_Late_blight',
#                 'Tomato_Leaf_Mold',
#                 'Tomato_Septoria_leaf_spot',
#                 'Tomato_Spider_mites_Two_spotted_spider_mite',
#                 'Tomato__Target_Spot',
#                 'Tomato__Tomato_YellowLeaf__Curl_Virus',
#                 'Tomato__Tomato_mosaic_virus',
#                 'Tomato_healthy'
#                ]
#
# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')
#
# @app.get("/ping")
# async def ping():
#     return "Hello I am alive"
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.post("/predict")
# async def predict(
#         file: UploadFile = File()
#
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#     predictions = Model.predict(img_batch)
#
#     predictions_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predictions_class,
#         'confidence': float(confidence)
#     }
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
#
#
# # if __name__ == "__main__":
# #     app.run (debug = True)


#
# from fastapi import FastAPI, File, UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# from fastapi.middleware.cors import CORSMiddleware
#
# app = FastAPI()
#
# origins = [
#     "https://your-vercel-app.vercel.app"
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# Model = tf.keras.models.load_model("VillageDataModellargemodel.h5")
#
# CLASS_NAMES = [
#     'Tomato_Bacterial_spot',
#     'Tomato_Early_blight',
#     'Tomato_Late_blight',
#     'Tomato_Leaf_Mold',
#     'Tomato_Septoria_leaf_spot',
#     'Tomato_Spider_mites_Two_spotted_spider_mite',
#     'Tomato__Target_Spot',
#     'Tomato__Tomato_YellowLeaf__Curl_Virus',
#     'Tomato__Tomato_mosaic_virus',
#     'Tomato_healthy'
# ]
#
# @app.get("/")
# async def index():
#     return {"message": "Welcome to the Tomato Disease Prediction API!"}
#
# # @app.route('/')
# # def home():
# #     return render_template('index.html')
#
# @app.get("/ping")
# async def ping():
#     return {"message": "Hello, I am alive"}
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.post("/predict")
# async def predict(file: UploadFile = File()):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#     predictions = Model.predict(img_batch)
#     predictions_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predictions_class,
#         'confidence': float(confidence)
#     }
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)



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

Model = tf.keras.models.load_model("VillageDataModellargemodel.h5")

CLASS_NAMES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File()):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = Model.predict(img_batch)
    predictions_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predictions_class,
        'confidence': float(confidence)
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

if __name__ == '__main__':
    app.run(debug=True)