import tf_keras
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from tf_keras.src.utils import load_img, img_to_array

from PIL import Image
import io
import numpy as np
import tensorflow as tf
import time
from datetime import datetime

app = FastAPI()
class Data(BaseModel):
    message: str

def predict_image(image: Image.Image):
    model = tf_keras.models.load_model("../tvb-model/Xception")

    img = image.resize((256, 256))  # 이미지 불러오기 및 크기 조정
    img_array = img_to_array(img)  # numpy 배열 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 224, 224, 3)
    img_array = img_array / 255.0  # 정규화 (모델 학습 시 정규화했다면 필요)


    prediction = model.predict(img_array)
    return prediction

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    start_time = time.time()
    prediction = predict_image(image)
    end_time = time.time()

    analyzeResult = JSONResponse(content=
    {
    "timestamp": datetime.now().timestamp(),
    "used_model": "Xception",
    "image_uuid" : file.filename,
    "prediction_time": round(end_time - start_time, 2),
    "predicted_probability": round(prediction[0].tolist()[0], 4),
    "predicted_class": "Real" if prediction[0].tolist()[0]>0.5 else "Fake",
})
    return analyzeResult
@app.get("/")
async  def index():
    return {"received_message": "Hello World"}