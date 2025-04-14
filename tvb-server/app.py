from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()
class Data(BaseModel):
    message: str

#$loaded_model = tf.keras.models.load_model('xception_model-2.h5')

def predict_image(image: Image.Image):


    img = image.resize((256, 256))  # 이미지 불러오기 및 크기 조정
    img_array = img_to_array(img)  # numpy 배열 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 224, 224, 3)
    img_array = img_array / 255.0  # 정규화 (모델 학습 시 정규화했다면 필요)

    loaded_model = tf.keras.models.load_model('../tvb-model/xception_model-2.h5')

    prediction = loaded_model.predict(img_array)
    return prediction

@app.post("/receive_data")
async def receive_data(data: Data):
    print(f"Received data: {data}")
    return {"received_message": data.message}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    prediction = predict_image(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return JSONResponse(content={"predicted_class": predicted_class})
    # return {"received_message": "good"}

@app.get("/")
async  def index():
    return {"received_message": "Hello World"}