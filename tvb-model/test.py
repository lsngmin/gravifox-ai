import numpy as np
import tensorflow as tf
from tf_keras.src.utils import load_img, img_to_array

from data_loader import get_testData_generators
import tf_keras

model = tf_keras.models.load_model("Xception")

image_path = "ex/t1.jpg"  # 테스트할 이미지 파일 경로

# 1. 이미지 로드 및 전처리
img_size = (256, 256) # 모델이 학습한 입력 크기에 맞춰야 함
img = load_img(image_path, target_size=img_size)  # 이미지 불러오기 및 크기 조정
img_array = img_to_array(img)  # numpy 배열 변환
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 224, 224, 3)
img_array = img_array / 255.0  # 정규화 (모델 학습 시 정규화했다면 필요)

prediction = model.predict(img_array)

# 3. 결과 출력
print(f"Prediction: {prediction}")