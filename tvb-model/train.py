from dd import train_generator, validation_generator
from model import build_xception
import tensorflow as tf
from data_loader import get_data_generators
from tensorflow.keras import mixed_precision
from config import BATCH_SIZE
from callback import *
gpus = tf.config.list_physical_devices('GPU')  # GPU 목록 확인
if gpus:
    try:
        # 모든 GPU에 대해 메모리 자동 확장 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("모든 GPU에 대해 메모리 자동 확장 설정 완료.")
    except RuntimeError as e:
        print("GPU 메모리 자동 확장 설정 실패:", e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

model = build_xception()
train, validation = get_data_generators()

model.fit(
    train,
    epochs=20,
    batch_size=32,
    validation_data=validation,
    callbacks=[es(), tb()]
)
model.save("xception_model")