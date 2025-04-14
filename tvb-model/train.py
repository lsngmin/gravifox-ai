import tensorflow as tf
from tf_keras import mixed_precision
from callback import *
from data_loader import get_data_generators
from model import build_xception
from config import *

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("모든 GPU에 대해 메모리 자동 확장 설정 완료.")
    except RuntimeError as e:
        print("GPU 메모리 자동 확장 설정 실패:", e)

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = build_xception()

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    train_dataset, val_dataset = get_data_generators()

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # 성능 향상: 데이터 전처리와 모델 훈련을 병렬로 수행
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_dataset,
        callbacks=[es(), tb()],
        use_multiprocessing=True,
        workers=128,
        max_queue_size=512
    )
    model.save("Xception", save_format="tf")