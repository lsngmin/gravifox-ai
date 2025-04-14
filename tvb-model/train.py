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

    train_ds, val_ds = get_data_generators()

    model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        callbacks=[es(), tb()],
        use_multiprocessing=True,
        workers=64,
        max_queue_size=256
    )
    model.save("Xception", save_format="tf")