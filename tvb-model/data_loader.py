from tf_keras.preprocessing.image import ImageDataGenerator
from config import *
import tensorflow as tf
import tf_keras

def get_data_generators():
    train_ds = tf_keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=64,
        label_mode='binary',  # CLASS_MODE에 따라 맞춤
    )

    val_ds = tf_keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMG_SIZE,
        batch_size=64,
        label_mode='binary',
    )

    AUTOTUNE = tf.data.AUTOTUNE
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # 증강은 따로 추가 가능 (아래 방법 참고)

    data_augmentation = tf_keras.Sequential([
        tf_keras.layers.Rescaling(1. / 255),
        tf_keras.layers.RandomRotation(0.2),
        tf_keras.layers.RandomZoom(0.2),
        tf_keras.layers.RandomFlip("horizontal"),
        tf_keras.layers.RandomTranslation(0.1, 0.1),
    ])

    def preprocess(image, label):
        # image에만 augmentation을 적용하고 label은 그대로 반환
        image = data_augmentation(image)
        return image, label

    train_ds = (
        train_ds
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds

# def get_data_generators():
#     train_datagen = ImageDataGenerator(
#         rescale=RESCALE,
#         rotation_range=ROTATION_RANGE,
#         width_shift_range=WIDTH_SHIFT_RANGE,
#         height_shift_range=HEIGHT_SHIFT_RANGE,
#         shear_range=SHEAR_RANGE,
#         zoom_range=ZOOM_RANGE,
#         horizontal_flip=HORIZONTAL_FLIP,
#         brightness_range=BRIGHTNESS_RANGE,
#         fill_mode=FILL_MODE
#     )
#
#     val_datagen = ImageDataGenerator(rescale=RESCALE)
#
#     train_generator = train_datagen.flow_from_directory(
#         TRAIN_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode=CLASS_MODE,
#     )
#
#     validation_generator = val_datagen.flow_from_directory(
#         VALIDATION_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode=CLASS_MODE,
#     )
#     return train_generator, validation_generator

def get_testData_generators():
    test_datagen = ImageDataGenerator(rescale=RESCALE)

    return test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=False
    )