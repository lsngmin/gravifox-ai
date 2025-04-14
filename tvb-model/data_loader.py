from tf_keras.preprocessing import image_dataset_from_directory
from tf_keras.preprocessing.image import ImageDataGenerator
from config import *
import tensorflow as tf

def load_data_from_directory(directory, batch_size, img_size, augment_fn=None):
    """
    디렉토리에서 이미지를 로드하고 tf.data.Dataset으로 변환 및 증강 함수 적용
    """
    dataset = image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        label_mode=CLASS_MODE,
        shuffle=True,  # 학습 데이터는 섞어줘야 하므로 True
    )

    # 이미지 크기와 증강을 처리하기 위해 `map` 사용
    if augment_fn:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def get_data_generators():
    # 훈련 데이터와 검증 데이터를 로드하고 증강 설정 적용
    train_dataset = load_data_from_directory(TRAIN_DIR, BATCH_SIZE, IMG_SIZE, augment)
    val_dataset = load_data_from_directory(VALIDATION_DIR, BATCH_SIZE, IMG_SIZE)

    return train_dataset, val_dataset



def get_testData_generators():
    test_dataset = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode=CLASS_MODE,
        shuffle=False  # 테스트 데이터는 섞지 않음
    )
    return test_dataset


def augment(image, label):
    """
    데이터 증강 함수
    """

    # 회전 (ROTATION_RANGE 범위 내에서)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # 회전
    if ROTATION_RANGE > 0:
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    # 밝기 조정 (BRIGHTNESS_RANGE 범위 내에서)
    image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_RANGE[1] - 1)

    # 이미지 크기 조정 (ZOOM_RANGE 범위 내에서)
    if ZOOM_RANGE > 0:
        image = tf.image.random_zoom(image, zoom_range=(1-ZOOM_RANGE, 1+ZOOM_RANGE))

    # 색상 변화 (SHEAR_RANGE 적용)
    image = tf.image.random_hue(image, max_delta=SHEAR_RANGE)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # 크기 조정
    image = tf.image.resize(image, IMG_SIZE)

    return image, label



#
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
#         workers=32,
#         max_queue_size=2560
#     )
#
#     validation_generator = val_datagen.flow_from_directory(
#         VALIDATION_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode=CLASS_MODE,
#         worker=32,
#         max_queue_size=2560
#     )
#     return train_generator, validation_generator
#
# def get_testData_generators():
#     test_datagen = ImageDataGenerator(rescale=RESCALE)
#
#     return test_datagen.flow_from_directory(
#         TEST_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode=CLASS_MODE,
#         shuffle=False
#     )