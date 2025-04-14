from tf_keras.preprocessing.image import ImageDataGenerator
from config import *
import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=RESCALE,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        brightness_range=BRIGHTNESS_RANGE,
        fill_mode=FILL_MODE
    )

    val_datagen = ImageDataGenerator(rescale=RESCALE)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE
    )

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE
    )
    return train_generator, validation_generator

def get_testData_generators():
    test_datagen = ImageDataGenerator(rescale=RESCALE)

    return test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=False
    )

def decode_and_resize(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) * RESCALE
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.1)

    image = tf.image.random_rotation(image, ROTATION_RANGE / 360.0)
    image = tf.image.random_zoom(image, [1 - ZOOM_RANGE, 1 + ZOOM_RANGE])
    return image, label

def load_dataset_from_directory(directory):
    class_names = sorted(os.listdir(directory))
    all_image_paths = []
    all_labels = []

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(directory, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_paths.append(os.path.join(class_path, fname))
                all_labels.append(idx)

    path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))
    ds = path_ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    return ds

def get_data_generators():
    train_ds = load_dataset_from_directory(TRAIN_DIR)
    val_ds = load_dataset_from_directory(VALIDATION_DIR)

    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, val_ds

def get_testData_generators():
    test_ds = load_dataset_from_directory(TEST_DIR)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return test_ds