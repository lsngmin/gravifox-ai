from tf_keras.preprocessing import image_dataset_from_directory
from tf_keras.preprocessing.image import ImageDataGenerator
from config import *
import tensorflow as tf

import tensorflow as tf
from config import *


def parse_image(file_path, label):
    """Parse image and label from file path."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # rescale equivalent to RESCALE=1./255
    return img, label


def augment_image(image, label):
    """Apply augmentations similar to your ImageDataGenerator settings."""
    # Random rotation
    if ROTATION_RANGE:
        angle = tf.random.uniform([], -ROTATION_RANGE, ROTATION_RANGE, dtype=tf.float32)
        image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))

    # Width and height shifts
    if WIDTH_SHIFT_RANGE:
        w_shift = tf.random.uniform([], -WIDTH_SHIFT_RANGE, WIDTH_SHIFT_RANGE) * IMG_SIZE[0]
        image = tf.roll(image, tf.cast(w_shift, tf.int32), axis=1)

    if HEIGHT_SHIFT_RANGE:
        h_shift = tf.random.uniform([], -HEIGHT_SHIFT_RANGE, HEIGHT_SHIFT_RANGE) * IMG_SIZE[1]
        image = tf.roll(image, tf.cast(h_shift, tf.int32), axis=0)

    # Random zoom
    if ZOOM_RANGE:
        zoom_factor = tf.random.uniform([], 1.0 - ZOOM_RANGE, 1.0 + ZOOM_RANGE)
        orig_height, orig_width = IMG_SIZE
        h = tf.cast(orig_height * zoom_factor, tf.int32)
        w = tf.cast(orig_width * zoom_factor, tf.int32)
        image = tf.image.resize(image, [h, w])
        image = tf.image.resize_with_crop_or_pad(image, orig_height, orig_width)

    # Horizontal flip
    if HORIZONTAL_FLIP:
        image = tf.image.random_flip_left_right(image)

    # Brightness adjustment
    if BRIGHTNESS_RANGE:
        min_brightness, max_brightness = BRIGHTNESS_RANGE
        brightness_factor = tf.random.uniform([], min_brightness, max_brightness)
        image = tf.image.adjust_brightness(image, brightness_factor - 1.0)
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def get_dataset_from_directory(directory, is_training=True):
    """Create a tf.data.Dataset from directory."""
    # Get file paths and labels
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        batch_size=None,  # We'll batch later
        image_size=IMG_SIZE,
        label_mode='categorical' if CLASS_MODE == 'categorical' else 'binary',
        shuffle=is_training
    )

    # Parse images
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation during training
    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(BATCH_SIZE)

    # Use prefetch to overlap data preprocessing and model execution
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_data_generators():
    """Replacement for the original function using tf.data API."""
    train_dataset = get_dataset_from_directory(TRAIN_DIR, is_training=True)
    validation_dataset = get_dataset_from_directory(VALIDATION_DIR, is_training=False)
    return train_dataset, validation_dataset


def get_testData_generators():
    """Replacement for the test data function using tf.data API."""
    test_dataset = get_dataset_from_directory(TEST_DIR, is_training=False)
    return test_dataset

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