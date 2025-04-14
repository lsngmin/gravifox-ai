from tf_keras.preprocessing import image_dataset_from_directory
from tf_keras.preprocessing.image import ImageDataGenerator
from config import *
import tensorflow as tf
import os
import glob
import tensorflow as tf
from config import *


def get_dataset_from_directory(directory, is_training=True):
    """Create a tf.data.Dataset from directory."""
    # Get all class subdirectories
    class_dirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    class_indices = {cls_name: i for i, cls_name in enumerate(class_dirs)}

    # Build paths and labels
    image_paths = []
    labels = []

    for class_name in class_dirs:
        class_path = os.path.join(directory, class_name)
        class_idx = class_indices[class_name]

        # Get all image files in this class directory
        class_images = glob.glob(os.path.join(class_path, "*.jpg")) + \
                       glob.glob(os.path.join(class_path, "*.jpeg")) + \
                       glob.glob(os.path.join(class_path, "*.png"))

        image_paths.extend(class_images)

        if CLASS_MODE == 'categorical':
            # One-hot encoding
            label = tf.keras.utils.to_categorical(class_idx, num_classes=len(class_dirs))
            labels.extend([label] * len(class_images))
        else:
            # Binary or sparse
            labels.extend([class_idx] * len(class_images))

    # Create tensor slices
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Combine paths and labels
    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    # Shuffle during training
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Map to parse and augment images
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def parse_image(file_path, label):
    """Parse image from a file path."""
    # file_path is now correctly a string tensor
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # rescale
    return img, label


def augment_image(image, label):
    """Apply image augmentations."""
    # Random rotation
    if ROTATION_RANGE:
        angle = tf.random.uniform([], -ROTATION_RANGE, ROTATION_RANGE, dtype=tf.float32)
        image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))

    # Horizontal flip
    if HORIZONTAL_FLIP:
        image = tf.image.random_flip_left_right(image)

    # Brightness adjustment
    if BRIGHTNESS_RANGE:
        min_brightness, max_brightness = BRIGHTNESS_RANGE
        brightness_factor = tf.random.uniform([], min_brightness, max_brightness)
        image = tf.image.adjust_brightness(image, brightness_factor - 1.0)

    # Random zoom (center crop and resize)
    if ZOOM_RANGE:
        scale = tf.random.uniform([], 1.0 - ZOOM_RANGE, 1.0 + ZOOM_RANGE)
        new_h = tf.cast(tf.cast(IMG_SIZE[0], tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(IMG_SIZE[1], tf.float32) * scale, tf.int32)

        # Resize then crop back to original size (zoom effect)
        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])

    # Ensure values are in valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def get_data_generators():
    """Create training and validation datasets."""
    train_dataset = get_dataset_from_directory(TRAIN_DIR, is_training=True)
    validation_dataset = get_dataset_from_directory(VALIDATION_DIR, is_training=False)
    return train_dataset, validation_dataset


def get_testData_generators():
    """Create test dataset."""
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