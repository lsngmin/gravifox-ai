from tf_keras.preprocessing import image_dataset_from_directory
from tf_keras.preprocessing.image import ImageDataGenerator
from config import *
import tensorflow as tf
import tensorflow as tf
import os
import glob
import multiprocessing
from config import *

# Get number of available CPU cores for parallelism
NUM_CPU = multiprocessing.cpu_count()


def get_dataset_from_directory(directory, is_training=True):
    """Create an optimized tf.data.Dataset from directory with proper parallelism."""
    # Use tf.data.Dataset.list_files for better performance
    pattern = os.path.join(directory, "*/*")  # Path pattern for all images in subdirectories
    file_dataset = tf.data.Dataset.list_files(pattern, shuffle=is_training)

    # Get class names from directory
    class_names = sorted([item for item in os.listdir(directory)
                          if os.path.isdir(os.path.join(directory, item))])

    class_dict = {class_name: i for i, class_name in enumerate(class_names)}

    # Set the number of parallel calls for better performance
    PARALLEL_CALLS = min(NUM_CPU, 16)  # Use up to 16 cores but not more

    # Extract class from file path and combine with file path
    def process_path(file_path):
        # Extract class name from the file path
        parts = tf.strings.split(file_path, os.path.sep)
        class_name = parts[-2]

        # Convert class name to index
        class_idx = tf.constant(-1, dtype=tf.int64)
        for name, idx in class_dict.items():
            is_match = tf.equal(class_name, name)
            class_idx = tf.where(is_match, idx, class_idx)

        # Handle class mode
        if CLASS_MODE == 'categorical':
            # One-hot encoding
            label = tf.one_hot(class_idx, depth=len(class_names))
        elif CLASS_MODE == 'binary':
            label = tf.cast(class_idx, tf.float32)
        else:  # 'sparse'
            label = class_idx

        return file_path, label

    # Create the dataset of image path and label pairs
    dataset = file_dataset.map(process_path, num_parallel_calls=PARALLEL_CALLS)

    # Optimize dataset performance with caching
    if not is_training:  # Only cache validation/test sets
        dataset = dataset.cache()

    if is_training:
        # Shuffle with a large enough buffer
        dataset_size = sum([len(files) for _, _, files in os.walk(directory)])
        buffer_size = min(dataset_size, 10000)  # Don't use too much memory
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Parse images in parallel
    dataset = dataset.map(parse_image, num_parallel_calls=PARALLEL_CALLS)

    # Apply augmentation during training
    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=PARALLEL_CALLS)

    # Batch images
    dataset = dataset.batch(BATCH_SIZE)

    # Use prefetch to overlap data preprocessing and model execution
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def parse_image(file_path, label):
    """Parse image from a file path with error handling."""
    # Read the image file
    img = tf.io.read_file(file_path)

    # Decode with error handling
    def _decode_img():
        # Try different decoders based on file extension
        decoded = tf.image.decode_image(img, channels=3, expand_animations=False)
        decoded.set_shape([None, None, 3])
        return decoded

    # Use try_and_recover pattern
    img = _decode_img()

    # Resize the image
    img = tf.image.resize(img, IMG_SIZE)

    # Normalize pixel values
    img = tf.cast(img, tf.float32) / 255.0

    return img, label


def augment_image(image, label):
    """Apply image augmentations efficiently."""
    # Random flip
    if HORIZONTAL_FLIP:
        image = tf.image.random_flip_left_right(image)

    # Random rotation - use more efficient built-in functions
    if ROTATION_RANGE:
        # Convert degrees to radians for tf.image.rot90
        image = tf.image.rot90(
            image,
            k=tf.random.uniform([], 0, 4, dtype=tf.int32)  # 0, 90, 180, or 270 degrees
        )

    # Random brightness
    if BRIGHTNESS_RANGE and BRIGHTNESS_RANGE[0] != BRIGHTNESS_RANGE[1]:
        delta = (BRIGHTNESS_RANGE[1] - BRIGHTNESS_RANGE[0]) / 2.0
        image = tf.image.random_brightness(image, max_delta=delta)

    # Random zoom
    if ZOOM_RANGE:
        # Implementation that's more efficient
        scale = 1.0 + tf.random.uniform([], -ZOOM_RANGE, ZOOM_RANGE)
        shape = tf.shape(image)
        h, w = shape[0], shape[1]

        # New dimensions
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

        # Resize and crop back to original size
        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.resize_with_crop_or_pad(image, h, w)

    # Ensure values stay within the valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def get_data_generators():
    """Create training and validation datasets with optimized settings."""
    train_dataset = get_dataset_from_directory(TRAIN_DIR, is_training=True)
    validation_dataset = get_dataset_from_directory(VALIDATION_DIR, is_training=False)
    return train_dataset, validation_dataset


def get_testData_generators():
    """Create test dataset with optimized settings."""
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