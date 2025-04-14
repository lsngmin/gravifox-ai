from tf_keras.preprocessing.image import ImageDataGenerator

from config import *


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
        class_mode=CLASS_MODE,
    )

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
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