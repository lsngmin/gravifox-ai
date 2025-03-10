from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

xception_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

x = xception_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(2, activation='softmax')(x)  # 예: 10개의 클래스


model = Model(inputs=xception_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = "Dataset/Train/"
batch_size = 32
img_size = (256, 256)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,  # 데이터 셔플링
    batch_size=batch_size,
    image_size=img_size
)

class_names = train_dataset.class_names
print("클래스:", class_names)  # ['fake', 'real']

normalization_layer = tf.keras.layers.Rescaling(1./255)

def one_hot_encode(image, label):
    return normalization_layer(image), tf.one_hot(label, depth=len(class_names))

train_dataset = train_dataset.map(one_hot_encode)

model.fit(train_dataset, epochs=10, batch_size=32)