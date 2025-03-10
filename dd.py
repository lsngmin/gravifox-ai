from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
print(f"사용 가능한 GPU 수: {strategy.num_replicas_in_sync}")

train_dir = "Dataset/Train/"
batch_size = 32 * strategy.num_replicas_in_sync
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

with strategy.scope():
    xception_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    x = xception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # 여기가 문제였음
    x = Dense(len(class_names), activation='softmax')(x)  # 클래스 개수만큼 출력

    model = Model(inputs=xception_model.input, outputs=x)  # ✅ 수정됨
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 추가
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=3, restore_best_weights=True
)

model.fit(train_dataset, epochs=10, batch_size=32)