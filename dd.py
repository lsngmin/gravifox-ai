from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# 다중 GPU 전략 설정
strategy = tf.distribute.MirroredStrategy()
print(f"사용 가능한 GPU 수: {strategy.num_replicas_in_sync}")

# 데이터셋 설정
train_dir = "Dataset/Train/"
val_dir = "Dataset/Validation/"  # 검증 데이터셋 디렉토리
batch_size = 32 * strategy.num_replicas_in_sync
img_size = (256, 256)

# 데이터 로드 (훈련 데이터)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

# 검증 데이터 로드
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

class_names = train_dataset.class_names
print("클래스:", class_names)  # ['fake', 'real']

# 데이터 정규화
normalization_layer = tf.keras.layers.Rescaling(1./255)

def one_hot_encode(image, label):
    return normalization_layer(image), tf.one_hot(tf.cast(label, tf.int32), depth=len(class_names))

train_dataset = train_dataset.map(one_hot_encode)
val_dataset = val_dataset.map(one_hot_encode)

# 모델 구성
with strategy.scope():
    xception_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    xception_model.trainable = False  # 초기에는 가중치 고정

    x = xception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=xception_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 추가
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# 모델 학습
model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[early_stopping])
