from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# 멀티 GPU 전략 사용
strategy = tf.distribute.MirroredStrategy()
print(f"사용 가능한 GPU 수: {strategy.num_replicas_in_sync}")

# 데이터 경로 및 설정
train_dir = "Dataset/Train/"
batch_size = 32 * strategy.num_replicas_in_sync  # GPU 개수에 따라 배치 크기 조정
img_size = (256, 256)

# 데이터 불러오기
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

# 클래스 정보 확인
class_names = train_dataset.class_names
print("클래스:", class_names)  # ['fake', 'real']

# 데이터 정규화 및 One-Hot Encoding
normalization_layer = tf.keras.layers.Rescaling(1./255)

def one_hot_encode(image, label):
    return normalization_layer(image), tf.one_hot(label, depth=len(class_names))

train_dataset = train_dataset.map(one_hot_encode)

# 멀티 GPU 내에서 모델 정의
with strategy.scope():
    xception_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    x = xception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(len(class_names), activation='softmax')  # 클래스 개수만큼 출력

    model = Model(inputs=xception_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping 콜백 추가
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=3, restore_best_weights=True
)

# 모델 학습
model.fit(train_dataset, epochs=10, batch_size=batch_size, callbacks=[early_stopping])
