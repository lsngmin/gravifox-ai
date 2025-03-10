from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# 모델 구성
xception_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

x = xception_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(2, activation='softmax')(x)  # 예: 2개의 클래스 (fake, real)

model = Model(inputs=xception_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터셋 경로 및 파라미터
train_dir = "Dataset/Train/"
validation_dir = "Dataset/Validation/"
batch_size = 32
img_size = (256, 256)

# 학습 데이터셋 로드
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=img_size
)

# 검증 데이터셋 로드
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=False,  # 검증 데이터는 셔플할 필요 없음
    batch_size=batch_size,
    image_size=img_size
)

# 클래스 이름 확인
class_names = train_dataset.class_names
print("클래스:", class_names)

# 정규화 레이어 및 데이터셋 변환
normalization_layer = tf.keras.layers.Rescaling(1./255)

def one_hot_encode(image, label):
    return normalization_layer(image), tf.one_hot(label, depth=len(class_names))

train_dataset = train_dataset.map(one_hot_encode)
validation_dataset = validation_dataset.map(one_hot_encode)

# 모델 학습
model.fit(
    train_dataset,
    epochs=10,
    batch_size=batch_size,
    validation_data=validation_dataset  # 검증 데이터 추가
)