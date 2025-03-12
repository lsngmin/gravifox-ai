from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.list_physical_devices('GPU')  # GPU 목록 확인
if gpus:
    try:
        # 모든 GPU에 대해 메모리 자동 확장 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("모든 GPU에 대해 메모리 자동 확장 설정 완료.")
    except RuntimeError as e:
        print("GPU 메모리 자동 확장 설정 실패:", e)

# 모델 구성
xception_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

x = xception_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)  # 드롭아웃 추가
x = Dense(1, activation='sigmoid')(x)  # 예: 2개의 클래스 (fake, real)

# 초기 학습률, 총 훈련 스텝
initial_learning_rate = 0.001
decay_steps = 43750  # 총 훈련 스텝 수에 맞춰 조정

# CosineDecay 학습률 스케줄러 설정
lr_schedule = CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=0.1  # 최소 학습률 (전체 학습률의 10%)
)

# Adam 옵티마이저에 학습률 스케줄러 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model = Model(inputs=xception_model.input, outputs=x)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

train_dir = "Dataset/Train/"
validation_dir = "Dataset/Validation/"
batch_size = 32
img_size = (256, 256)

# 학습 데이터셋 로드
train_datagen = ImageDataGenerator(
    rescale=1./255,               # 정규화
    rotation_range=40,            # 회전
    width_shift_range=0.2,        # 수평 이동
    height_shift_range=0.2,       # 수직 이동
    shear_range=0.2,              # 기울기 변환
    zoom_range=0.2,               # 확대/축소
    horizontal_flip=True,         # 수평 반전
    brightness_range=[0.5, 1.5],  # 밝기 변화
    fill_mode='nearest'           # 비어있는 공간 채우기
)

# 검증 데이터셋은 정규화만 적용
val_datagen = ImageDataGenerator(rescale=1./255)

# 학습 및 검증 데이터 생성기
train_generator = train_datagen.flow_from_directory(
    train_dir,                    # 훈련 데이터 디렉토리
    target_size=(256, 256),       # 입력 이미지 크기
    batch_size=batch_size,
    class_mode='binary'           # 이진 분류
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,               # 검증 데이터 디렉토리
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary'
)








# Mixed Precision Training 활성화
policy = mixed_precision.Policy('mixed_float16')  # mixed_float16은 16비트와 32비트 혼합 훈련을 사용
mixed_precision.set_global_policy(policy)  # set_policy 대신 set_global_policy 사용

# EarlyStopping 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',  # 'val_loss'가 개선되지 않으면 학습을 중지
    patience=5,          # 3 에폭 동안 개선되지 않으면 중지
    restore_best_weights=True  # 가장 좋은 모델 가중치를 복원
)

# 학습률 감소 콜백 설정 (Learning Rate Scheduler)
# lr_scheduler = ReduceLROnPlateau(
#     monitor='val_loss',  # 'val_loss'가 개선되지 않으면 학습률을 낮춤
#     factor=0.2,          # 학습률을 20%로 감소
#     patience=2,          # 2 에폭 동안 개선되지 않으면 학습률을 낮춤
#     verbose=1            # 변경 시 출력
# )

# TensorBoard 콜백 설정 (시각화 로그 저장)
tensorboard_callback = TensorBoard(
    log_dir='../logs',    # 로그 디렉토리 경로
    histogram_freq=1     # 히스토그램 주기 설정
)

# 모델 학습
model.fit(
    train_generator,
    epochs=20,
    batch_size=batch_size,
    validation_data=validation_generator,  # 검증 데이터 추가
    callbacks=[early_stopping, tensorboard_callback]  # EarlyStopping, 학습률 조정, TensorBoard 콜백 추가
)
