from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import mixed_precision
from tensorflow.python.keras.regularizers import l2

# GPU 초기화 설정 (모든 GPU에 대해 메모리 자동 확장 방식 설정)
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

# Mixed Precision Training 활성화
policy = mixed_precision.Policy('mixed_float16')  # mixed_float16은 16비트와 32비트 혼합 훈련을 사용
mixed_precision.set_global_policy(policy)  # set_policy 대신 set_global_policy 사용

# EarlyStopping 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',  # 'val_loss'가 개선되지 않으면 학습을 중지
    patience=3,          # 3 에폭 동안 개선되지 않으면 중지
    restore_best_weights=True  # 가장 좋은 모델 가중치를 복원
)

# 학습률 감소 콜백 설정 (Learning Rate Scheduler)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # 'val_loss'가 개선되지 않으면 학습률을 낮춤
    factor=0.2,          # 학습률을 20%로 감소
    patience=2,          # 2 에폭 동안 개선되지 않으면 학습률을 낮춤
    verbose=1            # 변경 시 출력
)

# TensorBoard 콜백 설정 (시각화 로그 저장)
tensorboard_callback = TensorBoard(
    log_dir='./logs',    # 로그 디렉토리 경로
    histogram_freq=1     # 히스토그램 주기 설정
)

# 모델 학습
model.fit(
    train_dataset,
    epochs=10,
    batch_size=batch_size,
    validation_data=validation_dataset,  # 검증 데이터 추가
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback]  # EarlyStopping, 학습률 조정, TensorBoard 콜백 추가
)
