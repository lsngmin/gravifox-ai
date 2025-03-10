from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

xception_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

x = xception_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(2, activation='softmax')(x)  # 예: 10개의 클래스


model = Model(inputs=xception_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
# model.fit(x_train, y_train, epochs=10, batch_size=32)==