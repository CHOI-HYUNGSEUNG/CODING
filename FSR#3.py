import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# 이미지 경로와 FSR값 파일 경로 매핑
mapping_table = {
    "examples/1.jpg": "examples/1_fsr.txt",
    "examples/2.jpg": "examples/2_fsr.txt",
    "examples/3.jpg": "examples/3_fsr.txt",
    "examples/4.jpg": "examples/4_fsr.txt",
    "examples/5.jpg": "examples/5_fsr.txt",
}

# 데이터 전처리
data = []
for image_file, pressure_file in mapping_table.items():
    image_path = os.path.join("C:/Users/user/Desktop", image_file)
    pressure_path = os.path.join("C:/Users/user/Desktop", pressure_file)
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (255, 255))  # 이미지 크기 조정
    image = np.expand_dims(image, axis=-1)  # 채널 차원 추가
    image = image / 255.0

    with open(pressure_path, 'r') as file:
        pressure_lines = file.readlines()
        pressure_data = []
        for line in pressure_lines:
            line = line.strip()
            pressure_values = line.split(',')
            pressure_data.extend([float(value) for value in pressure_values])

    data.append((image, pressure_data))

# 데이터셋 분리
images = np.array([d[0] for d in data])
pressures = np.array([d[1] for d in data])

x_train, x_test, y_train, y_test = train_test_split(
    images, pressures, test_size=0.2, random_state=42)

# 모델 정의 (CNN 모델)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(255, 255, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 학습률 조정
              loss='mse',
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# 학습 곡선 그리기
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()