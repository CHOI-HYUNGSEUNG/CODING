import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications
from tensorflow.keras import losses

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
    
    image = cv2.imread(image_path)  # 그레이스케일이 아닌 기본적으로 3채널 이미지 로드
    image = cv2.resize(image, (224, 224))  # VGG16 모델의 기본 입력 크기로 조정
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
# 전이 학습을 위한 사전 훈련된 모델 불러오기
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# 기존 모델의 상위 층 동결
for layer in base_model.layers:
    layer.trainable = False
# 새로운 출력층 추가
global_average_layer = layers.GlobalAveragePooling2D()(base_model.output)
output_layer = layers.Dense(4, activation='softmax')(global_average_layer)
# 새로운 모델 정의
model = models.Model(inputs=base_model.input, outputs=output_layer)
# 모델 컴파일
# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=losses.CategoricalCrossentropy(),  # 교차 엔트로피 손실 함수로 변경
              metrics=['accuracy'])
# 모델 훈련
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# 결과 출력: 이미지 및 압력 센서 그래프
for i in range(len(x_test)):
    image = x_test[i]
    pressure_values = y_test[i]

    plt.figure(figsize=(8, 4))

    # 이미지 출력
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')

    # 압력 센서 그래프 출력
    plt.subplot(1, 2, 2)
    plt.plot(pressure_values)
    plt.title('Pressure Sensor Values')
    plt.xlabel('Sensor Index')
    plt.ylabel('Pressure')
    plt.grid(True)

    plt.tight_layout()
    plt.show()