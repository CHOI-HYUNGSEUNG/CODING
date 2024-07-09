import numpy as np
import tensorflow as tf
import cv2
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

# 이미지, 압력센서값 매핑 함수
def load_images_and_FSR(folder, label):
    images = []
    pressures = []

    # 파일 목록을 정렬
    file_list = sorted(os.listdir(folder), key=lambda x: int(re.search(r'(\d+)', x).group()))

    for filename in file_list:
        img_path = os.path.join(folder, filename)

        # .jpg 확장자를 가진 이미지 파일만 처리
        if filename.endswith(".jpg"):
            # .txt 파일명 생성
            pressure_filename = filename.replace(".jpg", "_fsr.txt")
            pressure_filepath = os.path.join(folder, pressure_filename)

            # 이미지 로드 및 압력센서값 확인
            if os.path.exists(pressure_filepath):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (255, 255))
                    images.append(img)
                    
                    with open(pressure_filepath, 'r') as pressure_file:
                        fsr_values = [float(value) for value in pressure_file.read().strip().split('\n') if value.replace('.', '', 1).isdigit()]
                        pressures.append(fsr_values)
                else:
                    print(f"이미지 로드 실패: {img_path}")
            else:
                pressures.append([0.0, 0.0, 0.0, 0.0])

    return images, pressures, [label] * len(images)

# 데이터 경로 지정 (경로 구분자로 역슬래시 사용시 r 접두사 추가)
data_path_image1 = r'C:\Users\user\Desktop\examples'
data_path_image2 = r'C:\Users\user\Desktop\examples'

# 이미지1, 압력센서값 매핑
images_image1, pressures_image1, images_labels_image1 = load_images_and_FSR(data_path_image1, label=0)

# 이미지2, 압력센서값 매핑
images_image2, pressures_image2, images_labels_image2 = load_images_and_FSR(data_path_image2, label=1)

# 이미지와 압력 센서값을 하나로 합치기
images = images_image1 + images_image2
pressures = pressures_image1 + pressures_image2
images_labels = images_labels_image1 + images_labels_image2

# 데이터 전처리
images = np.array(images) / 255.0

# 데이터셋 분리  
if len(images) > 0:
    # 사용 가능한 데이터가 있을 때만 분리
    test_size = min(0.2, len(images) - 1)
    x_train, x_test, y_train, y_test = train_test_split(images, images_labels, test_size=test_size, random_state=42)
    
    # 모델 정의
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(255, 255, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 훈련
    model.fit(x_train, np.array(y_train), epochs=5, batch_size=32)

    # 모델 평가
    model.evaluate(x_test, np.array(y_test), verbose=2)

    # 모델 예측
    predictions = model.predict(x_test)

    # 예측 결과 및 정답 출력
    for i in range(len(predictions)):
        predicted_label = np.argmax(predictions[i])
        true_label = y_test[i]

        # 이미지 플로팅
        plt.subplot(1, 2, 1)
        plt.imshow(x_test[i].reshape(255, 255), cmap='gray')
        plt.title(f"Predicted: {predicted_label}, True: {true_label}")
        plt.show()

        # 압력 센서값 플로팅
        plt.plot(range(1, 5), pressures[i], marker='o', linestyle='-', color='b')
        plt.title("FSR Values")
        plt.show()
else:
    print("에러: 분리할 데이터가 부족합니다.")
