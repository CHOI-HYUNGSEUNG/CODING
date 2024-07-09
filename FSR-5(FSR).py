import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#데이터 경로
data_directory = r'C:\Users\user\Desktop\example\data'

#데이터 로드 및 전처리 함수
def load_data(data_directory):
    fsr_data = []
    labels = []
    for posture_folder in os.listdir(data_directory):
        posture_directory = os.path.join(data_directory, posture_folder)
        if os.path.isdir(posture_directory):
            posture_label = int(posture_folder) #폴더 이름을 자세 레이블로 이용
            for file in os.listdir(posture_directory):
                file_path = os.path.join(posture_directory, file)
                with open(file_path, 'r') as f:
                    data = [float(line.strip()) for line in f.readlines()]
                    fsr_data.append(data)
                    labels.append(posture_label)
    return np.array(fsr_data), np.array(labels)

#데이터 로드
fsr_data, labels = load_data(data_directory)

#모델 구성
model = Sequential([
    Dense(64, activation = 'relu', input_shape =(4,)),
    Dense(64, activation = 'relu'),
    Dense(3, activation = 'softmax')
])

#모델 컴파일
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(fsr_data, labels, epochs = 5, validation_split = 0.2)

#모델 평가
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch') 
plt.ylim([0.5,1])
plt.legend(loc = 'lower right')
test_loss, test_acc = model.evaluate(fsr_data, labels, verbose=2)

#모델 학습
model.fit(fsr_data, labels, epochs = 5, batch_size = 32, validation_split = 0.2)