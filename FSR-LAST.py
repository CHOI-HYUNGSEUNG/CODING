import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import serial

#데이터 경로
data_directory = r'C:\Users\user\Desktop\example\data'

#데이터 로드 함수
def load_data(directory):
    data = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if file.endswith(".txt"):
                    with open(file_path, 'r') as f:
                        fsr_data = np.loadtxt(f)
                        data.append(fsr_data)
                        labels.append(int(label))
    return np.array(data), np.array(labels)

#데이터 로드
data, labels = load_data(data_directory)

#Sequential 모델 설정
model = tf.keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(3, activation = 'softmax')
])

#데이터 분할
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)

#모델 컴파일
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#모델 학습
model.fit(train_data, train_labels, epochs = 10, validation_data=(val_data, val_labels))
history = model.fit(data, labels, epochs = 5, validation_split = 0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(data, labels, verbose=2)
print('Test accuracy: ', test_acc)

new_data = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]])

#모델 예측
predictions = model.predict(new_data)

#시리얼 통신
arduino_port = "COM5"
baud_rate = 9600

#시리얼 포트 열기
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

#모델 저장 및 불러오기
model.save('my_model.h5')
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')

#데이터 수신 함수
def receive_data_from_arduino():
    return ser.readline().decode().strip()

#데이터 송신 함수
def send_data_to_arduino(data):
    ser.write(data.encode())

while True:
    input_data = receive_data_from_arduino()
    prediction = model.predict(np.array([input_data]))[0]
    print("예측 결과: ",prediction)