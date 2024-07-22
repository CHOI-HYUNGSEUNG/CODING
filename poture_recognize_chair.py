import serial
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#시리얼 통신
arduino_port = "COM5"
baud_rate = 9600

#시리얼 포트 열기
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

#아두이노 센서값 읽기
def read_sensor_data():
    line = ser.readline().decode("UTF-8").strip()
    if line:
        s1,s2,s3,s4 = map(int,line.split(','))
        return np.array([s1,s2,s3,s4])
    else:
        return None

#데이터베이스 연동
conn = sqlite3.connect('posture.db')
c = conn.cursor()

#데이터베이스 데이터 저장
c.execute('SELECT sensor1,sensor2,sensor3,sensor4,posture FROM FSR')
data = c.fetchall()
conn.close()

#데이터셋 준비    
data = np.array(data)
X = data[:, :-1] #센서값
y = data[:, -1] #자세

#원-핫 코딩 -> 자세 번호를 변환
num_classes = len(np.unique(y))
y = tf.keras.utils.to_categorical(y, num_classes)

#모델 생성 
model = tf.keras.Sequential([
    layers.Input(shape = (4,)),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(num_classes, activation = 'softmax')
])

#모델 컴파일
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#모델 훈련
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)
model.fit(X_train,y_train,epochs=10, batch_size = 32, validation_data = (X_test,y_test))

#모델 평가
model.fit(X_train, y_train)
model.evaluate(X_test, y_test)

#모델 저장 및 불러오기
model.save('posture_recognization_model')
model.save_weights('posture_recognization_model_weights')
model.load_weights('posture_recognization_model_weights')

#시리얼 통신을 통한 센서값 로드 및 모델을 통한 자세 예측
while True:
    sensor_data = read_sensor_data()
    if sensor_data is not None:
        print(f"Sensor data: {sensor_data}")
        prediction = model.predict(sensor_data.reshape(1,-1))
        predicted_posture = np.argmax(prediction)
        print(f"Predicted posture num: {predicted_posture}")