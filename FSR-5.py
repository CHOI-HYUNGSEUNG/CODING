import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, concatenate
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#이미지 파일 경로 지정
images_directory = r'C:\Users\user\Desktop\example'

#이미지 로드 및 레이블 부여 함수
def load_images_and_labels(folder,label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder,filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img,(256,256))
            images.append(img)
            labels.append(label)
    return images, labels

#이미지 로드
posture_1 = os.path.join(images_directory, "normal") #정자세 이미지
posture_2 = os.path.join(images_directory, "front") #앞으로 숙인 이미지
posture_3 = os.path.join(images_directory, "back") #뒤로 누워있는 이미지

#이미지 라벨링
posture1_images, posture1_labels = load_images_and_labels(posture_1, label = 0) 
posture2_images, posture2_labels = load_images_and_labels(posture_2, label = 1)
posture3_images, posture3_labels = load_images_and_labels(posture_3, label = 2)

#데이터 전처리 및 병합
all_images = np.concatenate([posture1_images,posture2_images,posture3_images])
all_labels = np.concatenate([posture1_labels,posture2_labels,posture3_labels])
all_images = np.expand_dims(all_images, axis=-1)
all_images = all_images/255.0

#모델 정의(CNN모델)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (256,256,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))

#데이터 증강
data_agumentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2)
])

#데이터셋 분할
train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# 레이블을 원-핫 인코딩
train_labels_one_hot = to_categorical(train_labels, num_classes=3)
test_labels_one_hot = to_categorical(test_labels, num_classes=3)

#모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels_one_hot, epochs = 10, validation_data = (test_images, test_labels_one_hot))

#모델 평가
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch') 
plt.ylim([0.5,1])
plt.legend(loc = 'lower right')
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot, verbose=2) 