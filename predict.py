import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
IMG_HEIGHT = 200
IMG_WIDTH = 200

model = tf.keras.models.load_model('saved_model/my_model_vgg16')
lables = ['cat', 'dog']

# Check its architecture
model.summary()

predict_image_path = 'test'

plt.figure(figsize=(10,10))
i = 1
for file_name in os.listdir(predict_image_path)[:40]:
     if file_name.endswith(".jpg"):
        plt.subplot(4, 10, i)
        img = tf.keras.preprocessing.image.load_img(os.path.join(predict_image_path, file_name), target_size=[IMG_HEIGHT, IMG_WIDTH])
        image_numpy_array = tf.keras.preprocessing.image.img_to_array(img)
        image_numpy_array = np.reshape(image_numpy_array, [1, IMG_HEIGHT, IMG_WIDTH, 3])
        pred = model.predict(image_numpy_array)
        plt.grid(False)
        plt.imshow(img)
        plt.xlabel(lables[int(pred[0][0])])
        i = i + 1
     else:
         continue

plt.show()