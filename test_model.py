import tensorflow
from keras.utils import load_img
import numpy as np
import cv2

model = tensorflow.keras.models.load_model('saliency\mnist_models')
model.summary()


img1 = load_img('images/0/img_1.jpg', target_size=(28, 28))
input_arr = tensorflow.keras.utils.img_to_array(img1)

# Scale images to the [0, 1] range
input_arr = input_arr.astype("float32") / 255

print(input_arr.dtype)
print(input_arr.shape)

input_arr = cv2. cvtColor(input_arr, cv2.COLOR_BGR2GRAY)
input_arr = np.expand_dims(input_arr, -1)
print(input_arr.dtype)
print(input_arr.shape)

# input_arr = np.array([input_arr])[0]  # Convert single image to a batch.
print(input_arr.dtype)
print(input_arr.shape)

# input_arr = np.reshape(input_arr, (28, 28, 1))



y_prob = model.predict(input_arr) 
y_classes = y_prob.argmax(axis=-1)