import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import foolbox as fb
from keras.utils import load_img, img_to_array, array_to_img
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from PIL import Image, ImageChops, ImageEnhance

def print_prediction(predictions, best_k = 5):
  labels = decode_predictions(predictions)
  kind = labels[0][0][1].replace("_", " ").title()
  percent = round(labels[0][0][2] * 100, 2)
  print(f"This is a {kind}. I am {percent} % sure.")
  print()
  print("Other suggestions:")
  for i in range(best_k-1):
    kind = labels[0][i+1][1].replace("_", " ").title()
    percent = round(labels[0][i+1][2] * 100, 2)
    print(f"{kind}: {percent} %")

model = ResNet50()

orig_img = load_img("images/input/soldiers.jpg")
height = model.layers[0].input_shape[0][1]
width = model.layers[0].input_shape[0][2]
channels = model.layers[0].input_shape[0][3]

orig_img = orig_img.resize((width, int(orig_img.size[1] * width / orig_img.size[0])))

img_width, img_height = orig_img.size
left = (img_width - width)/2
top = (img_height - height)/2
right = (img_width + width)/2
bottom = (img_height + height)/2

orig_img = orig_img.crop((left, top, right, bottom))

plt.figure()
plt.imshow(np.array(orig_img))
plt.show()

orig_input = img_to_array(orig_img)
orig_input = orig_input.reshape((1, width, height, channels))
orig_input = preprocess_input(orig_input)

predictions = model.predict(orig_input)
print_prediction(predictions)

fmodel = fb.TensorFlowModel(model, bounds=(-255, 255))

from foolbox.criteria import TargetedMisclassification

PENGUIN_CLASS = 314 #goldfish = 1; brown bear = 294; assault rifle = 413
ADV_CLASS = 2 # 2 = great white shark
adv_label = tf.convert_to_tensor([ADV_CLASS])

criterion = TargetedMisclassification(adv_label)

from foolbox.attacks import PGD

attack = PGD()
input_as_tensor = tf.convert_to_tensor(orig_input)
adv_input = attack.run(fmodel, input_as_tensor, criterion, epsilon=10)

adv_img = (adv_input.numpy() + 255) / 2
adv_img = adv_img.reshape(width, height, channels)
adv_img = array_to_img(adv_img)
b, g, r = adv_img.split()
adv_img = Image.merge("RGB", (r, g, b))
plt.figure()
plt.imshow(np.array(adv_img))
plt.imsave('images/image tests/soldiersPGD.png', np.array(adv_img))
plt.show()

adv_input = img_to_array(adv_img)
adv_input = adv_input.reshape((1, width, height, channels))
adv_input = preprocess_input(adv_input)
predictions = model.predict(adv_input)
print_prediction(predictions)

difference = ImageChops.difference(adv_img, orig_img)
plt.figure()
plt.imshow(np.array(difference))
plt.show()

difference = ImageEnhance.Brightness(difference).enhance(10)
plt.figure()
plt.imshow(np.array(difference))
plt.show()
