import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import generate_model as gm

model = keras.models.load_model('saliency\mnist_models')

x_example = gm.x_test[0] # prendre la première image de test pour cet exemple
x_example = x_example.reshape(1, 28, 28, 1)
x_example = x_example / 255.0 # normalisation des pixels entre 0 et 1

preds = model.predict(x_example)
class_idx = np.argmax(preds[0])

# Calcul de la carte de saillance
grads = tf.GradientTape().gradient(model.output[:, class_idx], model.input)[0]
grads = tf.reduce_mean(grads, axis=3)[0]
grads = np.maximum(grads, 0)
grads /= np.max(grads)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].imshow(x_example[0, :, :, 0], cmap='gray')
ax[1].imshow(x_example[0, :, :, 0], cmap='gray')
ax[1].imshow(grads, alpha=0.5, cmap='jet')
plt.show()
