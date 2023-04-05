print("Executing : Load libraries")
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus
import numpy as np


#---------------------------------------------------------------------------------------------------
print("Executing : Load Model")
data = tf.keras.datasets.mnist

## Constitution des sous-jeux
(x_train,y_train), (x_test,y_test)=data.load_data()

x_train_norm = x_train / 255
x_test_norm = x_test / 255

#---------------------------------------------------------------------------------------------------
img1 = x_test_norm[1]


#---------------------------------------------------------------------------------------------------
print("Executing : creating mnist model")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(64, activation='sigmoid', kernel_initializer = tf.keras.initializers.he_normal() ))
model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer = tf.keras.initializers.glorot_normal()))

model.compile(loss='sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate=0.001), metrics=["accuracy"])
model.fit(x_train,y_train,epochs=10,validation_split=0.3)

#---------------------------------------------------------------------------------------------------
print("Executing : Modifying model")

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
replace2linear = ReplaceToLinear()

print("Executing : Creating score function")

from tf_keras_vis.utils.scores import CategoricalScore

score = CategoricalScore([y_test[1]])

#---------------------------------------------------------------------------------------------------
print("Executing : Classic Saliency")

from keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils import normalize

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True
)

other_score = y_test[1]

# Generate saliency map
saliency_map = saliency(score, img1)

plt.imshow(saliency_map, cmap='jet')
plt.savefig('saliency/images/output/saliency_classic_mnist.png')
plt.title("Classic saliency")
plt.show()