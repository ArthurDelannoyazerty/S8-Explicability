from tensorflow import keras
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import matplotlib.pyplot as plt
import numpy as np


model = keras.models.load_model('saliency\mnist_models')

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'visualized_layer')

# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

# Numbers to visualize, specify some samples
indices_to_visualize = [ 0, 12, 38, 83, 112, 74, 190 ]






#todo saliency map