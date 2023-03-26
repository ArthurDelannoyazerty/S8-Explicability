from tensorflow import keras
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import matplotlib.pyplot as plt
import numpy as np
import generate_model

model = keras.models.load_model('saliency\mnist_models')

# Find the index of the to be visualized layer above
# Helps find the layer of the soft max in the architecture
layer_index = utils.find_layer_idx(model, 'visualized_layer')

# Swap softmax with linear
# Uses backpropagation : compute how the output changes with respect to a change in input
# Softmax causes pb so this linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

# Numbers to visualize, specify some samples
indices_to_visualize = [ 0, 12, 38, 83, 112, 74, 190 ]

# Visualize
for index_to_visualize in indices_to_visualize:
    # Get input
    input_image = generate_model.x_test[index_to_visualize]
    input_class = np.argmax(generate_model.y_test[index_to_visualize])
    # Matplotlib preparations
    fig, axes = plt.subplots(1, 2)
    # Generate visualization
    visualization = visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)
    axes[0].imshow(input_image[..., 0])
    axes[0].set_title('Original image')
    axes[1].imshow(visualization)
    axes[1].set_title('Saliency map')
    fig.suptitle(f'MNIST target = {input_class}')
    plt.show()