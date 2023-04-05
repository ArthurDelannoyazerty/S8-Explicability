print("Executing : Load libraries")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

output_folder = "saliency/images/output/FGSM_MobileNetV2/"

#---------------------------------------------------------------------------------------------------
print("Executing : Load Model")

from keras.applications.vgg16 import VGG16 as Model

model = Model(weights='imagenet', include_top=True)
#model.summary()

#---------------------------------------------------------------------------------------------------
print("Executing : Preprocess Images")

from keras.utils import load_img
from keras.applications.vgg16 import preprocess_input

# Image titles
image_titles = ['Goldfish', 'Bear', 'Assault rifle']

# Load images and Convert them to a Numpy array
img1 = load_img('saliency/images/input/goldfish.jpg', target_size=(224, 224))
img2 = load_img('saliency/images/input/bear.jpg', target_size=(224, 224))
img3 = load_img('saliency/images/input/soldiers.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data for VGG16
X = preprocess_input(images)

# Rendering
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
    plt.suptitle("Original images")
plt.tight_layout()
plt.savefig(output_folder + 'original_images.png')
plt.show()

#teste
print("test git")

#---------------------------------------------------------------------------------------------------
print("Executing : Modifying model")

#En linéarisant la dernière couche, nous supprimons les effets des fonctions 
#d'activation pour obtenir une sortie brute qui représente les activations de chaque neurone dans cette couche. 
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()

# Instead of using the ReplaceToLinear instance above,
# you can also define the function from scratch as follows:
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear


#---------------------------------------------------------------------------------------------------
print("Executing : Creating score function")

# CategoricalScore = accuracy
from tf_keras_vis.utils.scores import CategoricalScore

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score = CategoricalScore([1, 294, 413])

# Instead of using CategoricalScore object,
# you can also define the function from scratch as follows:
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
    return (output[0][1], output[1][294], output[2][413])


#---------------------------------------------------------------------------------------------------
print("Executing : Classic Saliency")

from keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils import normalize

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)

# Generate saliency map
saliency_map = saliency(score, X)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# saliency_map = normalize(saliency_map)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
    plt.suptitle("Classic saliency")
plt.tight_layout()
plt.savefig(output_folder + 'saliency_classic.png')
plt.show()

#---------------------------------------------------------------------------------------------------
print("Executing : SmoothGrad Saliency")

# Generate saliency map with smoothing that reduce noise by adding noise
saliency_map = saliency(score,
                        X,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20) # noise spread level.

## Since v0.6.0, calling `normalize()` is NOT necessary.
# saliency_map = normalize(saliency_map)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
    plt.suptitle("SmoothGrad Saliency")
plt.tight_layout()
plt.savefig(output_folder + 'saliency_smoothgrad.png')
plt.show()

#---------------------------------------------------------------------------------------------------
print("Executing : GradCAM")

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Create Gradcam object
gradcam = Gradcam(model,
                  model_modifier=replace2linear,
                  clone=True)

# Generate heatmap with GradCAM
cam = gradcam(score,
              X,
              penultimate_layer=-1)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
    ax[i].axis('off')
    plt.suptitle("GradCAM")
plt.tight_layout()
plt.savefig(output_folder + 'GradCAM.png')
plt.show()

#---------------------------------------------------------------------------------------------------
print("Executing : GradCAM++")

from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# Create GradCAM++ object
gradcam = GradcamPlusPlus(model,
                          model_modifier=replace2linear,
                          clone=True)

# Generate heatmap with GradCAM++
cam = gradcam(score,
              X,
              penultimate_layer=-1)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
    plt.suptitle("GradCAM++")
plt.tight_layout()
plt.savefig(output_folder + 'gradcam_plus_plus.png')
plt.show()

#---------------------------------------------------------------------------------------------------
print("Executing : ScoreCAM")

from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import num_of_gpus

# Create ScoreCAM object
scorecam = Scorecam(model)

# Generate heatmap with ScoreCAM
cam = scorecam(score, X, penultimate_layer=-1)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
    plt.suptitle("ScoreCAM")
plt.tight_layout()
plt.savefig(output_folder + 'ScoreCAM.png')
plt.show()

#---------------------------------------------------------------------------------------------------
print("Executing : Fast ScoreCAM")

from tf_keras_vis.scorecam import Scorecam

# Create ScoreCAM object
scorecam = Scorecam(model, model_modifier=replace2linear)

# Generate heatmap with Faster-ScoreCAM
cam = scorecam(score,
               X,
               penultimate_layer=-1,
               max_N=10)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
    plt.suptitle("Fast ScoreCAM")
plt.tight_layout()
plt.savefig(output_folder + 'fast_ScoreCAM.png')
plt.show()


