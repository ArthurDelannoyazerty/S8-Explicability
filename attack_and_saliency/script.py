print("Executing : Load libraries")

import os
# uncomment to force the non use of GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

from keras.utils import load_img

from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.saliency import Saliency

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

output_folder = "attack_and_saliency/images/output/"


#---------------------------------------------------------------------------------------------------------
# Saliency

def load_model(model_name="MobileNetV2"):
    """
        Return the selected model. Argument : name of the model.
    """
    print("Executing : Load Model")
    model=""
    if model_name=="VGG16":
        from keras.applications.vgg16 import VGG16 as Model
        model = Model(weights='imagenet', include_top=True)
    elif model_name=="MobileNetV2":
        from keras.applications.mobilenet_v2 import MobileNetV2 as Model
        model = Model(weights='imagenet', include_top=True)
    else:
        print("Error : model not available")
        raise NameError()
    return model

def get_image(path, size=(224,224), model_name="MobileNetV2"):
    print("Executing : Load And Preprocess Image")
    image= load_img(path, target_size=size)
    image = tf.cast(image, tf.float32)
    if model_name=="VGG16":
        image = tf.keras.applications.vgg16.preprocess_input(image)
    elif model_name=="MobileNetV2":
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def get_score_function(index_list):
    print("Executing : Create Score Function")
    return CategoricalScore(index_list)

def get_saliency_object(model):
    print("Executing : Create Saliency Object")
    replace2linear = ReplaceToLinear()
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)
    return saliency

#---------------------------------------------------------------------------------------------------------
# attack

def get_attacked_image(image):
    #TODO 
    attacked_image = 0*image 
    return attacked_image


#---------------------------------------------------------------------------------------------------------
# metric
from metrics import *
# TO DO mettre les chemins des 2 cartes de sailance à étudier
# path_sal_map =
# path_attacked_sal_map = 


#---------------------------------------------------------------------------------------------------------
# Main Program

image_title = 'Goldfish'
index_model = [1]
path = "adversarial\images\output\PGD\goldfishPGD.png"

model = load_model("MobileNetV2")
image = get_image(path, model_name="MobileNetV2")
score = get_score_function(index_model)
saliency = get_saliency_object(model)

print("Executing : Calculating Saliency Image")
saliency_image = saliency(score, image)[0]
print("Executing : Calculating Smooth Saliency Image")
smooth_saliency_image = saliency(score, image, smooth_samples=20, smooth_noise=0.20)[0]


#attack image
attacked_image = get_attacked_image(image)



# saliency 
print("Executing : Calculating Saliency Image")
saliency_attacked_image = saliency(score, image)[0]
print("Executing : Calculating Smooth Saliency Image")
smooth_saliency_attacked_image = saliency(score, image, smooth_samples=20, smooth_noise=0.20)[0]


# difference metric
# diff_abs(path_sal_map,path_attacked_sal_map)
# diff_quadratique(path_sal_map,path_attacked_sal_map)
# coef_correlation(path_sal_map,path_attacked_sal_map)
# ssim_func(path_sal_map,path_attacked_sal_map)

