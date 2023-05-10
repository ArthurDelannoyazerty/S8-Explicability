import cv2
import os
import matplotlib.pyplot as plt

path_project_root = os.path.dirname(os.path.abspath(''))
path_not_formatted = os.path.join(path_project_root, "S8-Explicability\\attack_and_saliency\images\input\\not_formatted")

for filename in os.listdir(path_not_formatted):
    f = os.path.join(path_not_formatted, filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = cv2.imread(f)
        width = 224
        height = 224
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        plt.imsave(os.path.join(path_project_root, "S8-Explicability\\attack_and_saliency\images\input"+"\\"+filename,), resized)



