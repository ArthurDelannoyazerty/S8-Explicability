
from keras.applications.mobilenet_v2 import MobileNetV2 as ModelMobileNetV2

pretrained_model = ModelMobileNetV2(include_top=True, weights='imagenet')
pretrained_model.summary()
print("----------------------------------------------------------------------------")

from keras.applications.vgg16 import VGG16 as ModelVGG16

model = ModelVGG16(weights='imagenet', include_top=True)
model.summary()