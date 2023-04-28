import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.utils import load_img
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

#image_path = tf.keras.utils.get_file('goldfish.jpg','https://github.com/ArthurDelannoyazerty/S8-Explicability/blob/376eefa50230c8a614f3bc361418dc82318c7d57/adversarial/images/input/goldfish.jpg')
#image_raw = tf.io.read_file(image_path)
image_raw = load_img('images/input/soldiers.jpg', target_size=(224,224))
#image = tf.image.decode_image(image_raw)


image = preprocess(image_raw)
image_probs = pretrained_model.predict(image)

plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

# Get the input label of the image.
image_index = 413 #goldfish = 1; brown bear = 294; assault rifle = 413
label = tf.one_hot(image_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

plt.figure()
perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
plt.show()

def display_images(image, description):

  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  im = np.array(image[0]*0.5 + 0.5)
  plt.imsave('images/image tests/soldiersFGSM.png', im)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                 label, confidence*100))
  #plt.savefig('images/input/soldierattack.jpg')
  plt.show()


epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

#for i, eps in enumerate(epsilons):
adv_x = image + epsilons[3]*perturbations
adv_x = tf.clip_by_value(adv_x, -1, 1)
display_images(adv_x, descriptions[3])
