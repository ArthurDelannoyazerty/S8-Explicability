import os
from PIL import Image, ImageDraw, ImageFont

# Chemin des dossiers source contenant les images
source_dir1 = "saliency\images\output\VGG16\\basic_images"
source_dir2 = "saliency\images\output\VGG16\FGSM_MobileNetV2"
source_dir3 = "saliency\images\output\VGG16\PGD_MobileNetV2"

# Chemin du dossier de destination pour l'image combinée
destination_dir = "saliency\images\output\VGG16\\result_concat"

# Récupération de la liste des fichiers images dans chaque dossier source
files1 = os.listdir(source_dir1)
files2 = os.listdir(source_dir2)
files3 = os.listdir(source_dir3)

# Création d'une liste contenant le chemin complet de chaque image dans chaque dossier source
image_paths1 = [os.path.join(source_dir1, f) for f in files1 if f.endswith('.jpg') or f.endswith('.png')]
image_paths2 = [os.path.join(source_dir2, f) for f in files2 if f.endswith('.jpg') or f.endswith('.png')]
image_paths3 = [os.path.join(source_dir3, f) for f in files3 if f.endswith('.jpg') or f.endswith('.png')]

# Vérification que le nombre d'images dans chaque dossier est le même
if len(image_paths1) != len(image_paths2) or len(image_paths1) != len(image_paths3):
    print("Erreur: le nombre d'images dans chaque dossier est différent.")
    exit()

# Boucle sur chaque i-ème image dans les trois dossiers source
for i in range(len(image_paths1)):
    # Chargement des trois images correspondantes
    image1 = Image.open(image_paths1[i])
    image2 = Image.open(image_paths2[i])
    image3 = Image.open(image_paths3[i])

    # Taille des images
    width, height = image1.size

    # Vérification que les trois images ont la même taille
    if image1.size != image2.size or image1.size != image3.size:
        print("Erreur: les images n'ont pas la même taille.")
        exit()

    # Création de l'image de sortie en les combinant verticalement
    output_image = Image.new('RGB', (image1.width, image1.height*3))
    output_image.paste(image1, (0, 0))
    output_image.paste(image2, (0, image1.height))
    output_image.paste(image3, (0, image1.height*2))

    # Ajout des labels pour chaque ligne
    draw = ImageDraw.Draw(output_image)
    font = ImageFont.truetype('arial.ttf', 24)
    draw.text((10, 5), "Original", font=font, fill=(0, 0, 0))
    draw.text((10, height+5), "FGSM", font=font, fill=(0, 0, 0))
    draw.text((10, height*2+5), "PGD", font=font, fill=(0, 0, 0))

    # Enregistrement de l'image concaténée dans le dossier de destination
    output_image.save(os.path.join(destination_dir, f'concatenated_image_{i+1}.jpg'))



