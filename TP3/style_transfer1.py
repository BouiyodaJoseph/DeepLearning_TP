import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. FONCTIONS UTILITAIRES POUR LES IMAGES
# ------------------------------------------------------------------

def load_and_process_image(path_to_img, target_size=(512, 512)):
    """Charge une image, la redimensionne et la prepare pour VGG16."""
    # Charger l'image depuis le chemin du fichier
    img = Image.open(path_to_img)
    
    # Redimensionner l'image
    img = img.resize(target_size)
    
    # Convertir l'image en tableau NumPy
    img = keras.preprocessing.image.img_to_array(img)
    
    # Ajouter une dimension de batch (pour correspondre au format d'entree de VGG)
    img = np.expand_dims(img, axis=0)
    
    # Pre-traiter l'image avec la fonction specifique a VGG16
    # (convertit les couleurs RGB en BGR, centre les pixels, etc.)
    img = keras.applications.vgg16.preprocess_input(img)
    
    return img

def deprocess_image(processed_img):
    """Reconvertit une image pre-traitee en une image affichable."""
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    
    # Annuler le centrage des pixels
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # Inverser la conversion RGB -> BGR
    x = x[:, :, ::-1]
    
    # S'assurer que les valeurs sont dans l'intervalle [0, 255]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# ------------------------------------------------------------------
# 2. CHARGEMENT ET PRÉ-TRAITEMENT DES IMAGES
# ------------------------------------------------------------------
# TODO complété : Charger et pré-traiter les images

# !!! IMPORTANT : Assurez-vous d'avoir des fichiers 'content.jpg' et 'style.jpg'
# dans le meme dossier que ce script.
content_path = 'content.jpg'
style_path = 'style.jpg'

try:
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    print("Images de contenu et de style chargees avec succes.")

    # Afficher les images pour verification (optionnel)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(deprocess_image(content_image))
    plt.title("Image de Contenu")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(deprocess_image(style_image))
    plt.title("Image de Style")
    plt.axis('off')
    plt.show()

except FileNotFoundError:
    print("\nERREUR: Assurez-vous que les fichiers 'content.jpg' et 'style.jpg' existent.")
    print("Veuillez placer vos images dans le meme dossier que ce script et relancer.")
    exit() # Quitte le script si les images ne sont pas trouvees

# ------------------------------------------------------------------
# 3. CHARGEMENT DU MODÈLE VGG16
# ------------------------------------------------------------------
# TODO complété : Charger le modèle VGG16 pré-entraîné
vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False # Important: on ne re-entraine pas VGG

print("\nModele VGG16 charge et gele.")
vgg.summary()

# ------------------------------------------------------------------
# 4. CRÉATION DE L'EXTRACTEUR DE CARACTÉRISTIQUES
# ------------------------------------------------------------------
# Définition des couches pour l'extraction de contenu et de style
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 
                'block2_conv1', 
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def create_extractor(model, style_layers, content_layers):
    """
    Crée un modèle qui retourne les activations des couches de style et de contenu.
    """
    model.trainable = False # S'assurer a nouveau que le modele est gele
    
    # Obtenir les sorties de toutes les couches necessaires
    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    
    # Creer le nouveau modele
    extractor = keras.Model(inputs=model.input, outputs=outputs)
    return extractor

# Créer l'instance de l'extracteur
extractor = create_extractor(vgg, style_layers, content_layers)
print("\nExtracteur de caracteristiques cree.")

# ------------------------------------------------------------------
# FIN DE LA PARTIE PRATIQUE DU TP
# ------------------------------------------------------------------
print("\n--- Preparation pour le transfert de style terminee ---")
print("Les prochaines etapes (TODO du TP) consisteraient a definir les fonctions de perte")
print("et a lancer une boucle d'optimisation personnalisee sur les pixels d'une image cible.")