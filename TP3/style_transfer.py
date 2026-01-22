import tensorflow as tf
from tensorflow import keras

# TODO complété : Charger le modèle VGG16 pré-entraîné sur ImageNet
# 'include_top=False' signifie qu'on ne charge pas les couches de classification finales.
# On ne veut pas classifier, on veut les couches intermédiaires qui extraient les caractéristiques.
# 'weights='imagenet'' charge les poids appris sur le jeu de données ImageNet.
vgg = keras.applications.VGG16(include_top=False, weights='imagenet')

# TODO complété : Geler les poids du modèle VGG16
# C'est crucial. On utilise VGG comme un extracteur de caractéristiques fixe.
# Nous n'allons pas l'entraîner, nous allons optimiser les pixels de notre image générée.
vgg.trainable = False

print("Modèle VGG16 chargé et gelé avec succès.")
vgg.summary()

# Définition des couches pour l'extraction de contenu et de style
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 
                'block2_conv1', 
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']