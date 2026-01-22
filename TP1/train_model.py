# -----------------------------------------------------
# 1. Importation des bibliothèques nécessaires
# -----------------------------------------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np

# -----------------------------------------------------
# 2. Chargement du jeu de données MNIST
# -----------------------------------------------------
# Keras fournit des fonctions pratiques pour charger des jeux de données populaires.
# MNIST contient 70 000 images de chiffres manuscrits (de 0 à 9).
# 60 000 pour l'entraînement (x_train, y_train) et 10 000 pour le test (x_test, y_test).
# x_train contient les images, y_train contient les étiquettes (le chiffre correct).
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# -----------------------------------------------------
# 3. Préparation et normalisation des données
# -----------------------------------------------------
# Les images sont initialement des matrices de 28x28 pixels. Chaque pixel a une
# valeur entre 0 (noir) et 255 (blanc).

# On normalise les valeurs des pixels pour les ramener dans l'intervalle [0, 1].
# Cela aide l'algorithme d'optimisation à converger plus rapidement.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# On redimensionne (ou "aplatit") chaque image de 28x28 en un seul vecteur de 784 pixels.
# C'est nécessaire car notre premier type de couche (Dense) attend un vecteur en entrée.
# 60000 images d'entraînement, chacune de 784 pixels.
x_train = x_train.reshape(60000, 784)
# 10000 images de test, chacune de 784 pixels.
x_test = x_test.reshape(10000, 784)

# -----------------------------------------------------
# 4. Construction du modèle de réseau de neurones
# -----------------------------------------------------
# On utilise un modèle "Sequential", qui est une simple pile linéaire de couches.
model = keras.Sequential([
    # Première couche cachée : une couche "Dense" (entièrement connectée).
    # - 512 : le nombre de neurones dans cette couche.
    # - activation='relu' : la fonction d'activation "Rectified Linear Unit", très commune.
    # - input_shape=(784,) : spécifie la forme des données d'entrée (nos vecteurs de 784 pixels).
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    
    # Couche de Dropout : une technique de régularisation pour éviter le surapprentissage.
    # Elle "éteint" aléatoirement 20% des neurones pendant l'entraînement.
    keras.layers.Dropout(0.2),
    
    # Couche de sortie : une autre couche Dense.
    # - 10 : le nombre de neurones, un pour chaque classe (chiffres de 0 à 9).
    # - activation='softmax' : transforme les sorties en probabilités. La somme des 10
    #   probabilités sera égale à 1. Le neurone avec la plus haute probabilité
    #   correspondra au chiffre prédit.
    keras.layers.Dense(10, activation='softmax')
])

# -----------------------------------------------------
# 5. Compilation du modèle
# -----------------------------------------------------
# Avant l'entraînement, on doit configurer le processus d'apprentissage.
model.compile(
    # optimizer='adam' : L'algorithme utilisé pour mettre à jour les poids du réseau.
    # Adam est un optimiseur très efficace et populaire.
    optimizer='adam',
    
    # loss='sparse_categorical_crossentropy' : La fonction de coût que le modèle
    # essaiera de minimiser. Adaptée pour la classification multi-classe
    # quand les étiquettes sont des entiers (0, 1, 2...).
    loss='sparse_categorical_crossentropy',
    
    # metrics=['accuracy'] : La métrique que l'on veut suivre pendant l'entraînement.
    # Ici, on veut voir la précision (le pourcentage d'images correctement classées).
    metrics=['accuracy']
)

# -----------------------------------------------------
# 6. Entraînement du modèle
# -----------------------------------------------------
# C'est ici que l'apprentissage a lieu.
history = model.fit(
    x_train, y_train,        # Les données d'entraînement et leurs étiquettes
    epochs=5,                # Le nombre de fois que le modèle va voir tout le jeu de données
    batch_size=128,          # Le nombre d'images traitées avant chaque mise à jour des poids
    validation_split=0.1     # On met de côté 10% des données d'entraînement pour valider
                             # la performance du modèle à la fin de chaque "epoch".
)

# -----------------------------------------------------
# 7. Évaluation du modèle
# -----------------------------------------------------
# On évalue la performance finale du modèle sur le jeu de test,
# que le modèle n'a JAMAIS vu pendant l'entraînement.
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision sur les données de test: {test_acc:.4f}")

# -----------------------------------------------------
# 8. Sauvegarde du modèle entraîné
# -----------------------------------------------------
# On sauvegarde l'architecture du modèle, ses poids et sa configuration
# dans un seul fichier HDF5 (.h5).
model.save("mnist_model.h5")
print("Modèle sauvegardé sous mnist_model.h5")