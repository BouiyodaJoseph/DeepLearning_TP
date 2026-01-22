import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARTIE 1 : PRÉPARATION DES DONNÉES CIFAR-10
# ============================================================================

# 1. Charger le jeu de données CIFAR-10
# Il contient 60 000 images couleur 32x32 dans 10 classes
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 2. Définir les constantes
NUM_CLASSES = 10
# La forme d'entrée est (32, 32, 3) pour 32x32 pixels et 3 canaux de couleur (RGB)
INPUT_SHAPE = x_train.shape[1:] 

# 3. Normaliser les valeurs des pixels dans l'intervalle [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 4. Convertir les étiquettes en format One-Hot Encoding
# Les étiquettes initiales sont des entiers (0, 1, ..., 9).
# "to_categorical" les transforme en vecteurs. Ex: 3 -> [0,0,0,1,0,0,0,0,0,0]
# C'est nécessaire pour la fonction de perte 'categorical_crossentropy'.
y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

# Affichage pour vérification
print("--- Préparation des données terminée ---")
print(f"Forme des données d'entrée (Input data shape): {INPUT_SHAPE}")
print(f"Forme des données d'entraînement (x_train shape): {x_train.shape}")
# TODO complété : Afficher la forme des étiquettes après conversion
print(f"Forme des étiquettes d'entraînement (y_train shape): {y_train_cat.shape}")
print("--------------------------------------\n")

# ============================================================================
# PARTIE 2 / EXERCICE 1 : IMPLÉMENTATION D'UN CNN CLASSIQUE
# ============================================================================

def build_basic_cnn(input_shape, num_classes):
    """Construit un modèle CNN simple."""
    model = keras.Sequential([
        # Couche de Convolution 1 : 32 filtres de taille 3x3.
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        
        # Couche de Pooling 1 : Max Pooling avec une fenêtre de 2x2.
        # TODO complété : Ajouter la couche MaxPooling2D
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Couche de Convolution 2 : 64 filtres pour apprendre des motifs plus complexes.
        # TODO complété : Ajouter la seconde couche Conv2D
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        
        # Couche de Pooling 2
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Couche Flatten pour passer des cartes de caractéristiques 2D à un vecteur 1D
        keras.layers.Flatten(),
        
        # Couche Dense 1 : Une couche entièrement connectée classique.
        keras.layers.Dense(512, activation='relu'),
        
        # Couche de Sortie : 10 neurones (un par classe) avec activation softmax.
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Construire et compiler le modèle
print("--- Construction du modèle CNN de base ---")
model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()
print("--------------------------------------\n")

# Entraîner le modèle
print("--- Entraînement du modèle CNN de base ---")
history = model.fit(
    x_train, y_train_cat,
    batch_size=64,
    epochs=10,
    validation_split=0.1 # Utilise 10% des données d'entraînement pour la validation
)
print("--------------------------------------\n")

# Évaluer le modèle sur l'ensemble de test
print("--- Évaluation du modèle CNN de base ---")
# TODO complété : Évaluer le modèle sur x_test, y_test
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"\nPrécision sur l'ensemble de test : {test_acc:.4f}")
print("--------------------------------------\n")


# ============================================================================
# PARTIE 2 / EXERCICE 2 : BLOCS RÉSIDUELS (RESNETS)
# ============================================================================

# --- Définition de la fonction du bloc résiduel (VERSION CORRIGÉE) ---
def residual_block(x, filters, stride=1):
    """Construit un bloc résiduel robuste."""
    # Chemin du raccourci (shortcut)
    shortcut = x

    # Chemin principal (main path)
    y = keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)

    y = keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)

    # Si le stride est > 1 (réduction de dimension) ou si le nombre de filtres change,
    # on doit ajuster le raccourci pour que les dimensions correspondent à l'addition.
    if stride != 1 or x.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    # TODO complété : Addition du chemin principal et du raccourci
    z = keras.layers.Add()([shortcut, y])
    
    # L'activation finale se fait APRES l'addition
    z = keras.layers.Activation('relu')(z)
    return z

# --- Construction du modèle Mini-ResNet ---
def build_mini_resnet(input_shape, num_classes):
    """Construit une petite architecture ResNet."""
    # TODO complété : Construire une petite architecture avec 3 blocs résiduels
    
    inputs = keras.layers.Input(shape=input_shape)
    
    # Une couche de convolution initiale
    x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    # Bloc 1 (sans changement de dimension)
    x = residual_block(x, filters=32, stride=1)
    
    # Bloc 2 (réduit la dimension de 32x32 à 16x16 et augmente les filtres à 64)
    x = residual_block(x, filters=64, stride=2)
    
    # Bloc 3 (sans changement de dimension)
    x = residual_block(x, filters=64, stride=1)
    
    # Couches de classification finales
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Construire et compiler le modèle ResNet
print("\n--- Construction du modèle Mini-ResNet ---")
model_resnet = build_mini_resnet(INPUT_SHAPE, NUM_CLASSES)
model_resnet.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])

model_resnet.summary()
print("--------------------------------------\n")

# Entraîner le modèle ResNet
print("--- Entraînement du modèle Mini-ResNet ---")
history_resnet = model_resnet.fit(
    x_train, y_train_cat,
    batch_size=64,
    epochs=10,
    validation_split=0.1
)
print("--------------------------------------\n")

# Évaluer le modèle ResNet sur l'ensemble de test
print("--- Évaluation du modèle Mini-ResNet ---")
test_loss_resnet, test_acc_resnet = model_resnet.evaluate(x_test, y_test_cat, verbose=2)
print(f"\nPrécision du ResNet sur l'ensemble de test : {test_acc_resnet:.4f}")
print("--------------------------------------\n")

# --- Comparaison des modèles ---
print("--- COMPARAISON FINALE ---")
print(f"Précision du CNN de base : {test_acc:.4f}")
print(f"Précision du Mini-ResNet : {test_acc_resnet:.4f}")