import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARTIE 0 : SIMULATION DES DONNÉES (pour rendre le code exécutable)
# ============================================================================
def create_simulated_data(num_images, img_size):
    """Crée un jeu de données simulé avec des cercles pour la segmentation."""
    print(f"Generation de {num_images} images simulees...")
    x = np.zeros((num_images, img_size, img_size, 1), dtype=np.float32)
    y = np.zeros((num_images, img_size, img_size, 1), dtype=np.float32)

    for i in range(num_images):
        # Créer une image avec un ou deux cercles aléatoires
        center_x, center_y = np.random.randint(20, img_size - 20, 2)
        radius = np.random.randint(10, 25)
        
        # Dessiner le cercle sur l'image et le masque
        rr, cc = np.ogrid[:img_size, :img_size]
        circle = (rr - center_y)**2 + (cc - center_x)**2 < radius**2
        
        # Le masque est un cercle parfait
        y[i, circle] = 1.0
        # L'image d'entree est le cercle avec du bruit
        x[i, circle] = 1.0
        x[i] += np.random.normal(0, 0.1, x[i].shape)
        x[i] = np.clip(x[i], 0, 1)

    print("Generation de donnees terminee.")
    return x, y

# ============================================================================
# PARTIE 2 / EXERCICE 2 : MÉTRIQUES DE SEGMENTATION SPÉCIFIQUES
# ============================================================================
# Ces fonctions sont nécessaires pour l'exercice 1
def dice_coeff(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice pour la segmentation binaire."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Perte de Dice."""
    return 1 - dice_coeff(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    """Métrique Intersection over Union (Jaccard)."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ============================================================================
# PARTIE 2 / EXERCICE 1 : IMPLÉMENTATION DE L'ARCHITECTURE U-NET
# ============================================================================

# Instruction 1: Créer la fonction pour le bloc convolutif de base
def conv_block(input_tensor, num_filters):
    """Définit un bloc de deux convolutions (Conv -> BatchNorm -> ReLU) x 2."""
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(128, 128, 1)):
    """Construit l'architecture U-Net complète."""
    inputs = keras.layers.Input(input_shape)

    # --- CHEMIN DE L'ENCODEUR (CONTRACTANT) ---
    # Instruction 2: Implémenter le chemin de l'encodeur
    c1 = conv_block(inputs, 8) #32
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    # TODO complété : 2 étapes de contraction supplémentaires
    c2 = conv_block(p1, 16) #64
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 32) #128
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    # --- PONT / BOTTLENECK ---
    b = conv_block(p3, 128) #256

    # --- CHEMIN DU DÉCODEUR (EXPANSIF) ---
    # Instruction 3 & 4: Implémenter le chemin du décodeur et les skip connections
    
    # Étape 1 : Sur-échantillonnage + Skip Connection avec c3
    u1 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    # TODO complété : Concaténer la sortie correspondante de l'encodeur (c3)
    u1 = keras.layers.Concatenate()([u1, c3])
    d1 = conv_block(u1, 32) #128
    
    # TODO complété : 2 étapes d'expansion supplémentaires
    # Étape 2 : Sur-échantillonnage + Skip Connection avec c2
    u2 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = keras.layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 16)  #64

    # Étape 3 : Sur-échantillonnage + Skip Connection avec c1
    u3 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = keras.layers.Concatenate()([u3, c1])
    d3 = conv_block(u3, 8) #32
    
    # --- COUCHE DE SORTIE ---
    # 1 filtre pour la segmentation binaire avec une activation sigmoïde
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# --- POINT D'ENTRÉE PRINCIPAL POUR L'EXÉCUTION ---
if __name__ == "__main__":
    IMG_SIZE = 128
    
    # Créer les données simulées pour l'entraînement et la validation
    x_train, y_train = create_simulated_data(500, IMG_SIZE)
    x_val, y_val = create_simulated_data(100, IMG_SIZE)

    # Construire le modèle U-Net
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1))

    # TODO complété : Compiler et entraîner le modèle
    # On utilise la perte de Dice (dice_loss) qui est plus adaptée à la segmentation
    # et on suit nos deux métriques personnalisées, dice_coeff et iou_metric.
    model.compile(optimizer='adam', 
                  loss=dice_loss, 
                  metrics=[dice_coeff, iou_metric])
    
    model.summary()
    
    print("\n--- DEBUT DE L'ENTRAINEMENT DU U-NET ---")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10, #15 Un peu plus d'époques pour bien converger
        batch_size=4 #16
    )

    # --- Visualisation des résultats ---
    print("\n--- VISUALISATION DES RESULTATS DE SEGMENTATION ---")
    # Choisir 3 images au hasard dans le jeu de validation pour les afficher
    samples_to_show = np.random.choice(len(x_val), 3, replace=False)
    
    for i in samples_to_show:
        # Prédire le masque pour une image de validation
        pred_mask = model.predict(x_val[i:i+1])[0]
        
        plt.figure(figsize=(15, 5))
        
        # Afficher l'image d'entrée bruitée
        plt.subplot(1, 3, 1)
        plt.imshow(x_val[i, :, :, 0], cmap='gray')
        plt.title("Image d'Entrée")
        plt.axis('off')
        
        # Afficher le masque réel (la vérité terrain)
        plt.subplot(1, 3, 2)
        plt.imshow(y_val[i, :, :, 0], cmap='gray')
        plt.title("Masque Réel (Vérité)")
        plt.axis('off')
        
        # Afficher le masque prédit par le modèle
        # On applique un seuil de 0.5 sur les probabilités en sortie
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask[:, :, 0] > 0.5, cmap='gray')
        plt.title("Masque Prédit par l'U-Net")
        plt.axis('off')
        
        plt.show()