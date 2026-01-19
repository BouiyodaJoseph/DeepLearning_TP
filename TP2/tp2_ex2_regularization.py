import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers # Import nécessaire pour la régularisation L2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Chargement et préparation des données (inchangé)
# -----------------------------------------------------
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

x_val = x_train_full[54000:]
y_val = y_train_full[54000:]
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(54000, 784)
x_val = x_val.reshape(6000, 784)
x_test = x_test.reshape(10000, 784)

# -----------------------------------------------------
# 4. Construction du modele AVEC REGULARISATION
# -----------------------------------------------------
print("\n--- CONSTRUCTION DU MODELE REGULARISE ---")

# La syntaxe avec Input() est une bonne pratique, comme suggere par les warnings.
# Cela rend la structure du modele plus explicite.
model = keras.Sequential([
    # Definition explicite de la couche d'entree
    keras.layers.Input(shape=(784,)), 
    
    # Premiere couche Dense avec regularisation L2 sur les poids du noyau (kernel)
    # L2(0.001) ajoute une penalite a la fonction de perte pour les poids trop grands.
    keras.layers.Dense(512, activation='relu', 
                       kernel_regularizer=regularizers.l2(0.001)),
    
    # Couche de Dropout placee apres la couche d'activation
    # Elle "eteint" aleatoirement 50% des neurones pour cet exercice
    # afin de bien voir l'effet de la regularisation.
    keras.layers.Dropout(0.5), # Augmentation du Dropout a 0.5 pour un effet plus visible
    
    # Couche de sortie
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

# -----------------------------------------------------
# 5. Compilation du modele (inchangé)
# -----------------------------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------
# 6. Entrainement du modele
# -----------------------------------------------------
print("\n--- DEBUT DE L'ENTRAINEMENT DU MODELE REGULARISE ---")
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_val, y_val)
)
print("--- FIN DE L'ENTRAINEMENT ---\n")

# -----------------------------------------------------
# 7. Evaluation sur l'ensemble de test
# -----------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Performance finale sur l'ensemble de test (modele regularise) :")
print(f"  - Perte (Loss) : {test_loss:.4f}")
print(f"  - Precision (Accuracy) : {test_acc:.4f}")

# -----------------------------------------------------
# 8. Analyse des resultats
# -----------------------------------------------------
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
ecart = final_train_acc - final_val_acc

print("\n--- DIAGNOSTIC DU MODELE REGULARISE ---")
print(f"Precision finale sur l'entrainement : {final_train_acc:.4f}")
print(f"Precision finale sur la validation : {final_val_acc:.4f}")
print(f"Ecart entre les deux : {ecart:.4f}")

# -----------------------------------------------------
# 9. Visualisation des courbes d'apprentissage
# -----------------------------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precision Entrainement (Regularise)')
plt.plot(history.history['val_accuracy'], label='Precision Validation (Regularise)')
plt.title('Courbes de Precision')
plt.xlabel('Epoques')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte Entrainement (Regularise)')
plt.plot(history.history['val_loss'], label='Perte Validation (Regularise)')
plt.title('Courbes de Perte')
plt.xlabel('Epoques')
plt.ylabel('Perte')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()