import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt # Import pour la visualisation

# -----------------------------------------------------
# 1. Chargement des donnees
# -----------------------------------------------------
# On charge l'ensemble de donnees complet. x_train_full contient maintenant 60,000 images.
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

# -----------------------------------------------------
# 2. Creation des ensembles d'entrainement et de validation
# -----------------------------------------------------
# C'est une meilleure pratique que validation_split car le jeu de validation est fixe.
print(f"Taille initiale de l'ensemble d'entrainement : {len(x_train_full)}")

# On reserve les 6,000 dernieres images pour la validation (dev set).
x_val = x_train_full[54000:]
y_val = y_train_full[54000:]

# Les 54,000 premieres images constituent notre nouvel ensemble d'entrainement.
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

print(f"Taille du nouvel ensemble d'entrainement : {len(x_train)}")
print(f"Taille de l'ensemble de validation : {len(x_val)}")
print(f"Taille de l'ensemble de test : {len(x_test)}")

# -----------------------------------------------------
# 3. Normalisation et redimensionnement
# -----------------------------------------------------
# On applique les memes transformations aux 3 ensembles de donnees.
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(54000, 784)
x_val = x_val.reshape(6000, 784)
x_test = x_test.reshape(10000, 784)

# -----------------------------------------------------
# 4. Construction du modele (le meme qu'au TP1)
# -----------------------------------------------------
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Affichage du resume de l'architecture du modele
model.summary()

# -----------------------------------------------------
# 5. Compilation du modele
# -----------------------------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------
# 6. Entrainement du modele
# -----------------------------------------------------
print("\n--- DEBUT DE L'ENTRAINEMENT ---")
history = model.fit(
    x_train, y_train,
    epochs=15,  # J'ai augmente le nombre d'epoques a 15 pour mieux observer le surapprentissage
    batch_size=128,
    # On fournit notre ensemble de validation explicite ici
    validation_data=(x_val, y_val)
)
print("--- FIN DE L'ENTRAINEMENT ---\n")

# -----------------------------------------------------
# 7. Evaluation sur l'ensemble de test (performance finale)
# -----------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Performance finale sur l'ensemble de test :")
print(f"  - Perte (Loss) : {test_loss:.4f}")
print(f"  - Precision (Accuracy) : {test_acc:.4f}")

# -----------------------------------------------------
# 8. Analyse des resultats (Biais vs. Variance)
# -----------------------------------------------------
# Recuperation des metriques de la derniere epoque
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
ecart = final_train_acc - final_val_acc

print("\n--- DIAGNOSTIC BIAIS/VARIANCE ---")
print(f"Precision finale sur l'entrainement : {final_train_acc:.4f}")
print(f"Precision finale sur la validation : {final_val_acc:.4f}")
print(f"Ecart entre les deux : {ecart:.4f}")

if final_train_acc < 0.90:
    print("Diagnostic : BIAIS ELEVE. Le modele est trop simple et n'apprend pas assez.")
elif ecart > 0.02: # On considere un ecart de >2% comme significatif
    print("Diagnostic : VARIANCE ELEVEE (Overfitting). Le modele apprend trop bien l'entrainement et generalise mal.")
else:
    print("Diagnostic : Bon equilibre Biais/Variance.")

# -----------------------------------------------------
# 9. Visualisation des courbes d'apprentissage
# -----------------------------------------------------
# Creer un graphique pour la precision
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precision Entrainement')
plt.plot(history.history['val_accuracy'], label='Precision Validation')
plt.title('Courbes de Precision')
plt.xlabel('Epoques')
plt.ylabel('Precision')
plt.legend()

# Creer un graphique pour la perte (loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte Entrainement')
plt.plot(history.history['val_loss'], label='Perte Validation')
plt.title('Courbes de Perte')
plt.xlabel('Epoques')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()
plt.show()