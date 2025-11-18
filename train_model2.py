# -----------------------------------------------------
# 1. Importation des bibliothèques
# -----------------------------------------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow                  # <-- AJOUT : On importe MLflow
import mlflow.tensorflow       # <-- AJOUT : Module spécifique pour TensorFlow/Keras

# -----------------------------------------------------
# 2. Variables pour les paramètres de l'expérience
# -----------------------------------------------------
# Définir les paramètres ici les rend faciles à changer et à suivre.
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# -----------------------------------------------------
# 3. Chargement et préparation des données (inchangé)
# -----------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# -----------------------------------------------------
# 4. Lancement de la session de suivi MLflow
# -----------------------------------------------------
# Tout ce qui se trouve à l'intérieur de ce bloc "with" sera enregistré
# par MLflow comme une seule "exécution" ou "expérience".
with mlflow.start_run():
    print("Session MLflow démarrée.")
    
    # --- Enregistrement des paramètres ---
    # C'est comme écrire dans un cahier de labo : "Pour cette expérience,
    # j'ai utilisé les réglages suivants :"
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)

    # -----------------------------------------------------
    # 5. Construction du modèle
    # -----------------------------------------------------
    # On utilise maintenant les variables définies plus haut.
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE), # <-- Utilisation de la variable
        keras.layers.Dense(10, activation='softmax')
    ])

    # -----------------------------------------------------
    # 6. Compilation du modèle (inchangé)
    # -----------------------------------------------------
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # -----------------------------------------------------
    # 7. Entraînement du modèle
    # -----------------------------------------------------
    # On utilise aussi les variables ici.
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,             # <-- Utilisation de la variable
        batch_size=BATCH_SIZE,     # <-- Utilisation de la variable
        validation_split=0.1
    )

    # -----------------------------------------------------
    # 8. Évaluation du modèle (inchangé)
    # -----------------------------------------------------
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision sur les données de test: {test_acc:.4f}")

    # --- Enregistrement des métriques ---
    # On note les résultats obtenus : "Le résultat final était..."
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # -----------------------------------------------------
    # 9. Sauvegarde du modèle
    # -----------------------------------------------------
    # On garde la sauvegarde locale pour la suite du TP.
    model.save("mnist_model.h5")
    print("Modèle sauvegardé localement sous mnist_model.h5")

    # --- Enregistrement du modèle complet avec MLflow ---
    # On sauvegarde aussi le modèle comme un "artefact" dans MLflow.
    # Cela permet de retrouver facilement le modèle exact de cette expérience. pour lancer " http://127.0.0.1:5000" mlflow ui
    mlflow.keras.log_model(model, "mnist-model")
    print("Modèle enregistré dans MLflow.")

print("Script terminé.")