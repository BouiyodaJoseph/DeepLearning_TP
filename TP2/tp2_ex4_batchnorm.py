import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow
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
# 2. Fonction pour créer le modèle AVEC BATCH NORMALIZATION
# -----------------------------------------------------
def create_model_with_bn():
    """Crée et retourne un modèle Keras non compilé avec Batch Normalization."""
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(512, activation='relu'),
        
        # AJOUT : La couche Batch Normalization est placee APRES la couche Dense
        # et AVANT la fonction d'activation, mais Keras gere cela pour nous
        # quand l'activation est specifiee dans la couche Dense. 
        # L'ordre classique est : Dense -> Batch Norm -> Activation -> Dropout
        keras.layers.BatchNormalization(),
        
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# -----------------------------------------------------
# 3. Entraînement et évaluation avec MLflow
# -----------------------------------------------------
# On cree une nouvelle experience pour isoler ce test.
mlflow.set_experiment("Batch Normalization Test")

print("\n--- DEBUT DE L'ENTRAINEMENT AVEC BATCH NORMALIZATION ---")
EPOCHS = 10

with mlflow.start_run(run_name="Model_With_BatchNorm"):
    
    # --- Création et compilation du modèle ---
    model = create_model_with_bn()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # --- Entraînement ---
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=2
    )
    
    # --- Évaluation ---
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # --- Enregistrement des résultats dans MLflow ---
    print(f"\nResultat du modele avec BatchNorm: Test Accuracy = {test_acc:.4f}")
    
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_normalization", "True")
    
    mlflow.log_metric("final_test_accuracy", test_acc)
    mlflow.log_metric("final_test_loss", test_loss)
    
    for epoch in range(EPOCHS):
        mlflow.log_metric("validation_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        mlflow.log_metric("validation_loss", history.history['val_loss'][epoch], step=epoch)

print("\n--- EXPERIMENTATION TERMINEE ---")
print("Lancez 'mlflow ui' pour visualiser les resultats.")

# -----------------------------------------------------
# 4. Visualisation locale des courbes d'apprentissage
# -----------------------------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precision Entrainement')
plt.plot(history.history['val_accuracy'], label='Precision Validation')
plt.title('Courbes de Precision (avec BatchNorm)')
plt.xlabel('Epoques')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte Entrainement')
plt.plot(history.history['val_loss'], label='Perte Validation')
plt.title('Courbes de Perte (avec BatchNorm)')
plt.xlabel('Epoques')
plt.ylabel('Perte')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()