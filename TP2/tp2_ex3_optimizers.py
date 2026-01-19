import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow

# -----------------------------------------------------
# 1. Chargement et préparation des données
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
# 2. Fonction pour créer le modèle
# -----------------------------------------------------
def create_model():
    """Crée et retourne un modèle Keras non compilé."""
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# -----------------------------------------------------
# 3. Définition des optimiseurs à tester
# -----------------------------------------------------
optimizers = {
    'SGD_with_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': keras.optimizers.RMSprop(),
    'Adam': keras.optimizers.Adam(),
}

# -----------------------------------------------------
# 4. Boucle d'expérimentation avec MLflow
# -----------------------------------------------------
# CORRECTION : Définir un nom d'expérience explicite pour la robustesse.
# Toutes les exécutions de ce script iront dans ce "dossier" MLflow.
mlflow.set_experiment("Optimizer Comparison")

print("\n--- DEBUT DE LA COMPARAISON DES OPTIMISEURS ---")
EPOCHS = 10

for opt_name, optimizer_instance in optimizers.items():
    print(f"\n--- Entrainement avec l'optimiseur : {opt_name} ---")
    
    # Chaque passage dans la boucle est une "Run" dans l'expérience MLflow.
    with mlflow.start_run(run_name=f"Optimizer_{opt_name}"):
        
        # --- Création et compilation du modèle ---
        model = create_model()
        model.compile(optimizer=optimizer_instance,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # --- Entraînement ---
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=2 # Mode d'affichage concis
        )
        
        # --- Évaluation sur le jeu de test ---
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # --- Enregistrement des résultats dans MLflow ---
        print(f"Resultat pour {opt_name}: Test Accuracy = {test_acc:.4f}")
        
        # Enregistrement des paramètres
        mlflow.log_param("optimizer", opt_name)
        mlflow.log_param("epochs", EPOCHS)
        
        # Enregistrement des métriques finales
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_loss", test_loss)
        
        # Enregistrement de l'historique complet pour chaque métrique, époque par époque
        for epoch in range(EPOCHS):
            mlflow.log_metric("training_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("validation_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("training_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("validation_loss", history.history['val_loss'][epoch], step=epoch)

print("\n--- COMPARAISON TERMINEE ---")
print("Lancez 'mlflow ui' pour visualiser les resultats.")