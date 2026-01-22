import requests
import numpy as np
from tensorflow import keras

# Charger le jeu de données MNIST pour avoir une image de test
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Sélectionner la première image du jeu de test (qui est un 7)
image_index = 0
test_image = x_test[image_index]
true_label = y_test[image_index]

print(f"Test avec l'image d'index {image_index}, qui est un '{true_label}'")

# Aplatir l'image de 28x28 en un vecteur de 784
# et la convertir en une liste simple pour l'envoyer en JSON
image_data_flat = test_image.flatten().tolist()

# URL de notre API qui tourne localement
url = 'http://localhost:5000/predict'

# Envoyer la requête POST avec l'image au format JSON
try:
    response = requests.post(url, json={'image': image_data_flat})
    response.raise_for_status()  # Lève une exception si la requête a échoué (ex: erreur 500)

    # Afficher la réponse du serveur
    result = response.json()
    print("\nRéponse de l'API :")
    print(f"  Prédiction    : {result['prediction']}")

    # Afficher les 3 probabilités les plus élevées pour plus de clarté
    probs = np.array(result['probabilities'][0])
    top3_indices = probs.argsort()[-3:][::-1]
    print("  Top 3 Probabilités :")
    for i in top3_indices:
        print(f"    - Chiffre {i}: {probs[i]:.4f}")

except requests.exceptions.RequestException as e:
    print(f"\nErreur lors de la connexion à l'API : {e}")