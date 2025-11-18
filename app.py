from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle Keras entraîné
# Assurez-vous que le fichier 'mnist_model.h5' est dans le même dossier
try:
    model = keras.models.load_model('mnist_model.h5')
    print("* Modèle chargé avec succès")
except Exception as e:
    print(f"* Erreur lors du chargement du modèle : {e}")
    model = None

# Définition de la route '/predict' pour les prédictions
# Cette fonction ne sera appelée que pour les requêtes de type POST
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Le modèle n\'a pas pu être chargé'}), 500
    
    # Récupération des données JSON envoyées dans la requête
    data = request.json
    
    # Vérification que la clé 'image' est bien présente dans les données
    if 'image' not in data:
        return jsonify({'error': 'Aucune image fournie'}), 400

    # Conversion des données de l'image en tableau NumPy
    image_data = np.array(data['image'])
    
    # Pré-traitement de l'image :
    # 1. Redimensionner en (1, 784) pour correspondre à l'entrée du modèle
    # 2. Normaliser les pixels entre 0 et 1
    image_data = image_data.reshape(1, 784)
    image_data = image_data.astype("float32") / 255.0

    # Prédiction avec le modèle
    prediction_probs = model.predict(image_data)
    
    # Trouver la classe avec la plus haute probabilité (le chiffre prédit)
    predicted_class = np.argmax(prediction_probs, axis=1)[0]

    # Renvoyer le résultat au format JSON
    return jsonify({
        'prediction': int(predicted_class),
        'probabilities': prediction_probs.tolist()
    })

# Point d'entrée pour démarrer le serveur Flask
if __name__ == '__main__':
    # '0.0.0.0' rend l'application accessible depuis l'extérieur du conteneur
    app.run(host='0.0.0.0', port=5000)