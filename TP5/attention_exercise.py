import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# EXERCICE 1 : Implémentation d'une Couche d'Attention Simple
# ============================================================================

class SimpleAttention(layers.Layer):
    """
    Couche d'attention simple (Bahdanau-style attention).
    Cette couche calcule un vecteur de contexte en donnant des poids différents
    à chaque pas de temps de la séquence d'entrée.
    """
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Crée les poids de la couche (W et b).
        input_shape: (batch_size, seq_len, hidden_dim)
        """
        # Poids pour calculer le score
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        # Biais pour calculer le score
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        """
        Définit la logique de la passe avant (forward pass).
        x: Tenseur de sortie du GRU, de forme (batch_size, seq_len, hidden_dim)
        """
        # --- CORRECTION : Remplacement de K.* par tf.* ---
        
        # 1. Calculer les scores d'attention
        # Forme de e : (batch_size, seq_len, 1)
        # On utilise tf.linalg.matmul pour le produit matriciel
        e = tf.keras.activations.tanh(tf.linalg.matmul(x, self.W) + self.b)
        
        # 2. 'Squeeze' pour enlever la dernière dimension et appliquer softmax
        # Forme de e_squeezed : (batch_size, seq_len)
        e_squeezed = tf.squeeze(e, axis=-1)
        # Forme de alignment_weights : (batch_size, seq_len)
        alignment_weights = tf.keras.activations.softmax(e_squeezed)
        
        # Pour le calcul du vecteur de contexte, on a besoin de remettre la dernière dimension
        # Forme de alpha_expanded : (batch_size, seq_len, 1)
        alpha_expanded = tf.expand_dims(alignment_weights, axis=-1)
        
        # 3. Calculer le vecteur de contexte
        # C'est la somme pondérée des états cachés de la séquence d'entrée.
        # Forme de context_vector : (batch_size, hidden_dim)
        context_vector = tf.reduce_sum(x * alpha_expanded, axis=1)
        
        return context_vector, alignment_weights

# --- Point d'entrée principal pour tester la couche ---
if __name__ == "__main__":

    # --- 1. Simulation de données NLP ---
    vocab_size = 1000
    max_len = 20
    num_samples = 1000

    # Générer des phrases aléatoires
    X_train = np.random.randint(1, vocab_size, size=(num_samples, max_len))
    # Le "sentiment" (0 ou 1) est basé sur la parité de la somme des indices des mots
    y_train = (np.sum(X_train, axis=1) % 2).astype(np.float32)

    # --- 2. Construction du modèle ---
    hidden_dim = 32
    inputs = layers.Input(shape=(max_len,))
    
    # Couche d'embedding pour transformer les mots en vecteurs denses
    embedding_layer = layers.Embedding(vocab_size, 16, input_length=max_len)(inputs)
    
    # Couche GRU qui retourne la séquence complète de ses états cachés
    gru_outputs = layers.GRU(hidden_dim, return_sequences=True)(embedding_layer)
    
    # Notre couche d'attention personnalisée
    context_vector, alignment_weights = SimpleAttention()(gru_outputs)
    
    # Couche de sortie pour la classification binaire
    outputs = layers.Dense(1, activation='sigmoid')(context_vector)
    
    # Définir le modèle complet
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # --- 3. Entraînement ---
    print("\n--- Debut de l'entrainement ---")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # --- 4. Visualisation de l'Attention ---
    # Créons un modèle séparé pour récupérer les poids d'attention
    attention_model = keras.Model(inputs=inputs, outputs=[outputs, alignment_weights])
    
    # Prenons une phrase de test au hasard
    test_index = np.random.randint(0, len(X_train))
    test_sentence = X_train[test_index:test_index+1]
    
    # Obtenir la prédiction et les poids
    prediction, att_weights = attention_model.predict(test_sentence)
    
    print(f"\nPhrase de test (indices de mots): \n{test_sentence[0]}")
    print(f"Prediction de sentiment (0 ou 1): {prediction[0][0]:.2f}")
    
    # Visualiser les poids
    plt.figure(figsize=(12, 3))
    plt.bar(range(max_len), att_weights[0])
    plt.xticks(ticks=range(max_len), labels=[str(word) for word in test_sentence[0]], rotation=45)
    plt.title("Poids d'Attention sur la Phrase de Test")
    plt.xlabel("Mots de la séquence d'entrée (indices)")
    plt.ylabel("Score d'Attention")
    plt.tight_layout()
    plt.show()