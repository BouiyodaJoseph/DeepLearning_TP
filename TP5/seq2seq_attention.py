import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import mlflow

# ============================================================================
# 1. GÉNÉRATION D'UN JEU DE DONNÉES SYNTHÉTIQUE
# ============================================================================
def generate_time_series(num_points):
    """Génère une série temporelle complexe."""
    time = np.arange(num_points)
    # Combinaison de plusieurs sinusoides et d'une tendance
    series = 0.5 * np.sin(time / 20) + 0.3 * np.sin(time / 10) + 0.2 * time / num_points
    # Ajout d'un peu de bruit
    series += 0.1 * np.random.randn(num_points)
    return series

def create_supervised_dataset(series, input_len, output_len):
    """Transforme une série temporelle en un jeu de données supervisé."""
    X, y = [], []
    for i in range(len(series) - input_len - output_len + 1):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

# --- Paramètres du jeu de données ---
INPUT_SEQUENCE_LENGTH = 50
OUTPUT_SEQUENCE_LENGTH = 10
TOTAL_POINTS = 2000

# Générer la série et le jeu de données
time_series = generate_time_series(TOTAL_POINTS)
X, y = create_supervised_dataset(time_series, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH)

# Ajouter une dimension pour correspondre à l'entrée des LSTMs
X = np.expand_dims(X, -1)
y = np.expand_dims(y, -1)

# Diviser en ensembles d'entraînement et de validation
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print("--- Jeu de donnees cree ---")
print(f"Forme de X_train : {X_train.shape}") # (batch, input_len, features)
print(f"Forme de y_train : {y_train.shape}") # (batch, output_len, features)

# ============================================================================
# 2. CONSTRUCTION DU MODÈLE SEQUENCE-TO-SEQUENCE AVEC ATTENTION
# ============================================================================

def build_encoder_decoder_attention_model(input_len, output_len, hidden_dim):
    """Construit un modèle Seq2Seq avec Encoder, Decoder et Cross-Attention."""
    
    # --- ENCODER ---
    encoder_inputs = layers.Input(shape=(input_len, 1))
    # LSTM Bi-directionnel pour capturer les dépendances dans les deux sens
    encoder_lstm = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_inputs)
    state_h = layers.Concatenate()([forward_h, backward_h])
    state_c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # --- DECODER ---
    decoder_inputs = layers.Input(shape=(output_len, 1))
    # Le LSTM du décodeur doit retourner la séquence complète pour l'attention
    decoder_lstm = layers.LSTM(hidden_dim * 2, return_sequences=True) # *2 car l'état de l'encodeur est concaténé
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # --- CROSS-ATTENTION ---
    # La couche d'attention de Keras est parfaite pour la Cross-Attention
    # Query: decoder_outputs (ce que je suis en train de générer)
    # Value: encoder_outputs (sur quoi je dois porter mon attention dans la séquence d'entrée)
    attention_layer = layers.Attention()
    attention_result = attention_layer([decoder_outputs, encoder_outputs])

    # Concaténer la sortie du décodeur et le résultat de l'attention
    concat = layers.Concatenate()([decoder_outputs, attention_result])
    
    # Couche de sortie pour prédire le point suivant de la série
    decoder_dense = layers.Dense(1, activation='linear')
    decoder_outputs = decoder_dense(concat)
    
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# --- Préparation des entrées pour le décodeur (Teacher Forcing) ---
# Pour l'entraînement, on donne au décodeur la séquence attendue, décalée d'un pas
decoder_input_train = np.zeros_like(y_train)
decoder_input_train[:, 1:, :] = y_train[:, :-1, :]
decoder_input_val = np.zeros_like(y_val)
decoder_input_val[:, 1:, :] = y_val[:, :-1, :]


# ============================================================================
# 3. ENTRAÎNEMENT ET SUIVI MLOps
# ============================================================================
mlflow.set_experiment("TimeSeries_Seq2Seq_Attention")

with mlflow.start_run(run_name="LSTM_CrossAttention_Baseline"):
    hidden_dim = 64
    
    # Construire et compiler le modèle
    model = build_encoder_decoder_attention_model(INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, hidden_dim)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Enregistrer les paramètres avec MLflow
    mlflow.log_param("input_sequence_length", INPUT_SEQUENCE_LENGTH)
    mlflow.log_param("output_sequence_length", OUTPUT_SEQUENCE_LENGTH)
    mlflow.log_param("hidden_dim", hidden_dim)
    
    # Entraîner le modèle
    history = model.fit(
        [X_train, decoder_input_train], y_train,
        validation_data=([X_val, decoder_input_val], y_val),
        epochs=20,
        batch_size=64
    )
    
    # Enregistrer les métriques finales
    final_val_loss = history.history['val_loss'][-1]
    mlflow.log_metric("final_val_loss", final_val_loss)
    
    # TODO (MLOps Best Practice) : Simuler le suivi de l'étendue de l'attention
    # Pour une vraie implémentation, on créerait un modèle pour extraire les poids
    # d'attention, on les analyserait et on calculerait une métrique (ex: l'entropie,
    # ou la position moyenne du poids maximum). Ici, nous simulons simplement.
    simulated_attention_span = np.random.uniform(low=INPUT_SEQUENCE_LENGTH/2, high=INPUT_SEQUENCE_LENGTH)
    mlflow.log_metric("simulated_attention_span", simulated_attention_span)
    
    # --- Visualisation d'une prédiction ---
    sample_index = np.random.randint(0, len(X_val))
    input_seq = X_val[sample_index:sample_index+1]
    decoder_input_seq = decoder_input_val[sample_index:sample_index+1]
    
    prediction = model.predict([input_seq, decoder_input_seq])[0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(INPUT_SEQUENCE_LENGTH), input_seq[0], 'b-', label='Input Sequence')
    true_output_range = np.arange(INPUT_SEQUENCE_LENGTH, INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH)
    plt.plot(true_output_range, y_val[sample_index], 'g-', label='True Future')
    plt.plot(true_output_range, prediction, 'r--', label='Predicted Future')
    plt.title("Prédiction de Série Temporelle avec Attention")
    plt.legend()
    plt.show()

print("\n--- Exercice 2 terminé. Lancez 'mlflow ui' pour voir les résultats. ---")