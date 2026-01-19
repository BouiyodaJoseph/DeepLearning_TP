import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import json # Necessaire pour la serialisation de l'architecture

# ============================================================================
# PARTIE 3 / EXERCICE 3 : BLOC CONV3D ET DISCIPLINE D'INGÉNIERIE
# ============================================================================

def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    """
    Définit un bloc Conv3D simple pour comprendre les formes et la structure.
    L'entrée est un tenseur 5D : (batch, profondeur, hauteur, largeur, canaux).
    """
    inputs = keras.layers.Input(input_shape)
    
    # Bloc Convolutif 3D - 1
    # Le noyau est (3, 3, 3), il se deplace dans les 3 dimensions spatiales.
    x = keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    # MaxPool3D reduit la taille sur les 3 dimensions.
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    
    # TODO complété : Ajouter un second bloc Conv3D avec 32 filtres et un autre MaxPool3D
    x = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    
    # Couches finales pour avoir une sortie simple (factice pour cet exercice)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# --- POINT D'ENTRÉE PRINCIPAL POUR L'EXÉCUTION ---
if __name__ == '__main__':
    
    # 1. Définir le nom de l'expérience dans MLflow
    # C'est une bonne pratique pour organiser ses projets.
    mlflow.set_experiment("3D_Volumetric_Analysis")
    
    # 2. Démarrer une nouvelle exécution (run) dans MLflow
    with mlflow.start_run(run_name="Conv3D_Baseline"):
        print("Creation du modele Conv3D et enregistrement de l'experience avec MLflow...")
        
        # Créer le modèle
        model_3d = simple_conv3d_block()
        print("\n--- Resume de l'architecture du modele 3D ---")
        model_3d.summary()
        
        # --- Application des bonnes pratiques d'ingénierie avec MLflow ---
        
        # Pratique 1 : Enregistrer l'architecture complete du modele
        # Cela garantit la reproductibilite. On sait exactement quel modele a ete utilise.
        model_config_json = model_3d.to_json()
        # On utilise log_dict pour sauvegarder le JSON dans un fichier d'artefact.
        mlflow.log_dict({"model_config": model_config_json}, "artifacts/model_architecture.json")
        print("\nArchitecture du modele enregistree dans les artefacts MLflow.")

        # Pratique 2 : Enregistrer les hyperparametres cles
        print("Enregistrement des hyperparametres...")
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("filters_start", 16)
        mlflow.log_param("input_shape", "(32, 32, 32, 1)")
        mlflow.log_param("kernel_size", "(3, 3, 3)")
        
        # Pratique 3 : Simuler un entrainement et enregistrer les metriques finales
        # TODO complété : Simuler un résultat d'entraînement en loggant une métrique finale.
        print("Simulation de l'entrainement et enregistrement des metriques...")
        # On genere des valeurs aleatoires pour simuler le resultat
        final_val_loss = np.random.uniform(0.1, 0.4)
        final_val_metric = 1 - final_val_loss
        
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("simulated_final_dice", final_val_metric)
        
        print(f"Metriques simulees enregistrees : final_val_loss={final_val_loss:.4f}")

        print("\nSuivi MLflow complet pour l'experience du bloc 3D.")
        print("Lancez 'mlflow ui' pour verifier les resultats dans l'experience '3D_Volumetric_Analysis'.")