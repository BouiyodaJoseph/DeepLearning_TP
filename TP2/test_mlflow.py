import mlflow

# On définit un nom d'expérience pour être plus clair
mlflow.set_experiment("Test Minimal")

# On démarre une seule exécution
with mlflow.start_run(run_name="Mon Premier Test"):
    print("Enregistrement d'un parametre et d'une metrique...")
    
    # On enregistre une fausse information
    mlflow.log_param("test_param", "valeur1")
    mlflow.log_metric("test_metric", 0.95)
    
    print("Enregistrement termine.")