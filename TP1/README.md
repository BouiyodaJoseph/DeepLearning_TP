# Projet de Deep Learning Engineering (TP1 à TP5)

Ce dépôt documente une série de cinq travaux pratiques couvrant le cycle de vie complet du développement de modèles en Deep Learning. Le projet commence par les fondations théoriques et l'implémentation de réseaux de neurones simples, progresse vers des techniques d'amélioration et des architectures avancées comme les CNNs et l'U-Net, et se termine par l'exploration de la modélisation de séquences avec les mécanismes d'attention.

## Navigation dans le Dépôt

Le projet est organisé en utilisant des branches Git, chaque branche contenant le code final d'un travail pratique spécifique.

- **`main`**: Cette branche contient le README principal et potentiellement la version finale ou une synthèse du projet.
- **`TP1`**: Fondations du Deep Learning (MNIST, Flask, Docker).
- **`TP2`**: Amélioration des Réseaux de Neurones (Biais/Variance, Régularisation).
- **`TP3`**: Réseaux de Neurones Convolutifs (CNNs, ResNet sur CIFAR-10).
- **`TP4`**: Segmentation d'Image et Données 3D (U-Net, Conv3D).
- **`TP5`**: Modélisation de Séquences et Attention (Seq2Seq, Recherche).

Pour accéder au code d'un TP, utilisez la commande `git checkout <nom_de_la_branche>`.

## Instructions Générales d'Installation

La plupart des TPs nécessitent les mêmes dépendances de base. Assurez-vous d'être sur la branche du TP que vous souhaitez exécuter avant de suivre ces étapes.

1.  **Clonez le dépôt (si ce n'est pas déjà fait) :**
    ```bash
    git clone https://github.com/votre-nom/votre-depot.git
    cd votre-depot
    ```

2.  **Basculez sur la branche désirée :**
    ```bash
    git checkout TP4  # Exemple pour le TP4
    ```

3.  **Créez un environnement virtuel :**
    ```bash
    python -m venv env
    ```

4.  **Activez l'environnement virtuel :**
    *   Sur Windows : `.\env\Scripts\activate`
    *   Sur macOS/Linux : `source env/bin/activate`

5.  **Installez les dépendances nécessaires :**
    ```bash
    pip install tensorflow numpy matplotlib mlflow scikit-learn
    # Pour les TPs avec API, ajoutez Flask et Requests
    pip install Flask requests
    ```

---

## Détail des Travaux Pratiques

###  BRANCH `TP1` : Fondations du Deep Learning

Ce TP pose les bases de l'entraînement d'un modèle et de son déploiement.

- **Objectifs :**
  - Entraîner un classifieur d'images sur **MNIST**.
  - Mettre en place le suivi d'expériences avec **MLflow**.
  - Créer une API web avec **Flask**.
  - Conteneuriser l'application avec **Docker**.
- **Fichiers clés :** `train_model.py`, `app.py`, `Dockerfile`.
- **Commandes d'exécution :**
  - Pour entraîner le modèle : `python train_model.py`
  - Pour lancer le serveur de l'API localement : `python app.py`
  - Pour construire l'image Docker : `docker build -t mnist-api .`
  - Pour lancer le conteneur Docker : `docker run -p 5000:5000 mnist-api`

### BRANCH `TP2` : Amélioration de Réseaux de Neurones

Ce TP explore comment diagnostiquer et corriger les problèmes courants des modèles, comme le surapprentissage.

- **Objectifs :**
  - Analyser le **biais et la variance** d'un modèle.
  - Appliquer la **régularisation (L2, Dropout)** pour réduire le surapprentissage.
  - Comparer l'efficacité de différents **optimiseurs** (SGD, RMSprop, Adam).
  - Évaluer l'impact de la **Batch Normalization**.
- **Fichiers clés :** `tp2_ex1.py`, `tp2_ex2_regularization.py`, `tp2_ex3_optimizers.py`, `tp2_ex4_batchnorm.py`.
- **Commandes d'exécution :**
  - Pour lancer une expérience : `python <nom_du_script>.py`
  - Pour visualiser les résultats des comparaisons : `mlflow ui`

### BRANCH `TP3` : Réseaux de Neurones Convolutifs (CNN)

Ce TP introduit les architectures spécialisées pour la vision par ordinateur.

- **Objectifs :**
  - Entraîner un **CNN** sur **CIFAR-10**.
  - Implémenter un **Mini-ResNet** avec des blocs résiduels.
  - Explorer les concepts de segmentation, détection, et transfert de style.
- **Fichiers clés :** `cnn_classification.py`, `style_transfer.py`.
- **Commandes d'exécution :**
  - Pour entraîner et comparer les modèles CNN et ResNet : `python cnn_classification.py`
  - Pour préparer le transfert de style (nécessite des images `content.jpg` et `style.jpg`) : `python style_transfer.py`

### BRANCH `TP4` : Segmentation d'Image et Données 3D

Ce TP aborde la segmentation sémantique avec l'architecture U-Net et les convolutions 3D.

- **Objectifs :**
  - Implémenter une architecture **U-Net** pour la segmentation.
  - Créer des métriques personnalisées (**Dice**, **IoU**) et une perte adaptée.
  - Comprendre et implémenter un bloc de **convolution 3D**.
- **Fichiers clés :** `unet_segmentation.py`, `conv3d_analysis.py`.
- **Commandes d'exécution :**
  - Pour entraîner l'U-Net sur des données simulées : `python unet_segmentation.py`
  - Pour analyser le modèle 3D et enregistrer ses métriques avec MLflow : `python conv3d_analysis.py`

### BRANCH `TP5` : Modélisation de Séquences et Attention

Ce dernier TP se concentre sur le traitement de séquences, les mécanismes d'attention, et se termine par un défi de recherche.

- **Objectifs :**
  - Implémenter une **couche d'attention personnalisée**.
  - Construire un modèle **Sequence-to-Sequence** avec attention pour la prédiction de séries temporelles.
  - Analyser un article de recherche (modèle **TAP**) et proposer une amélioration architecturale.
- **Fichiers clés :** `attention_exercise.py`, `seq2seq_attention.py`.
- **Commandes d'exécution :**
  - Pour tester la couche d'attention personnalisée : `python attention_exercise.py`
  - Pour entraîner le modèle de prédiction de séries temporelles : `python seq2seq_attention.py`