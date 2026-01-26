"# DeepLearning_TP" 
"# DeepLearning_TP" 
# Travaux Pratiques en Deep Learning Engineering

Ce dépôt contient l'ensemble des travaux pratiques réalisés dans le cadre du cours de Deep Learning Engineering du Département Génie Informatique de l'ENSPY. Le projet couvre le cycle de vie complet d'un modèle de Deep Learning, de la théorie fondamentale à la mise en production, en passant par l'amélioration des architectures et l'exploration de sujets avancés.

## Structure du Dépôt

Ce projet est organisé en utilisant des branches Git, où chaque branche correspond à un Travail Pratique (TP) spécifique. Pour consulter le code d'un TP, veuillez basculer sur la branche correspondante.

- **[TP1](https://github.com/BouiyodaJoseph/DeepLearning_TP/tree/TP1) :** Fondations du Deep Learning.
- **[TP2](https://github.com/BouiyodaJoseph/DeepLearning_TP/tree/TP2) :** Amélioration de Réseaux de Neurones.
- **[TP3](https://github.com/BouiyodaJoseph/DeepLearning_TP/tree/TP3) :** Réseaux de Neurones Convolutifs (CNNs).
- **[TP4](https://github.com/BouiyodaJoseph/DeepLearning_TP/tree/TP4) :** Segmentation et Données 3D (U-Net).
- **[TP5](https://github.com/BouiyodaJoseph/DeepLearning_TP/tree/TP5) :** Modélisation de Séquences et Attention.

*Remplacez `BouiyodaJoseph/DeepLearning_TP` par votre propre nom d'utilisateur et nom de dépôt si nécessaire.*

## Contenu des Travaux Pratiques

### TP1 : Fondations du Deep Learning
*(Branche : `TP1`)*

Ce premier TP couvre les concepts fondamentaux de l'apprentissage profond et l'ingénierie logicielle de base.
- **Concepts théoriques :** Descente de gradient, backpropagation, rôle des couches.
- **Implémentation :** Construction d'un réseau de neurones dense avec Keras pour la classification sur le jeu de données **MNIST**.
- **Ingénierie :** Mise en place du versionnement avec **Git/GitHub** et du suivi d'expériences avec **MLflow**.
- **Déploiement :** Création d'une API web avec **Flask** et conteneurisation de l'application avec **Docker**.

### TP2 : Amélioration de Réseaux de Neurones
*(Branche : `TP2`)*

Ce TP se concentre sur les techniques permettant de diagnostiquer et d'améliorer la performance d'un modèle.
- **Concepts théoriques :** Diagnostic Biais/Variance, régularisation (L2, Dropout), Batch Normalization, optimiseurs avancés (Adam, RMSprop).
- **Implémentation :**
  - Analyse du surapprentissage sur le modèle du TP1.
  - Application de la **régularisation L2** et du **Dropout** pour corriger le surapprentissage.
  - Comparaison expérimentale de différents **optimiseurs** via une boucle et suivi avec MLflow.
  - Évaluation de l'impact de la **Batch Normalization** sur la vitesse de convergence.

### TP3 : Réseaux de Neurones Convolutifs (CNNs)
*(Branche : `TP3`)*

Ce TP introduit les architectures spécialisées pour la vision par ordinateur.
- **Concepts théoriques :** Opérations de convolution et de pooling, architecture des CNNs, réseaux résiduels (ResNets).
- **Implémentation :**
  - Construction d'un **CNN classique** pour la classification sur le jeu de données **CIFAR-10**.
  - Implémentation d'un **Mini-ResNet** à l'aide de blocs résiduels pour améliorer la performance et réduire le surapprentissage.
- **Concepts avancés :** Exploration théorique de la segmentation d'image (U-Net), de la détection d'objets et du transfert de style neuronal (VGG16).

### TP4 : Segmentation d'Image et Données 3D
*(Branche : `TP4`)*

Ce TP plonge dans des tâches de vision avancées, notamment la segmentation sémantique et la gestion de données volumétriques.
- **Concepts théoriques :** Architecture U-Net, skip connections par concaténation, métriques de segmentation (IoU, Dice), convolutions 3D.
- **Implémentation :**
  - Construction complète d'une architecture **U-Net** pour une tâche de segmentation binaire sur des données simulées.
  - Implémentation de la **perte de Dice** et des métriques Dice/IoU comme fonctions personnalisées dans Keras.
  - Création d'un modèle basé sur des **convolutions 3D** et application des bonnes pratiques MLOps pour l'enregistrement de son architecture et de ses hyperparamètres avec MLflow.

### TP5 : Modélisation de Séquences et Mécanismes d'Attention
*(Branche : `TP5`)*

Ce dernier TP explore le traitement de séquences, les mécanismes d'attention et se termine par un défi de recherche.
- **Concepts théoriques :** Mécanismes d'attention (Scaled Dot-Product, Self-Attention vs. Cross-Attention).
- **Implémentation :**
  - Création d'une **couche d'attention personnalisée** en Keras et visualisation de son effet.
  - Construction d'un modèle **Sequence-to-Sequence (Encoder-Decoder)** avec un LSTM bi-directionnel et une cross-attention pour la prédiction de séries temporelles.
- **Défi de recherche :**
  - Analyse de l'article "Temporal Latent Space Modeling for Video Generation" (TAP).
  - Proposition d'une amélioration architecturale (ex: ajout de mémoire externe, remplacement par un Transformer) pour améliorer la consistance à long terme.
  - Rédaction d'un **article scientifique** de 4 pages présentant la méthode proposée, conformément aux standards des conférences internationales.

## Comment Lancer les Projets

Chaque dossier de TP (sur sa branche respective) contient les scripts Python nécessaires. Les instructions pour exécuter chaque exercice sont détaillées dans les commentaires des fichiers.

1.  **Clonez le dépôt :**
    ```bash
    git clone https://github.com/BouiyodaJoseph/DeepLearning_TP.git
    cd DeepLearning_TP
    ```
2.  **Basculez sur la branche du TP souhaité :**
    ```bash
    git checkout TP2  # (ou TP3, TP4, etc.)
    ```
3.  **Créez et activez un environnement virtuel :**
    ```bash
    python -m venv env
    source env/bin/activate  # Sur Linux/macOS
    .\env\Scripts\activate   # Sur Windows
    ```
4.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt 
    # (Note : un fichier requirements.txt doit être créé pour chaque TP)
    # ou pip install tensorflow numpy matplotlib mlflow ...
    ```
5.  **Exécutez les scripts Python** comme indiqué dans les consignes des TPs.



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