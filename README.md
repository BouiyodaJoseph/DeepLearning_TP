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