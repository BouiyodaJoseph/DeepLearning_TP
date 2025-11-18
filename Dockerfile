# Utiliser une image de base Python légère
FROM python:3.9-slim

# Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copier le fichier des dépendances. On le copie en premier pour profiter
# du cache de Docker. Si ce fichier ne change pas, les dépendances
# n'auront pas besoin d'être réinstallées à chaque build.
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application (app.py et mnist_model.h5)
COPY . .

# Indiquer que le conteneur expose le port 5000
EXPOSE 5000

# La commande à exécuter pour démarrer l'application quand le conteneur se lance
CMD ["python", "app.py"]