
# Analyse et Clustering des Avis

Ce projet analyse les avis utilisateurs en appliquant un traitement de texte avancé pour le clustering et la prédiction des sentiments. Il utilise des techniques de traitement du langage naturel (NLP) et de machine learning pour explorer et structurer les données textuelles.

## Table des matières
- [Structure du projet](#structure-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Auteurs](#auteurs)

---

## Structure du projet

```
src/
├── main.py              # Script pour le traitement et le clustering des avis
├── feelings_analyzer.py # Script pour la prédiction des sentiments
data/
├── reviews.jsonl        # Fichier contenant les avis au format JSONL
├── meta.jsonl           # Métadonnées associées aux avis
processed_data/
├── reviews_processed.jsonl   # Avis prétraités
├── reviews_clustered.csv     # Résultats du clustering
├── reviews_with_feelings.csv # Résultats de la prédiction des sentiments
```

---

## 📋 Fonctionnalités

1. **Chargement des données :**
   - Lecture de fichiers JSONL contenant des avis et des métadonnées.

2. **Traitement des textes :**
   - Tokenisation avec spaCy.
   - Lemmatisation des tokens.
   - Suppression des stopwords et nettoyage des tokens.
   - Conversion des textes nettoyés en une représentation vectorielle (TF-IDF).

3. **Clustering :**
   - Groupement des textes en clusters avec l'algorithme KMeans.
   - Calcul des mots les plus fréquents pour chaque cluster.

4. **Évaluation des clusters :**
   - Calcul du **Silhouette Score** pour mesurer la qualité des regroupements.

5. **Export des résultats :**
   - Sauvegarde des documents enrichis avec leur cluster dans des fichiers JSONL.

6. **Prédiction des sentiments :**
   - Utilisation du modèle pré-entraîné `nlptown/bert-base-multilingual-uncased-sentiment`.
   - Prédiction des sentiments des avis avec calcul de la corrélation entre les notes et les sentiments.

---

## Prérequis

- Python 3.12
- Bibliothèques Python :
  - `pandas`
  - `scikit-learn`
  - `spacy`
  - `transformers`
  - `torch`
  - `regex`
  - Modèle de langue spaCy : `en_core_web_sm`
    ```bash
    python -m spacy download en_core_web_sm
    ```

---

## 🚀 Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/Aghiles-S/NLP_Project.git
   cd NLP_Project
   ```

2. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez et installez le modèle spaCy :
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 🛠️ Utilisation

1. **Clustering des avis** :
   - Exécutez le script `main.py` :
     ```bash
     python src/main.py
     ```
   - Les résultats seront enregistrés dans `processed_data/reviews_clustered.csv`.

2. **Prédiction des sentiments** :
   - Exécutez le script `feelings_analyzer.py` :
     ```bash
     python src/feelings_analyzer.py
     ```
   - Les résultats seront enregistrés dans `processed_data/reviews_with_feelings.csv`.

---

## 📊 Résultats

1. **Clustering** :
   - Les avis sont regroupés en 5 clusters. Les mots et bigrams les plus fréquents dans chaque cluster sont affichés dans la console.

2. **Prédiction des sentiments** :
   - Les sentiments des avis sont prédits et enregistrés dans un fichier CSV.
   - Une corrélation est calculée entre les notes données par les utilisateurs et les sentiments prédits.

---

## 📈 Évaluation

- Le **Silhouette Score** est calculé pour évaluer la qualité des clusters. Ce score est affiché dans la console.

## 🛡️ Conventions de codage

- Le code utilise Python 3.12.
- Les noms de variables sont explicites et en minuscules.
- Les commentaires sont écrits en anglais.
- Le code est formaté avec `black` et `isort`.
- Les dépendances sont listées dans `requirements.txt`.
- Les fichiers inutiles sont ignorés par Git.

---

## 📚 Ressources

- **spaCy** : [Documentation officielle](https://spacy.io/)
- **Scikit-learn** : [Documentation officielle](https://scikit-learn.org/)
- **TF-IDF** : [Principe expliqué](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## 📝 Auteurs

- **Aghiles SAGHIR**
- **Amayas MAHMOUDI**