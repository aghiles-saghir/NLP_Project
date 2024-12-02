
# Clustering et Analyse de Fréquences des Mots dans des Documents

Ce projet implémente un pipeline de traitement de texte pour regrouper des documents en clusters à l'aide de l'algorithme KMeans et analyser les mots fréquents dans chaque cluster.

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

## 🛠️ Utilisation

1. Placez vos fichiers JSONL d'entrée dans le dossier `./data` :
   - `reviews.jsonl` : Contient les avis à analyser.
   - `meta.jsonl` : Contient des métadonnées.

2. Exécutez le script principal :
   ```bash
   python main.py
   ```

3. Les résultats seront sauvegardés dans le dossier `./processed_data` avec un horodatage.

## 📊 Résultats

- Chaque cluster contient les textes groupés selon leur similarité.
- Les 10 mots les plus fréquents par cluster sont affichés dans la console.
- Les données enrichies (clusters, textes nettoyés) sont sauvegardées dans un fichier JSONL.

## 📈 Évaluation

- Le **Silhouette Score** est calculé pour évaluer la qualité des clusters. Ce score est affiché dans la console.

## 🛡️ Conventions de codage

- Le code utilise Python 3.12.
- Les bibliothèques principales sont :
  - `pandas` pour la manipulation des données.
  - `scikit-learn` pour la vectorisation et le clustering.
  - `spaCy` pour le traitement du langage naturel.

## 📂 Structure du projet

```plaintext
.
├── data/
│   ├── reviews.jsonl          # Fichier d'avis (input)
│   ├── meta.jsonl             # Fichier de métadonnées (facultatif)
├── processed_data/
│   ├── reviews_processed_<date>.jsonl # Fichiers nettoyés
│   ├── reviews_clustered_<date>.jsonl # Fichiers regroupés par cluster
├── src                  # Script principal
│   ├── main.py           # Pipeline de traitement de texte
├── requirements.txt           # Dépendances
├── README.md                  # Documentation
└── .gitignore                 # Fichiers ignorés par Git
```

## 📚 Ressources

- **spaCy** : [Documentation officielle](https://spacy.io/)
- **Scikit-learn** : [Documentation officielle](https://scikit-learn.org/)
- **TF-IDF** : [Principe expliqué](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## 📝 Auteurs

- **Aghiles SAGHIR**
- **Amayas MAHMOUDI**