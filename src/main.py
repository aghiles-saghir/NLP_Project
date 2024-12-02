import os
from collections import Counter
import datetime
import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import spacy
import string


# Fonction pour charger un fichier JSONL
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Fonction de tokenisation avec spaCy
def tokenize_spacy(text):
    if pd.isnull(text):  # Gérer les textes manquants
        return []
    doc = nlp(text)
    return [token.text for token in doc]

#Fonction de lemmatisation 
def lemmatize_tokens(tokens):
    # Vérifiez si les tokens sont bien une liste
    if not isinstance(tokens, list) or not tokens:
        return []
    # Convertir les tokens en une chaîne de caractères pour spaCy
    text = " ".join(tokens)
    doc = nlp(text)
    return [token.lemma_ for token in doc]

# Fonction de suppression des stopwords
def remove_stopwords(tokens):
    # Filtrer les tokens qui ne sont pas des stopwords
    return [token for token in tokens if not nlp.vocab[token].is_stop]


# Fonction pour nettoyer les tokens
def clean_tokens(tokens):
    if not isinstance(tokens, list) or not tokens:
        return []

    # Compilation des regex pour optimisation
    regex_multiple_punctuations = re.compile(r'[\.\,\!\?\;\:]{2,}')  # Ponctuation répétée
    regex_multiple_spaces = re.compile(r'\s{2,}')  # Espaces multiples
    # regex_numbers = re.compile(r'\d+')  # Numéros
    regex_emojis = re.compile(r'[^\w\s,]')  # Émojis (tout caractère non alphanumérique ou ponctuation classique)
    regex_br_tag = re.compile(r'\.<br')  # Détecte la séquence.<br rattachée à d'autres mots

    cleaned_tokens = []
    for token in tokens:
        # Supprimer la séquence '.<br' dans un token
        token = regex_br_tag.sub('', token)

        # Supprimer les ponctuations répétées
        if regex_multiple_punctuations.match(token):
            continue
        # Supprimer les espaces multiples (inutile dans les tokens, mais par sécurité)
        if regex_multiple_spaces.match(token):
            continue
        # Supprimer les émojis
        if regex_emojis.match(token):
            continue
        # Supprimer les ponctuations uniques
        if token in string.punctuation:
            continue
        # Ajouter le token nettoyé s'il reste du contenu
        if token.strip():
            cleaned_tokens.append(token)
    return cleaned_tokens

# Fonction pour calculer les mots fréquents dans chaque cluster
def get_top_words(documents, top_n=10):
    word_list = []
    for doc in documents:
        word_list.extend(doc)
    return Counter(word_list).most_common(top_n)

## ---------------------------------------------------------------------

# Chemins des fichiers d'entrée
reviews_file_path = './data/reviews.jsonl'
meta_file_path = './data/meta.jsonl'

# Créer le dossier de sortie si nécessaire
output_dir = './processed_data'
os.makedirs(output_dir, exist_ok=True)

# Date pour l'enregistrement des fichiers
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Charger les données
reviews_data = load_jsonl(reviews_file_path)
meta_data = load_jsonl(meta_file_path)

# Convertir en DataFrame
reviews_df = pd.DataFrame(reviews_data)
meta_df = pd.DataFrame(meta_data)

# Sélectionner les champs pertinents
reviews_selected = reviews_df[['title', 'text']].copy()
meta_selected = meta_df[['main_category', 'title', 'average_rating', 'rating_number']].copy()

# Charger le modèle de langue de spaCy
nlp = spacy.load('en_core_web_sm')

# Ajouter une colonne tokenisée (tokenisation)
reviews_selected['tokens_spacy'] = reviews_selected['text'].apply(tokenize_spacy)

# Appliquer la lemmatisation
reviews_selected['lemmas_from_tokens'] = reviews_selected['tokens_spacy'].apply(lemmatize_tokens)

# Supprimer les stopwords
reviews_selected['lemmas_no_stopwords'] = (reviews_selected['lemmas_from_tokens'].apply(remove_stopwords))

# Supprimer la ponctuation après suppression des stopwords
reviews_selected['lemmas_cleaned'] = reviews_selected['lemmas_no_stopwords'].apply(clean_tokens)

# Convertir les listes nettoyées en chaînes de caractères pour TfidfVectorizer
reviews_selected['cleaned_text'] = reviews_selected['lemmas_cleaned'].apply(lambda x: " ".join(x))

# Convertir en minuscules
reviews_selected['cleaned_text'] = reviews_selected['cleaned_text'].str.lower()

# Afficher un aperçu des données nettoyées
print(reviews_selected[['text', 'cleaned_text']].head())

# Enregistrer les données traitées
processed_file_path = os.path.join(output_dir, f'reviews_processed_{date}.jsonl')
reviews_selected[['cleaned_text']].to_json(processed_file_path, orient='records', lines=True)

# Charger les textes nettoyés pour vectorisation
cleaned_texts = reviews_selected['cleaned_text']

# Représenter les documents sous forme vectorielle
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)
print(X)

print("\n------ Clustering des avis avec KMeans...\n")

# Appliquer l'algorithme KMeans pour regrouper les avis
kmeans = KMeans(n_clusters=5, random_state=42).fit(X)

# Enregistrer les groupes
reviews_selected['cluster'] = kmeans.labels_

# Afficher les groupes
print(reviews_selected[['cluster', 'text']].head())

# Afficher les mots les plus fréquents par cluster
for cluster_id in range(5):
    cluster_docs = reviews_selected[reviews_selected['cluster'] == cluster_id]['lemmas_cleaned']
    top_words = get_top_words(cluster_docs, top_n=10)
    print(f"Cluster {cluster_id} : {top_words}")

# Enregistrer les groupes
clustered_file_path = os.path.join(output_dir, f'reviews_clustered_{date}.jsonl')

# Enregistrer les données traitées
reviews_selected[['cluster', 'cleaned_text']].to_json(clustered_file_path, orient='records', lines=True)

print("Fin du traitement ------\n")

top_words = get_top_words(reviews_selected['lemmas_cleaned'], top_n=10)
print(f"Top words : {top_words}")

# Appliquer DBSCAN pour regrouper les avis
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)
