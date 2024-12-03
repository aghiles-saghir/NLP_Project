"""This script processes the reviews data and clusters them using KMeans."""

# Importing libraries
import os
from collections import Counter
import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import spacy
import string

# ---------------------------------------------------------------------

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

#Fonction bigrams
def get_top_bigrams_cleaned(documents, top_n=10):
    # Convertir les listes de tokens en chaînes de caractères
    documents = [" ".join(doc) for doc in documents]
    # Initialiser CountVectorizer pour les bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(documents)
    # Extraire les bigrams et leurs fréquences
    bigrams = vectorizer.get_feature_names_out()
    frequencies = X.toarray().sum(axis=0)
    # Trier les bigrams par fréquence et convertir en tuples propres
    bigrams_freq = [(bigram, int(freq)) for bigram, freq in sorted(zip(bigrams, frequencies), key=lambda x: x[1], reverse=True)]
    return bigrams_freq[:top_n]

# ---------------------------------------------------------------------
# Initialisation des variables

# Chemins des fichiers d'entrée
reviews_file_path = './data/reviews.jsonl'
meta_file_path = './data/meta.jsonl'

# Créer le dossier de sortie si nécessaire
output_dir = './processed_data'
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Traitement des données

# Charger les données
reviews_data = load_jsonl(reviews_file_path)
# meta_data = load_jsonl(meta_file_path)

# Convertir en DataFrame
reviews_df = pd.DataFrame(reviews_data)
# meta_df = pd.DataFrame(meta_data)

# Sélectionner les champs pertinents
reviews_selected = reviews_df[['rating', 'title', 'text']].copy()
# meta_selected = meta_df[['main_category', 'title', 'average_rating', 'rating_number']].copy()

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
processed_file_path = os.path.join(output_dir, f'reviews_processed.jsonl')
reviews_selected[['rating', 'cleaned_text']].to_json(processed_file_path, orient='records', lines=True)

# Charger les textes nettoyés pour vectorisation
cleaned_texts = reviews_selected['cleaned_text']

# ---------------------------------------------------------------------
# Clustering des avis

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
    cluster_docs = reviews_selected[reviews_selected['cluster'] == cluster_id]['cleaned_text']
    # Convertir les textes en listes de mots
    cluster_docs_tokens = [doc.split() for doc in cluster_docs]
    top_words = get_top_words(cluster_docs_tokens, top_n=10)
    print(f"Cluster {cluster_id} : {top_words}")

print("Fin du traitement ------\n")

# Afficher les bigrams les plus fréquents par cluster
for cluster_id in range(5):
    cluster_docs = reviews_selected[reviews_selected['cluster'] == cluster_id]['cleaned_text']
    # Convertir les textes en listes de mots
    cluster_docs_tokens = [doc.split() for doc in cluster_docs]
    top_bigrams = get_top_bigrams_cleaned(cluster_docs_tokens, top_n=10)
    print(f"Cluster {cluster_id} : {top_bigrams}")

# Top mots global
all_docs_tokens = [doc.split() for doc in reviews_selected['cleaned_text']]
top_words = get_top_words(all_docs_tokens, top_n=10)
print(f"\n\nTop words : {top_words}\n\n")
