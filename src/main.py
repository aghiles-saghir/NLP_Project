import datetime
import json
import pandas as pd
import re
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
    regex_numbers = re.compile(r'\d+')  # Numéros
    regex_emojis = re.compile(r'[^\w\s,]')  # Émojis (tout caractère non alphanumérique ou ponctuation classique)

    cleaned_tokens = []
    for token in tokens:
        # Supprimer les ponctuations répétées
        if regex_multiple_punctuations.match(token):
            continue
        # Supprimer les espaces multiples (inutile dans les tokens, mais par sécurité)
        if regex_multiple_spaces.match(token):
            continue
        # Supprimer les numéros
        if regex_numbers.match(token):
            continue
        # Supprimer les émojis
        if regex_emojis.match(token):
            continue
        # Supprimer les ponctuations uniques
        if token in string.punctuation:
            continue
        # Ajouter le token nettoyé
        cleaned_tokens.append(token)
    return cleaned_tokens

## ---------------------------------------------------------------------

# Charger les fichiers
reviews_file_path = './data/reviews.jsonl'
meta_file_path = './data/meta.jsonl'
# Date pour l'enregistrement des fichiers
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

reviews_data = load_jsonl(reviews_file_path)
meta_data = load_jsonl(meta_file_path)

# Convertir en DataFrame
reviews_df = pd.DataFrame(reviews_data)
meta_df = pd.DataFrame(meta_data)

# Sélectionner les champs pertinents
reviews_selected = reviews_df[['title', 'text']]  # Champs pertinents des avis
meta_selected = meta_df[['main_category', 'title', 'average_rating', 'rating_number']]  # Champs pertinents des métadonnées

# Afficher un aperçu des données
print("Reviews DataFrame:")
print(reviews_selected.head())

print("\nMeta DataFrame:")
print(meta_selected.head())

# Charger le modèle de langue de spaCy
nlp = spacy.load('en_core_web_sm')

# Ajouter une colonne tokenisée (tokenisation)
reviews_selected['tokens_spacy'] = reviews_selected['text'].apply(tokenize_spacy)

# Appliquer la lemmatisation
reviews_selected['lemmas_from_tokens'] = reviews_selected['tokens_spacy'].apply(lemmatize_tokens)

# Supprimer les stopwords
reviews_selected['lemmas_no_stopwords'] = reviews_selected['lemmas_from_tokens'].apply(remove_stopwords)

# Supprimer la ponctuation après suppression des stopwords
reviews_selected['lemmas_cleaned'] = reviews_selected['lemmas_no_stopwords'].apply(clean_tokens)

print(reviews_selected[['lemmas_no_stopwords', 'lemmas_cleaned']].head())


# Enregistrer les données traitées
reviews_selected['lemmas_cleaned'].to_csv(f'./processed_data/reviews_processed_{date}.csv', index=False)