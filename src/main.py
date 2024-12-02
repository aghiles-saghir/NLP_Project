import pandas as pd
import json
import spacy
import pandas as pd
import datetime
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

# Fonction pour supprimer ponctuation et espaces vides
def remove_punctuation_and_spaces(tokens):
    # Filtrer les tokens qui ne sont pas de la ponctuation et qui ne sont pas des espaces vides
    return [token for token in tokens if token not in string.punctuation and token.strip() != '']


## ---------------------------------------------------------------------

# Charger les fichiers
reviews_file_path = './data/reviews.jsonl'
meta_file_path = './data/meta.jsonl'

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
print(reviews_selected[['text', 'tokens_spacy']].head())

# Appliquer la lemmatisation
reviews_selected['lemmas_from_tokens'] = reviews_selected['tokens_spacy'].apply(lemmatize_tokens)
print(reviews_selected[['tokens_spacy', 'lemmas_from_tokens']].head())

# Supprimer les stopwords
reviews_selected['lemmas_no_stopwords'] = reviews_selected['lemmas_from_tokens'].apply(remove_stopwords)
print(reviews_selected[['lemmas_from_tokens', 'lemmas_no_stopwords']].head())

# Supprimer la ponctuation après suppression des stopwords
reviews_selected['lemmas_cleaned'] = reviews_selected['lemmas_no_stopwords'].apply(remove_punctuation_and_spaces)

print(reviews_selected[['lemmas_no_stopwords', 'lemmas_cleaned']].head())




date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Enregistrer les données traitées
reviews_selected.to_json(f'./processed_data/reviews_processed_{date}.jsonl', lines=True, orient='records')