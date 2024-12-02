import pandas as pd
import json

# Fonction pour charger un fichier JSONL
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

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

