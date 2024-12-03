"""This script processes the reviews data and clusters them using KMeans."""

# Importing libraries
import json
import pandas as pd
import re

# ---------------------------------------------------------------------

# Fonction pour charger un fichier JSONL
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# ---------------------------------------------------------------------

# Charger 200 lignes depuis le fichier des données
data = load_jsonl('./processed_data/reviews_processed.jsonl')[:200]
# Afficher les 5 premières lignes
df = pd.DataFrame(data)
print(df.head())
