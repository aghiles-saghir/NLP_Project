"""Version python 3.12"""

"""This script processes the reviews data and predicts the sentiments of the reviews."""

# Importation des bibliothèques nécessaires
import json

import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------
# Fonction pour charger un fichier JSONL
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------
# Fonction pour prétraiter les données avec le tokenizer
def preprocess_texts(texts, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids.append(encoding["input_ids"].squeeze(0))
        attention_masks.append(encoding["attention_mask"].squeeze(0))

    return torch.stack(input_ids), torch.stack(attention_masks)


# ---------------------------------------------------------------------
# Fonction pour prédire les sentiments
def predict_sentiments(input_ids, attention_masks, model, batch_size):
    model.eval()  # Mode évaluation
    sentiments = []

    # Diviser les données en lots
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_masks = attention_masks[i : i + batch_size]

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            sentiments.extend(predictions.tolist())

    return sentiments


# ---------------------------------------------------------------------

# Charger et préparer les données
file_path = "./data/reviews.jsonl"
# Prendre seulement 200 lignes random
data = load_jsonl(file_path)
df = pd.DataFrame(data)
df = df.sample(n=200, random_state=42)  # Prendre seulement 200 lignes random

# Prétraitement : Extraire les textes des avis concaténé avec les titles
texts = df["title"] + ", " + df["text"]
texts = texts.tolist()  # Remplacer 'text' par la colonne correcte

# Charger le modèle et le tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prétraiter les textes
max_len = 512
input_ids, attention_masks = preprocess_texts(texts, tokenizer, max_len)

# Analyser les sentiments
batch_size = 32
sentiments = predict_sentiments(input_ids, attention_masks, model, batch_size)

# Ajouter les résultats au DataFrame
df["sentiment"] = sentiments

# Afficher les premières lignes avec les sentiments prédits
print(df[["rating", "title", "text", "sentiment"]].head())

# Calculer la corrélation entre les notes et les sentiments prédits
correlation = df["rating"].corr(df["sentiment"])

# Enregistrer les résultats dans un fichier CSV
output_file_path = "./processed_data/reviews_with_feelings.csv"
df.to_csv(output_file_path, index=False)

print(f"Corrélation entre les notes et les sentiments prédits : {correlation*100:.2f}%")
