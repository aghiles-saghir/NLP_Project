# Importation des bibliothèques nécessaires
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

# ---------------------------------------------------------------------
# Fonction pour charger un fichier JSONL
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
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
            padding='max_length',
            return_tensors="pt",
        )
        input_ids.append(encoding['input_ids'].squeeze(0))
        attention_masks.append(encoding['attention_mask'].squeeze(0))

    return torch.stack(input_ids), torch.stack(attention_masks)

# ---------------------------------------------------------------------
# Fonction pour prédire les sentiments
def predict_sentiments(input_ids, attention_masks, model, batch_size):
    model.eval()  # Mode évaluation
    sentiments = []

    # Diviser les données en lots
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_masks = attention_masks[i:i + batch_size]

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            sentiments.extend(predictions.tolist())

    return sentiments

# ---------------------------------------------------------------------
# Charger et préparer les données
file_path = './processed_data/reviews_processed.jsonl'
data = load_jsonl(file_path)[:200]  # Limitation à 200 lignes pour les tests
df = pd.DataFrame(data)

# Prétraitement : Extraire les textes des avis
texts = df['cleaned_text'].tolist()  # Remplacer 'cleaned_text' par la colonne correcte

# Charger le modèle et le tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prétraiter les textes
max_len = 128
input_ids, attention_masks = preprocess_texts(texts, tokenizer, max_len)

# Analyser les sentiments
batch_size = 16
sentiments = predict_sentiments(input_ids, attention_masks, model, batch_size)

# Ajouter les résultats au DataFrame
df['sentiment'] = sentiments

# Afficher les premières lignes avec les sentiments prédits
print(df.head())
