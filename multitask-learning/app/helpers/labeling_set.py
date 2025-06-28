import pandas as pd
import torch
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === 1. Load dataset ===
file_path = "app/file/dataset_gabungan.csv"
df = pd.read_csv(file_path)

# === 2. Load model sentiment ===
model_name = "taufiqdp/indonesian-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

label_map = ["Negatif", "Netral", "Positif"]

def classify_sentiment(text):
    try:
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            return label_map[pred]
    except Exception:
        return "Netral"

print("📊 Melabeli sentimen otomatis...")
df["sentiment_label"] = df["Content"].astype(str).apply(classify_sentiment)

# === 3. Heuristik Fakeness ===
def classify_fakeness(score, sentiment_label, content):
    content = str(content).lower()
    promosi_keywords = r"\b(pakai|gunakan|install|download|unduh|rekomendasi|coba)\b"

    if score == 5:
        if sentiment_label == "Negatif":
            return "Palsu"
        if sentiment_label == "Netral":
            return "Asli"
        if sentiment_label == "Positif" and re.search(promosi_keywords, content):
            return "Palsu"
        return "Asli"
    
    if score == 4:
        if sentiment_label == "Negatif":
            return "Palsu"
        if sentiment_label in ["Netral", "Positif"] and re.search(promosi_keywords, content):
            return "Palsu"
        return "Asli"
    
    if score == 3:
        if re.search(promosi_keywords, content):
            return "Palsu"
        return "Asli"
    
    if score == 2:
        if sentiment_label == "Netral":
            return "Palsu"
        if re.search(promosi_keywords, content):
            return "Palsu"
        return "Asli"
    
    if score == 1:
        if sentiment_label in ["Netral", "Positif"]:
            return "Palsu"
        if re.search(promosi_keywords, content):
            return "Palsu"
        return "Asli"

    return "Asli"

print("🔍 Menentukan fake label berdasarkan skor dan isi ulasan...")
df["fake_label"] = df.apply(lambda row: classify_fakeness(
    score=int(row["Score"]),
    sentiment_label=row["sentiment_label"],
    content=row["Content"]
), axis=1)

# === 4. Siapkan kolom emotion_label kosong ===
df["emotion_label"] = ""

# === 5. Simpan hasil ===
output_file = "app/file/dataset_labeled.csv"
df.to_csv(output_file, index=False)
print(f"✅ Dataset berhasil dilabel. Simpan di: {output_file}")