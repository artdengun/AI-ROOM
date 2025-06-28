import pandas as pd
import re
from collections import Counter

# === 1. Load kamus alay dari file lokal ===
kamus1 = pd.read_csv("new_kamusalay.csv", names=["alay", "baku"], encoding="ISO-8859-1")
kamus2 = pd.read_csv("combined_slang_words.txt", sep="\t", names=["alay", "baku"], engine="python")

kamus_alay = pd.concat([kamus1, kamus2]).drop_duplicates(subset="alay")
alay_dict = dict(zip(kamus_alay["alay"], kamus_alay["baku"]))

# === 2. Load dataset gabungan ===
df = pd.read_csv("dataset_gabungan.csv")
cleaned_sentences = []

# === 3. Normalisasi konten ===
for text in df["Content"].dropna().astype(str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    normalized = ' '.join([alay_dict.get(word, word) for word in text.split()])
    cleaned_sentences.append(normalized.strip())

# === 4. Tokenisasi & hitung frekuensi kata ===
all_words = []
for sentence in cleaned_sentences:
    all_words.extend(sentence.split())

word_counts = Counter(all_words)

# === 5. Simpan hasil ke CSV ===
freq_df = pd.DataFrame(word_counts.items(), columns=["Kata", "Jumlah"])
freq_df = freq_df.sort_values(by="Jumlah", ascending=False)
freq_df.to_csv("frekuensi_kata_baku.csv", index=False)

print("✅ Frekuensi kata setelah normalisasi disimpan di: app/file/frekuensi_kata_baku.csv")