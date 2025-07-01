import pandas as pd
import re
import os
import time
import string
import unicodedata
import csv


def get_preprocessed_outputs():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(project_root, 'uploads', 'outputs')
    files = [f for f in os.listdir(path) if f.startswith("preprocessed_") and f.endswith(".csv")]
    return sorted(files)


def load_normalization_dict(filepath='file/kamus_normalisasi.csv'):
    normalization_dict = {}
    if not os.path.exists(filepath):
        print("🚫 File kamus tidak ditemukan:", filepath)
        return normalization_dict
    with open(filepath, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip baris header: ['tidak_baku', 'kata_baku']
        for row in reader:
            if len(row) >= 2:
                kata_tidak_baku = row[0].strip().lower()
                bentuk_baku = row[1].strip().lower()
                if kata_tidak_baku and bentuk_baku:
                    normalization_dict[kata_tidak_baku] = bentuk_baku
    print(f"✅ Kamus berhasil dimuat: {len(normalization_dict)} entri")
    return normalization_dict

def case_folding(text):
    return unicodedata.normalize("NFKD", str(text)).lower()

def cleansing(text):
    text = re.sub(r'http\S+', '', text)                          # hapus URL
    text = re.sub(r'[^\w\s]', ' ', text)                         # hapus simbol
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)                     # hapus angka & karakter aneh
    return re.sub(r'\s+', ' ', text).strip()

def tokenizing(text):
    return text.split()

def stopword_removal(tokens):
    stopwords = set([
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", 
        "dengan", "agar", "jika", "pada", "adalah"
    ])
    return [token for token in tokens if token not in stopwords]

def detect_unknown_words(text_series, normalization_dict):
    detected = set()
    for text in text_series.dropna():
        words = str(text).split()
        for word in words:
            if word not in normalization_dict and len(word) > 2:
                detected.add(word)
    return sorted(detected)

def run_preprocessing_pipeline(df, original_name):
    if 'Content' not in df.columns:
        raise ValueError("Kolom 'Content' tidak ditemukan.")

    print("▶️ Memuat kamus...")
    kamus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'file', 'kamus_normalisasi.csv'))   
    normalization_dict = load_normalization_dict(kamus_path)
    print("📦 Entri pertama:", list(normalization_dict.items())[:5])

    def full_preprocessing_local(text):
        if pd.isna(text):
            return ""
        text = case_folding(text)
        text = cleansing(text)
        text = ' '.join([normalization_dict.get(word, word) for word in text.split()])
        tokens = tokenizing(text)
        tokens = stopword_removal(tokens)
        return ' '.join(tokens)

    df['Cleaned'] = df['Content'].apply(full_preprocessing_local)

    # Simpan hasil
    timestamp = int(time.time())
    output_name = f"preprocessed_{timestamp}_{original_name}"
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'uploads', 'outputs'))
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path, index=False)

    # Simpan kata tidak dikenal
    unknown_words = detect_unknown_words(df['Cleaned'], normalization_dict)
    unknown_file = f"unknown_{timestamp}.txt"
    unknown_path = os.path.join(output_dir, unknown_file)
    with open(unknown_path, 'w', encoding='utf-8') as f:
        for word in unknown_words:
            f.write(word + '\n')

    total = len(df)
    kosong = df['Content'].isna().sum()
    log = f"✅ Preprocessing selesai. Total: {total}, kosong: {kosong}. Disimpan sebagai '{output_name}'."
    log += f"\n🧐 Ditemukan {len(unknown_words)} kata tidak dikenal, simpan ke '{unknown_file}'."

    return log, output_name