import time
import re
import pandas as pd

def simulate_step(step_name, delay=50, fail=False):
    time.sleep(delay)
    if fail:
        raise Exception(f"{step_name} gagal diproses")
    return f"{step_name} berhasil"


# --- Tahapan Preprocessing ---
def case_folding(text):
    return text.lower()

def cleansing(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def normalization(text):
    kamus = {"ga": "tidak", "gak": "tidak", "nggak": "tidak"}
    return ' '.join([kamus.get(kata, kata) for kata in text.split()])

def tokenizing(text):
    return text.split()

def stopword_removal(tokens):
    stopwords = {"yang", "dan", "di", "ke", "dari", "ini", "itu"}
    return [t for t in tokens if t not in stopwords]

def stemming(tokens):
    # Dummy stemming
    return [t.rstrip('ing').rstrip('an') for t in tokens]

def full_preprocessing(text):
    if pd.isna(text):
        return ""
    text = case_folding(str(text))
    text = cleansing(text)
    text = normalization(text)
    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return ' '.join(tokens)

# --- Baca file dan proses ---
def run_preprocessing_pipeline(input_file, output_file):
    df = pd.read_csv(input_file)
    
    if 'Content' not in df.columns:
        raise ValueError("Kolom 'Content' tidak ditemukan dalam file CSV.")
    
    df['Cleaned'] = df['Content'].apply(full_preprocessing)
    df.to_csv(output_file, index=False)
    print(f"✅ File hasil preprocessing disimpan ke: {output_file}")