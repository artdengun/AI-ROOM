import pandas as pd
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Aktifkan tqdm untuk Pandas apply
tqdm.pandas(desc="⏳ Memproses KBBI")



factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 0. Load definisi KBBI offline (CSV lokal)
df_kbbi = pd.read_csv("kbbi_v.csv")  # kolom: lema, definisi
print("Kolom KBBI CSV:", df_kbbi.columns.tolist())
# 1. Load file kata & mapping
df_kata = pd.read_csv("frekuensi_kata_baku.csv")
df_map = pd.read_csv("mapping_kata.csv")
# 3. Gabungkan mapping awal
df = pd.merge(df_kata, df_map, on="Kata", how="left")
df = df.head(50)
# 4. Fungsi klasifikasi berdasarkan definisi dari CSV lokal
def klasifikasi_dari_kbbi(kata):
    dasar = stemmer.stem(kata)
    entri = df_kbbi[df_kbbi["nama"].str.lower() == dasar]

    if entri.empty:
        return "Netral"

    definisi = entri.iloc[0]["submakna"]
    if pd.isna(definisi):
        return "Netral"

    print(f"{kata} → {dasar} → {definisi}")  # Debug

    if any(w in definisi for w in ["puas", "senang", "gembira", "bahagia", "baik"]):
        return "Senang"
    elif any(w in definisi for w in ["marah", "emosi", "kesal", "jengkel"]):
        return "Emosi"
    elif any(w in definisi for w in ["kecewa", "gagal", "tidak berhasil", "penolakan"]):
        return "Kecewa"
    elif any(w in definisi for w in ["sedih", "murung", "galau", "patah hati"]):
        return "Sedih"
    elif any(w in definisi for w in ["cemas", "khawatir", "bingung", "panik"]):
        return "Cemas"
    elif any(w in definisi for w in ["takut", "resiko", "kerugian", "bahaya"]):
        return "Takut"
    elif any(w in definisi for w in ["syukur", "terima kasih", "alhamdulillah", "berkah"]):
        return "Syukur"
    else:
        return "Netral"

# 5. Isi kategori kosong dengan hasil dari KBBI (pakai progress bar)
df["Emosi"] = df["Emosi"].fillna(df["Kata"].progress_apply(klasifikasi_dari_kbbi))

# 6. Kelompokkan hasil akhir
hasil = df.groupby("Emosi")["Kata"].apply(list).to_dict()

# 7. Simpan ke file Python
with open("keyword_emosi.py", "w", encoding="utf-8") as f:
    f.write("keyword_emosi = ")
    f.write(repr(hasil))

print("✅ keyword_emosi.py berhasil dibuat dengan 100 kata pertama.")