import pandas as pd
import glob, os

# === KONFIGURASI ===
folder_path = r"C:\Users\denig\AI-ROOM\dataset"
output_file = "gabungan_Data.csv"  # Output terkompresi

# === CARI SEMUA FILE .csv DI FOLDER ===
csv_files = glob.glob(f"{folder_path}/*.csv")
print(f"Ditemukan {len(csv_files)} file CSV:")
for f in csv_files:
    print(" •", os.path.basename(f))

# === BACA & GABUNG DATAFRAME DENGAN KOLOM SUMBER ===
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df['Sumber File'] = os.path.basename(file)  # Nama file saja
    dfs.append(df)

gabungan_df = pd.concat(dfs, ignore_index=True)

# Hitung jumlah baris
num_rows = len(gabungan_df)  # Bisa juga pakai gabungan_df.shape[0] :contentReference[oaicite:4]{index=4}

# === SIMPAN DENGAN KOLOM KOMPRIMI GZIP ===
gabungan_df.to_csv(output_file, index=False)
print(f"✔️ File gabungan disimpan sebagai: {output_file}")
print(f"Total baris data: {num_rows}")
