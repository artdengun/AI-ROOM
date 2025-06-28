# Multitask IndoBERT Backend (Flask)

API Backend untuk menjalankan model IndoBERT multitask yang mengklasifikasikan **sentimen**, **emosi**, dan **keaslian ulasan** (asli/palsu) pada aplikasi pinjaman online. Dibangun dengan Flask dan PyTorch.

## 🚀 Fitur
- Analisis Sentimen (Negatif, Netral, Positif)
- Deteksi Emosi (6 Kategori: Marah, Sedih, dll.)
- Deteksi Ulasan Palsu (Asli / Palsu)
- Model: `indobenchmark/indobert-base-p1` dengan multitask head
- Disusun dengan struktur clean code + Flask Blueprint
- Terhubung ke PostgreSQL via SQLAlchemy

## 🧰 Instalasi

```bash
git clone <repo-url>
cd multitask_backend
python -m venv venv
source venv/bin/activate  # atau .\venv\Scripts\activate di Windows
pip install -r requirements.txt