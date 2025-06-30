import os
import pandas as pd
from datetime import datetime

DATASET_FOLDER = 'uploads/datasets'

def save_uploaded_file(file):
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    file_path = os.path.join(DATASET_FOLDER, file.filename)
    file.save(file_path)

def get_all_datasets():
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    files = []
    for filename in os.listdir(DATASET_FOLDER):
        path = os.path.join(DATASET_FOLDER, filename)
        size = os.path.getsize(path) / 1024  # KB
        uploaded_at = datetime.fromtimestamp(os.path.getctime(path))
        files.append({
            'name': filename,
            'size': round(size, 2),
            'uploaded_at': uploaded_at.strftime("%Y-%m-%d %H:%M")
        })
    return sorted(files, key=lambda x: x['uploaded_at'], reverse=True)

def read_dataset_preview(filename):
    if not filename:
        return None
    path = os.path.join(DATASET_FOLDER, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {
        'headers': df.columns.tolist(),
        'rows': df.head(10).values.tolist()
    }

def delete_dataset(filename):
    path = os.path.join(DATASET_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
