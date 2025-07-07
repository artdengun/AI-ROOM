from flask import Blueprint, current_app, render_template, request, redirect, send_file, session, url_for, flash, jsonify
import os
import pandas as pd
from app.services.dataset_service import (
    save_uploaded_file, 
    get_all_datasets, 
    read_dataset_preview, 
    delete_dataset
)
from app.services.preprocessing_service import get_preprocessed_outputs, run_preprocessing_pipeline
import torch
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

ui_bp = Blueprint('ui', __name__)
cache_dir = os.path.join(os.getcwd(), "app", "models_cache")

# === 1. Form Pilih Dataset ===
@ui_bp.route('/predict', methods=['GET'])
def prediction_form():
    return render_template("predict.html", dataset_list=get_preprocessed_outputs())

@ui_bp.route('/predict/run', methods=['POST'])
def run_prediction():
    selected_file = request.form.get('dataset')
    if not selected_file:
        flash("Dataset belum dipilih!", "error")
        return redirect(url_for('ui.prediction_form'))

    file_path = os.path.join('uploads', 'outputs', selected_file)
    if not os.path.exists(file_path):
        flash("File tidak ditemukan!", "error")
        return redirect(url_for('ui.prediction_form'))

#   df = pd.read_csv(file_path).head(20)
    df = pd.read_csv(file_path)

    if 'Cleaned' not in df.columns or 'Score' not in df.columns:
        flash("Dataset harus mengandung kolom 'Cleaned' dan 'Score'", "error")
        return redirect(url_for('ui.prediction_form'))

    session['progress'] = 0
    session['progress_row'] = 0
    session['progress_total'] = len(df)
    session['progress_stage'] = 'Memuat model...'

    # === Setup Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Load Models ===
    sent_model = "taufiqdp/indonesian-sentiment"
    emo_model = "thoriqfy/indobert-emotion-classification"
    tok_sent = AutoTokenizer.from_pretrained(sent_model, cache_dir=cache_dir)
    mod_sent = AutoModelForSequenceClassification.from_pretrained(sent_model, cache_dir=cache_dir).eval()
    tok_emo = AutoTokenizer.from_pretrained(emo_model, cache_dir=cache_dir)
    mod_emo = AutoModelForSequenceClassification.from_pretrained(emo_model, cache_dir=cache_dir).eval()
    sent_labels = ["Negatif", "Netral", "Positif"]
    emo_labels = ["marah", "takut", "sedih", "senang", "netral", "lainnya"]
    session['progress'] = 10

    # === Sentiment Prediction ===
    session['progress_stage'] = 'Klasifikasi sentimen...'
    texts = df["Cleaned"].astype(str).tolist()
    inputs_sent = tok_sent(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = mod_sent(**inputs_sent).logits
        sent_preds = torch.argmax(logits, dim=1).tolist()
    df["sentiment_label"] = [sent_labels[i] for i in sent_preds]
    session['progress'] = 30

    # === Emotion Prediction ===
    session['progress_stage'] = 'Klasifikasi emosi...'
    inputs_emo = tok_emo(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = mod_emo(**inputs_emo).logits
        emo_preds = torch.argmax(logits, dim=1).tolist()
    df["emotion_label"] = [emo_labels[i] if i < len(emo_labels) else "unknown" for i in emo_preds]
    session['progress'] = 50

    # === Evaluasi Review ===
    session['progress_stage'] = 'Mengevaluasi ulasan...'
    verdicts = []
    promo = r"\b(pakai|gunakan|install|download|unduh|rekomendasi|coba|ayo)\b"

    for i, row in df.iterrows():
        sent, score, text = row["sentiment_label"], row["Score"], str(row["Cleaned"]).lower()
        if score == 5:
            verdict = "Palsu" if sent == "Negatif" or (sent == "Positif" and re.search(promo, text)) else "Asli"
        elif score == 4:
            verdict = "Palsu" if sent == "Negatif" or (sent in ["Netral", "Positif"] and re.search(promo, text)) else "Asli"
        elif score == 3:
            verdict = "Palsu" if re.search(promo, text) else "Asli"
        elif score == 2:
            verdict = "Palsu" if sent == "Netral" or re.search(promo, text) else "Asli"
        elif score == 1:
            verdict = "Palsu" if sent in ["Netral", "Positif"] or re.search(promo, text) else "Asli"
        else:
            verdict = "Asli"
        verdicts.append(verdict)
        session['progress_row'] = i + 1
        session['progress'] = 50 + int(((i + 1) / len(df)) * 40)

    df["review_verdict"] = verdicts

    # === Evaluasi Metrik (jika ada label sentimen asli)
    if "sentimen_label" in df.columns:
        report = classification_report(df["sentimen_label"], df["predicted_sentimen"], output_dict=True, zero_division=0)
        evaluation = {
            "precision": round(report["weighted avg"]["precision"], 3),
            "recall": round(report["weighted avg"]["recall"], 3),
            "f1_score": round(report["weighted avg"]["f1-score"], 3),
            "accuracy": round(accuracy_score(df["sentimen_label"], df["predicted_sentimen"]), 3)
        }
    else:
        evaluation = None

    session['progress_stage'] = 'Menyimpan hasil...'
    session['progress'] = 100

    output_file = os.path.join('uploads', 'outputs', f'hasil_prediksi_{selected_file}')
    df.to_csv(output_file, index=False)

    preview = df[["Cleaned", "Score", "sentiment_label", "emotion_label", "review_verdict"]].head(10).to_dict(orient="records")
    return render_template("predict.html", dataset_list=get_preprocessed_outputs(), rows=preview,
                        filename=f'hasil_prediksi_{selected_file}', evaluation=evaluation)

# === 3. Download Hasil Prediksi ===
@ui_bp.route('/predict/download', methods=['POST'])
def download_prediction():
    filename = request.form.get('filename')
    if not filename:
        flash("File tidak valid.", "error")
        return redirect(url_for('ui.prediction_form'))
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..'))
    file_path = os.path.join(app_dir, 'uploads', 'outputs', filename)
    if not os.path.exists(file_path):
        flash("File tidak ditemukan.", "error")
        return redirect(url_for('ui.prediction_form'))

    return send_file(file_path, as_attachment=True)

# === 4. API: Lacak Progres ===
@ui_bp.route('/predict/progress')
def predict_progress():
    return jsonify(
        percent=session.get('progress', 0),
        current=session.get('progress_row', 0),
        total=session.get('progress_total', 0),
        stage=session.get('progress_stage', 'Menunggu...')
    )
    
    
@ui_bp.route('/')
def dashboard():
    stat = {
        'sentiment': 91.6,
        'emotion': 84.3,
        'fake_review': 93.2,
        'total_reviews': 12430
    }

    predictions = [
        {
            "text": "Aplikasinya sangat mudah digunakan dan membantu saya",
            "sentiment": "Positif",
            "emotion": "Senang",
            "fake": "Asli"
        },
        {
            "text": "Saya kecewa karena proses pinjaman sangat lama",
            "sentiment": "Negatif",
            "emotion": "Marah",
            "fake": "Asli"
        },
        {
            "text": "Pelayanannya tidak jelas, saya ragu ini asli",
            "sentiment": "Negatif",
            "emotion": "Takut",
            "fake": "Palsu"
        },
        {
            "text": "Cukup oke tapi masih banyak bug",
            "sentiment": "Netral",
            "emotion": "Netral",
            "fake": "Asli"
        },
        {
            "text": "Saya merasa sedih setelah gagal pinjam dua kali",
            "sentiment": "Negatif",
            "emotion": "Sedih",
            "fake": "Asli"
        }
    ]

    emotion_distribution = [
        {'label': 'Senang', 'count': 3200, 'percent': 25.6},
        {'label': 'Marah', 'count': 1240, 'percent': 9.9},
        {'label': 'Sedih', 'count': 940, 'percent': 7.5},
        {'label': 'Takut', 'count': 860, 'percent': 6.9},
        {'label': 'Netral', 'count': 6190, 'percent': 49.9}
    ]

    return render_template('dashboard.html', 
                        stat=stat, 
                        predictions=predictions,
                        emotion_distribution=emotion_distribution)

@ui_bp.route('/preprocessing')
def preprocessing():
    output_file = session.pop('output_file', None)
    return render_template(
        "preprocessing.html",
        dataset_list=get_all_datasets(),
        output_file=output_file
    )


@ui_bp.route('/preprocessing/run', methods=['POST'])
def run_pipeline():
    try:
        input_file = request.form.get("input_file")
        if not input_file:
            flash("Input file harus dipilih.", "danger")
            return redirect(url_for('ui.preprocessing'))

        app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..'))
        dataset_path = os.path.join(app_dir, 'uploads', 'datasets', input_file)
        if not os.path.isfile(dataset_path):
            flash(f"File '{input_file}' tidak ditemukan.", "danger")
            return redirect(url_for('ui.preprocessing'))

        # Baca dataset
        df = pd.read_csv(dataset_path)

        # Jalankan preprocessing
        log, output_file = run_preprocessing_pipeline(df, input_file)

        session['output_file'] = output_file
        flash(log, "success")
        return redirect(url_for('ui.preprocessing'))


    except Exception as e:
        flash(f"Terjadi kesalahan: {str(e)}", "danger")
        return redirect(url_for('ui.preprocessing'))
    
@ui_bp.route('/sign-in')
def signin():
    return render_template('sign-in.html')

@ui_bp.route('/logout')
def logout():
    return render_template('sign-in.html')

@ui_bp.route('/sign-up')
def singnup():
    return render_template('sign-up.html')

@ui_bp.route('/training', methods=['GET', 'POST'])
def training():
    datasets = get_preprocessed_outputs()

    if request.method == 'POST':
        selected_file = request.form.get('dataset')
        if not selected_file:
            flash("Dataset belum dipilih.", "error")
            return redirect(url_for('ui.training'))

        project_root = os.path.abspath(os.path.join(current_app.root_path, '..'))
        file_path = os.path.join(project_root, 'uploads', 'outputs', selected_file)
        df = pd.read_csv(file_path)

        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2, random_state=42)

        # Save hasil training ke session untuk render chart + detail
        session['train_chart'] = {
            'filename': selected_file,
            'total': len(df),
            'train': len(train),
            'test': len(test)
        }

        flash("Training selesai!", "success")
        return redirect(url_for('ui.training'))

    chart_data = session.pop('train_chart', None)
    return render_template("training.html", datasets=datasets, chart_data=chart_data)

@ui_bp.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        uploaded_file = request.files.get('dataset_file')
        if uploaded_file and uploaded_file.filename != '':
            try:
                save_uploaded_file(uploaded_file)
                flash('Dataset berhasil diupload', 'success')
            except Exception as e:
                flash(f'Gagal mengupload dataset: {str(e)}', 'error')
        else:
            flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('ui.dataset'))

    filename = request.args.get('view')
    preview = None
    if filename:
        try:
            preview = read_dataset_preview(filename)
        except Exception as e:
            flash(f'Gagal membuka dataset: {str(e)}', 'error')

    return render_template(
        'dataset.html',
        dataset_list=get_all_datasets(),
        selected_dataset=filename,
        preview=preview
    )

@ui_bp.route('/dataset/delete/<filename>')
def delete_file(filename):
    try:
        delete_dataset(filename)
        flash(f'Dataset "{filename}" berhasil dihapus.', 'success')
    except Exception as e:
        flash(f'Gagal menghapus dataset: {str(e)}', 'error')
    return redirect(url_for('ui.dataset'))