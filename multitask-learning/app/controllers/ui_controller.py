from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from app.services.dataset_service import (
    save_uploaded_file, 
    get_all_datasets, 
    read_dataset_preview, 
    delete_dataset
)
from app.services.preprocessing_service import run_preprocessing_pipeline



ui_bp = Blueprint('ui', __name__)

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
    dataset_list = get_all_datasets()
    return render_template('preprocessing.html', dataset_list=dataset_list)

@ui_bp.route('/preprocessing/run', methods=['POST'])
def run_pipeline():
    try:
        input_file = request.form.get("input_file")
        output_file = request.form.get("output_file")

        if not input_file or not output_file:
            raise ValueError("input_file dan output_file harus disertakan.")

        log = run_preprocessing_pipeline(input_file, output_file)
        return jsonify({"status": "success", "log": log})
    except Exception as e:
        return jsonify({"status": "error", "log": str(e)})

@ui_bp.route('/profile')
def profile():
    return render_template('profile.html')
@ui_bp.route('/rtl')
def rtl():
    return render_template('rtl.html')
@ui_bp.route('/sign-in')
def signin():
    return render_template('sign-in.html')
@ui_bp.route('/sign-up')
def singnup():
    return render_template('sign-up.html')


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