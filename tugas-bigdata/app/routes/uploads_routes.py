from flask import Blueprint, request, jsonify
from app.spark.spark_job import create_spark_session, process_csv
from flask_cors import CORS

upload_bp = Blueprint('upload', __name__)
spark = create_spark_session()
CORS(upload_bp)

@upload_bp.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = file.filename
    file.save(file_path)

    try:
        process_csv(spark, file_path)
        return jsonify({'message': 'CSV uploaded and processed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
