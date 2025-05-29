from flask import Blueprint, request, jsonify # type: ignore
from .hive_conn import get_conn
from .spark_job import create_spark_session, process_csv
from flask_cors import CORS

bp = Blueprint('main', __name__)
spark = create_spark_session()
CORS(bp)

@bp.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Sampan file ke folder lokal
    file_path = file.filename  
    file.save(file_path)
    try:
        process_csv(spark, file_path)
        return jsonify({'message': 'CSV uploaded and processed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# PEGAWAI

@bp.route('/pegawai', methods=['POST'])
def create_pegawai():
    data = request.get_json()
    nama = data.get('nama')
    jabatan_id = data.get('jabatan_id')

    if not nama or jabatan_id is None:
        return jsonify({'error': 'nama and jabatan_id are required'}), 400

    conn = get_conn()
    cursor = conn.cursor()

    # Cari ID terakhir
    cursor.execute("SELECT MAX(id) FROM pegawai")
    last_id = cursor.fetchone()[0]
    new_id = (last_id or 0) + 1

    cursor.execute(f"""
        INSERT INTO pegawai (id, nama, jabatan_id)
        VALUES ({new_id}, '{nama}', {jabatan_id})
    """)
    return jsonify({'message': 'Pegawai created', 'id': new_id}), 201


# READ all pegawai
@bp.route('/pegawai', methods=['GET'])
def get_pegawai():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.nama, j.nama_jabatan
        FROM pegawai p LEFT JOIN jabatan j ON p.jabatan_id = j.id
    """)
    rows = cursor.fetchall()
    return jsonify([{'id': r[0], 'nama': r[1], 'jabatan': r[2]} for r in rows])


# UPDATE pegawai by id
@bp.route('/pegawai/<int:id>', methods=['PUT'])
def update_pegawai(id):
    data = request.get_json()
    nama = data.get('nama')
    jabatan_id = data.get('jabatan_id')

    if not nama or jabatan_id is None:
        return jsonify({'error': 'nama and jabatan_id are required'}), 400

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        UPDATE pegawai
        SET nama = '{nama}', jabatan_id = {jabatan_id}
        WHERE id = {id}
    """)
    return jsonify({'message': 'Pegawai updated'}), 200


# DELETE pegawai by id
@bp.route('/pegawai/<int:id>', methods=['DELETE'])
def delete_pegawai(id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        DELETE FROM pegawai
        WHERE id = {id}
    """)
    return jsonify({'message': 'Pegawai deleted'}), 200


# JABATAN
@bp.route('/jabatan', methods=['POST'])
def create_jabatan():
    data = request.get_json()
    nama_jabatan = data.get('nama_jabatan')

    if not nama_jabatan:
        return jsonify({'error': 'nama_jabatan is required'}), 400

    conn = get_conn()
    cursor = conn.cursor()

    # Dapatkan id terakhir, lalu +1
    cursor.execute("SELECT MAX(id) FROM jabatan")
    last_id = cursor.fetchone()[0]
    new_id = (last_id or 0) + 1  # Jika NULL, mulai dari 1

    cursor.execute(f"""
        INSERT INTO jabatan (id, nama_jabatan)
        VALUES ({new_id}, '{nama_jabatan}')
    """)
    return jsonify({'message': 'Jabatan created', 'id': new_id}), 201

# READ all jabatan
@bp.route('/jabatan', methods=['GET'])
def get_jabatan():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, nama_jabatan FROM jabatan
    """)
    rows = cursor.fetchall()
    return jsonify([{'id': r[0], 'nama_jabatan': r[1]} for r in rows])


# UPDATE jabatan by id
@bp.route('/jabatan/<int:id>', methods=['PUT'])
def update_jabatan(id):
    data = request.get_json()
    nama_jabatan = data.get('nama_jabatan')

    if not nama_jabatan:
        return jsonify({'error': 'nama_jabatan is required'}), 400

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        UPDATE jabatan
        SET nama_jabatan = '{nama_jabatan}'
        WHERE id = {id}
    """)
    return jsonify({'message': 'Jabatan updated'}), 200


# DELETE jabatan by id
@bp.route('/jabatan/<int:id>', methods=['DELETE'])
def delete_jabatan(id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        DELETE FROM jabatan
        WHERE id = {id}
    """)
    return jsonify({'message': 'Jabatan deleted'}), 200
