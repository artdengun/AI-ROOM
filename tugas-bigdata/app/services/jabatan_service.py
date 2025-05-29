# app/services/jabatan_service.py
from flask import jsonify
from app.hive.hive_conn import get_conn

def create_jabatan(nama_jabatan):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM jabatan")
    last_id = cursor.fetchone()[0]
    new_id = (last_id or 0) + 1
    cursor.execute(f"""
        INSERT INTO jabatan (id, nama_jabatan)
        VALUES ({new_id}, '{nama_jabatan}')
    """)
    return new_id

def get_all_jabatan():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, nama_jabatan FROM jabatan")
    return cursor.fetchall()

def update_jabatan(id, nama_jabatan):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        UPDATE jabatan SET nama_jabatan='{nama_jabatan}'
        WHERE id={id}
    """)


def delete_jabatan(id):
    try:
        id = int(id)  # Validasi manual untuk mencegah injection
    except ValueError:
        return jsonify({'error': 'Invalid ID'}), 400

    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM jabatan WHERE id = {id}")
    return jsonify({'message': 'Jabatan deleted'}), 200