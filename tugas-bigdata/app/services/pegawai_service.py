# app/services/pegawai_service.py
from app.hive.hive_conn import get_conn

def create_pegawai(nama, jabatan_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM pegawai")
    last_id = cursor.fetchone()[0]
    new_id = (last_id or 0) + 1
    cursor.execute(f"""
        INSERT INTO pegawai (id, nama, jabatan_id)
        VALUES ({new_id}, '{nama}', {jabatan_id})
    """)
    return new_id

def get_all_pegawai():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.nama, j.nama_jabatan
        FROM pegawai p LEFT JOIN jabatan j ON p.jabatan_id = j.id
    """)
    return cursor.fetchall()

def update_pegawai(id, nama, jabatan_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        UPDATE pegawai SET nama='{nama}', jabatan_id={jabatan_id}
        WHERE id={id}
    """)

def delete_pegawai(id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM pegawai WHERE id={id}")
