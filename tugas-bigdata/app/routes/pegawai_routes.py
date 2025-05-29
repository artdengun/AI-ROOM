from flask import Blueprint, request, jsonify
from app.services.pegawai_service import (
    create_pegawai, 
    get_all_pegawai, 
    update_pegawai, 
    delete_pegawai
)

pegawai_bp = Blueprint('pegawai', __name__)

@pegawai_bp.route('/pegawai', methods=['POST'])
def add_pegawai():
    data = request.get_json()
    nama = data.get('nama')
    jabatan_id = data.get('jabatan_id')
    if not nama or jabatan_id is None:
        return jsonify({'error': 'nama and jabatan_id are required'}), 400
    new_id = create_pegawai(nama, jabatan_id)
    return jsonify({'message': 'Pegawai created', 'id': new_id}), 201

@pegawai_bp.route('/pegawai', methods=['GET'])
def list_pegawai():
    rows = get_all_pegawai()
    return jsonify([{'id': r[0], 'nama': r[1], 'jabatan': r[2]} for r in rows])

@pegawai_bp.route('/pegawai/<int:id>', methods=['PUT'])
def edit_pegawai(id):
    data = request.get_json()
    nama = data.get('nama')
    jabatan_id = data.get('jabatan_id')
    if not nama or jabatan_id is None:
        return jsonify({'error': 'nama and jabatan_id are required'}), 400
    update_pegawai(id, nama, jabatan_id)
    return jsonify({'message': 'Pegawai updated'})

@pegawai_bp.route('/pegawai/<int:id>', methods=['DELETE'])
def remove_pegawai(id):
    delete_pegawai(id)
    return jsonify({'message': 'Pegawai deleted'})
