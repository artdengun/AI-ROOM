from flask import Blueprint, request, jsonify
from app.services.jabatan_service import (
    create_jabatan,
    get_all_jabatan,
    update_jabatan,
    delete_jabatan
)

jabatan_bp = Blueprint('jabatan', __name__)

# CREATE jabatan
@jabatan_bp.route('/jabatan', methods=['POST'])
def add_jabatan():
    data = request.get_json()
    nama_jabatan = data.get('nama_jabatan')

    if not nama_jabatan:
        return jsonify({'error': 'nama_jabatan is required'}), 400

    new_id = create_jabatan(nama_jabatan)
    return jsonify({'message': 'Jabatan created', 'id': new_id}), 201

# READ all jabatan
@jabatan_bp.route('/jabatan', methods=['GET'])
def list_jabatan():
    rows = get_all_jabatan()
    return jsonify([{'id': r[0], 'nama_jabatan': r[1]} for r in rows])

# UPDATE jabatan by id
@jabatan_bp.route('/jabatan/<int:id>', methods=['PUT'])
def edit_jabatan(id):
    data = request.get_json()
    nama_jabatan = data.get('nama_jabatan')

    if not nama_jabatan:
        return jsonify({'error': 'nama_jabatan is required'}), 400

    update_jabatan(id, nama_jabatan)
    return jsonify({'message': 'Jabatan updated'}), 200

# DELETE jabatan by id
@jabatan_bp.route('/jabatan/<int:id>', methods=['DELETE'])
def remove_jabatan(id):
    delete_jabatan(id)
    return jsonify({'message': 'Jabatan deleted'}), 200
