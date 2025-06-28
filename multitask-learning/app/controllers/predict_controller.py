from flask import Blueprint, request, jsonify
from app.utils.inference import infer
from app.services.predict_service import save_prediction

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    result = infer(text)
    save_prediction(text, **result)
    return jsonify(result)