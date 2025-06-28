import torch
import os
from transformers import AutoTokenizer
from app.services.model_architecture import IndoBERTMultitask

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = IndoBERTMultitask()
    model.load_state_dict(torch.load(os.getenv("MODEL_PATH"), map_location="cpu"))
    model.eval()
    return model, tokenizer