import torch
from app.utils.model_utils import load_model
from app.utils.preprocessing import preprocess
from app.components.label_config import decode_logits

model, tokenizer = load_model()

def infer(text):
    inputs = preprocess(text, tokenizer)
    with torch.no_grad():
        s, e, f = model(**inputs)
    return decode_logits(s, e, f)