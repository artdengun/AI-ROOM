import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
print("Versi PyTorch:", torch.__version__)


if torch.cuda.is_available():
    print("CUDA tersedia!")
    print("Nama GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA tidak tersedia. Menggunakan CPU.")
    
    
# Cek apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Menggunakan device:", device)

# Load model dan tokenizer
model_name = "taufiqdp/indonesian-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Tes input
text = "Saya sangat senang dengan pelayanan aplikasi ini."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

# Prediksi
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1)

print("Prediksi kelas:", pred.item())
