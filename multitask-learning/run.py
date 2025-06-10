from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# === LABELS ===
SENTIMENT_LABELS = ["Negatif", "Netral", "Positif"]
EMOTION_LABELS = ["Marah", "Sedih", "Senang", "Takut", "Jijik", "Terkejut"]
FAKE_LABELS = ["Asli", "Palsu"]

# === MODEL MULTITASK ===
class MultiTaskIndoBERT(nn.Module):
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.sentiment_head = nn.Linear(hidden_size, 3)
        self.emotion_head = nn.Linear(hidden_size, 6)
        self.fake_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        sentiment = self.sentiment_head(cls_output)
        emotion = self.emotion_head(cls_output)
        fake = self.fake_head(cls_output)
        return sentiment, emotion, fake

# === LOAD TOKENIZER & MODEL ===
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = MultiTaskIndoBERT()
model.eval()  # Model belum dilatih, hanya demo

# === INPUT TEXT ===
text = "Saya sangat kecewa dengan pelayanan di restoran ini."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# === INFERENSI ===
with torch.no_grad():
    sent_logits, emo_logits, fake_logits = model(**inputs)

    sent_pred = torch.argmax(F.softmax(sent_logits, dim=1), dim=1).item()
    emo_pred = torch.argmax(F.softmax(emo_logits, dim=1), dim=1).item()
    fake_pred = torch.argmax(F.softmax(fake_logits, dim=1), dim=1).item()

# === HASIL KLASIFIKASI ===
print("Input Teks:", text)
print("Prediksi Sentimen:", SENTIMENT_LABELS[sent_pred])
print("Prediksi Emosi:", EMOTION_LABELS[emo_pred])
print("Ulasan:", FAKE_LABELS[fake_pred])
