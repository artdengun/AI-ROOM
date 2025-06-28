SENTIMENT_LABELS = ["Negatif", "Netral", "Positif"]
EMOTION_LABELS = ["Marah", "Sedih", "Senang", "Takut", "Jijik", "Terkejut"]
FAKE_LABELS = ["Asli", "Palsu"]

def decode_logits(s, e, f):
    return {
        "sentiment": SENTIMENT_LABELS[s.argmax().item()],
        "emotion": EMOTION_LABELS[e.argmax().item()],
        "fakeness": FAKE_LABELS[f.argmax().item()]
    }