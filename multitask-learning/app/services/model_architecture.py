import torch
import torch.nn as nn
from transformers import AutoModel

class IndoBERTMultitask(nn.Module):
    def __init__(self, bert_name="indobenchmark/indobert-base-p1"):
        super(IndoBERTMultitask, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden_size = self.bert.config.hidden_size

        # Output head untuk masing-masing task
        self.sentiment_head = nn.Linear(hidden_size, 3)  # Negatif, Netral, Positif
        self.emotion_head = nn.Linear(hidden_size, 6)    # Marah, Sedih, Senang, Takut, Jijik, Terkejut
        self.fakeness_head = nn.Linear(hidden_size, 2)   # Asli, Palsu

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        sent_logits = self.sentiment_head(pooled_output)
        emo_logits = self.emotion_head(pooled_output)
        fake_logits = self.fakeness_head(pooled_output)

        return sent_logits, emo_logits, fake_logits