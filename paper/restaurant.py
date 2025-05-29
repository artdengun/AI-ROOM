import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import plotly.io as pio
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from bertopic import BERTopic
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from textblob import TextBlob
from collections import Counter
from gensim.models import KeyedVectors

# Pastikan Plotly bisa menampilkan grafik
pio.renderers.default = "colab"
nltk.download('stopwords')
nltk.download('wordnet')

# Upload dataset
df = pd.read_csv("data-review-richeese.csv")  # Pastikan file ada di folder kerja

# Membersihkan teks dengan stopwords dan lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned_review'] = df['Review'].apply(clean_text)

# Analisis Sentimen dengan TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['cleaned_review'].apply(get_sentiment)

# Distribusi rating
plt.figure(figsize=(6,4))
sns.countplot(x=df['Rating'], palette='viridis')
plt.title("Distribusi Rating")
plt.show()

# Distribusi Sentimen
plt.figure(figsize=(6,4))
sns.countplot(x=df['Sentiment'], palette='coolwarm')
plt.title("Distribusi Sentimen")
plt.show()

# Tokenisasi & Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_review'])
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
padded_sequences = pad_sequences(sequences, padding='post')

Embedding(input_dim=len(tokenizer.word_index)+1,
          output_dim=100,
          input_length=padded_sequences.shape[1],
          trainable=True)

# Bagi dataset untuk BiLSTM
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['Rating'], test_size=0.2, random_state=42)
y_train, y_test = np.array(y_train), np.array(y_test)
embedding_dim = 100  # misal glove.6B.100d.txt

# Load GloVe embeddings ke dictionary
embedding_index = {}
with open(r"C:\nlp_data\glove.6B.100d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Buat embedding matrix dari tokenizer.word_index dan embedding_index
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Buat model BiLSTM dengan GloVe embeddings
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1,
              output_dim=embedding_dim,
              weights=[embedding_matrix],
              input_length=padded_sequences.shape[1],
              trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.6),
    Dense(6, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Jalankan BiLSTM dengan EarlyStopping dan Learning Rate Scheduling
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])

# Prediksi BiLSTM
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(y_test, predicted_classes))

# Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, predicted_classes), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.show()

# BERTopic untuk Analisis Topik Lebih Lanjut
bertopic = BERTopic()
topics, _ = bertopic.fit_transform(df['cleaned_review'])
df['Topic'] = topics

# Visualisasi Topik BERTopic
fig = bertopic.visualize_barchart()
fig.show()

# Analisis Kata Umum dalam Ulasan Negatif dan Positif
neg_reviews = df[df['Rating'] <= 2]['cleaned_review']
pos_reviews = df[df['Rating'] >= 4]['cleaned_review']

neg_keywords = ' '.join(neg_reviews).split()
pos_keywords = ' '.join(pos_reviews).split()

common_neg_words = pd.Series(neg_keywords).value_counts().head(10).index.tolist()
common_pos_words = pd.Series(pos_keywords).value_counts().head(10).index.tolist()

print("\n📌 Kesimpulan & Rekomendasi berdasarkan data:")
print(f"1. Mayoritas ulasan negatif terkait dengan: {', '.join(common_neg_words)}.")
print(f"2. Ulasan positif sering menyebutkan: {', '.join(common_pos_words)}.")
print("3. Berdasarkan analisis topik, pihak Richeese bisa meningkatkan kepuasan pelanggan dengan fokus pada area yang sering disebut di ulasan negatif.")

