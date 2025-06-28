import pandas as pd # type: ignore
import numpy as np # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import re
import plotly.io as pio # type: ignore
import nltk # type: ignore
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import ticker # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from bertopic import BERTopic # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from textblob import TextBlob # type: ignore

# Pastikan Plotly bisa menampilkan grafik
pio.renderers.default = "colab"
nltk.download('stopwords')
nltk.download('wordnet')

# Upload dataset
df = pd.read_csv("data_clean.csv")

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
df['Rating'] = df['Rating'].astype(int)
ax = sns.countplot(x=df['Rating'], 
                  palette='viridis',
                  order=sorted(df['Rating'].unique()))
ax.set_xticklabels([str(int(x)) for x in sorted(df['Rating'].unique())])

plt.title("Distribusi Rating", fontsize=14, pad=20)
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Jumlah", fontsize=12)

# Tambahkan anotasi
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
               (p.get_x() + p.get_width()/2., p.get_height()),
               ha='center', va='center',
               xytext=(0, 5),
               textcoords='offset points',
               fontsize=10)

# Hilangkan border yang tidak perlu
sns.despine()
plt.tight_layout()
plt.show()

# Distribusi Sentimen

# 1. First clean and convert the Rating column
print("Nilai unik Rating sebelum pembersihan:", df['Rating'].unique())

# Convert to numeric (handles strings and other formats)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Round and convert to integer
df['Rating'] = df['Rating'].round().astype('Int64')  # Uses pandas' nullable integer type

# Remove any null values and ratings outside 1-5 range
df = df.dropna(subset=['Rating'])
df = df[df['Rating'].between(1, 5)]

print("Nilai unik Rating setelah pembersihan:", df['Rating'].unique())

# 2. Define sentiment mapping
sentimen_mapping = {
    1: 'Sangat Negatif',
    2: 'Negatif',
    3: 'Netral',
    4: 'Positif',
    5: 'Sangat Positif'
}

# Map to sentiment categories
df['Sentimen'] = df['Rating'].map(sentimen_mapping)

# 3. Create color palette
sentimen_palette = {
    'Sangat Negatif': '#ff0000',  # Red
    'Negatif': '#ff6b6b',         # Light red
    'Netral': '#feca57',          # Yellow
    'Positif': '#7ee8fa',         # Light blue
    'Sangat Positif': '#1dd1a1'   # Green
}

# 4. Create the plot
plt.figure(figsize=(10,6))
ax = sns.countplot(
    x=df['Sentimen'],
    palette=sentimen_palette,
    order=['Sangat Negatif', 'Negatif', 'Netral', 'Positif', 'Sangat Positif']
)

# 5. Formatting
plt.title('Distribusi Sentimen', fontsize=16, pad=20)
plt.xlabel('Kategori Sentimen', fontsize=12)
plt.ylabel('Jumlah Review', fontsize=12)
plt.xticks(rotation=15)

# 6. Add annotations
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',  # Ensure integer display
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center',
        va='bottom',
        xytext=(0, 5),
        textcoords='offset points',
        fontsize=10
    )

# Clean up
sns.despine()
plt.tight_layout()

# Show statistics
print("\nDistribusi Sentimen:")
print(df['Sentimen'].value_counts().sort_index())
plt.show()

# Tokenisasi & Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_review'])
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
padded_sequences = pad_sequences(sequences, padding='post')

# Bagi dataset untuk BiLSTM
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, df['Rating'], test_size=0.2, random_state=42
)
y_train, y_test = np.array(y_train), np.array(y_test)

# Hitung distribusi rating
train_rating_counts = pd.Series(y_train).value_counts().sort_index()
test_rating_counts = pd.Series(y_test).value_counts().sort_index()
all_ratings = np.concatenate([y_train, y_test])
rating_counts = pd.Series(all_ratings).value_counts().sort_index()
total_reviews = len(all_ratings)
average_rating = all_ratings.mean()

# ==================== VISUALIZATION ====================

plt.figure(figsize=(15, 10))

# 1. Main Distribution Plot
plt.subplot(2, 2, 1)
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
plt.title('Distribusi Rating Keseluruhan', fontsize=14)
plt.xlabel('Rating')
plt.ylabel('Jumlah Review')
for i, v in enumerate(rating_counts.values):
    plt.text(i, v + 3, str(v), ha='center')

# 2. Training vs Test Comparison
plt.subplot(2, 2, 2)
width = 0.35
x = np.arange(len(train_rating_counts))
plt.bar(x - width/2, train_rating_counts.values, width, label='Training', color='#1f77b4')
plt.bar(x + width/2, test_rating_counts.values, width, label='Testing', color='#ff7f0e')
plt.xticks(x, train_rating_counts.index)
plt.title('Perbandingan Training vs Testing', fontsize=14)
plt.xlabel('Rating')
plt.ylabel('Jumlah Review')
plt.legend()
for i, (train, test) in enumerate(zip(train_rating_counts.values, test_rating_counts.values)):
    plt.text(i - width/2, train + 3, str(train), ha='center')
    plt.text(i + width/2, test + 3, str(test), ha='center')

# 3. Pie Chart
plt.subplot(2, 2, 3)
colors = ['#ff6b6b', '#ffb347', '#feca57', '#7ee8fa', '#1dd1a1']
plt.pie(rating_counts, 
        labels=[f'Rating {i}' for i in rating_counts.index],
        colors=colors[:len(rating_counts)],
        autopct='%1.1f%%',
        startangle=90)
plt.title('Persentase Distribusi', fontsize=14)

# 4. Summary Table
plt.subplot(2, 2, 4)
plt.axis('off')
summary_data = [
    ["Total Reviews", total_reviews],
    ["Average Rating", f"{average_rating:.2f}"],
    ["Most Common", f"Rating {rating_counts.idxmax()}"],
    ["Training Data", len(X_train)],
    ["Testing Data", len(X_test)]
]
table = plt.table(cellText=summary_data,
                 loc='center',
                 cellLoc='left',
                 colWidths=[0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
plt.title('Ringkasan Statistik', fontsize=14)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

# Print the same conclusions
print("\n=== 📝 Kesimpulan ===")
if average_rating >= 4:
    print("Mayoritas pengguna sangat puas terhadap layanan/produk ini.")
elif average_rating >= 3:
    print("Sebagian besar pengguna merasa cukup puas, meskipun ada beberapa keluhan.")
else:
    print("Mayoritas pengguna merasa kurang puas, perlu adanya perbaikan signifikan.")

print(f"\nRating paling banyak diberikan: {rating_counts.idxmax()} ({rating_counts.max()/total_reviews:.1%})")

# Load GloVe Embedding dan buat embedding_matrix
print("Start Matrix Prosess")
embedding_dim = 100
embedding_index = {}

# Pastikan file glove.6B.100d.txt berada di folder yang sama dengan script ini
with open("glove.6B.100d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Buat embedding matrix untuk kata-kata dalam tokenizer
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# Model BiLSTM yang lebih besar dengan Glove Embeddings
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=padded_sequences.shape[1], trainable=False),
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
pio.renderers.default = 'browser'
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

