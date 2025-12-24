import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())
lemmatizer = WordNetLemmatizer()

# ============================
# 2. Load Dataset from CSV
# ============================
df = pd.read_csv('clean.csv')
# Rename columns to match the expected format
df.columns = ['ulasan', 'sentimen']


# ============================
# 3. Preprocessing
# ============================
def preprocessing(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Tokenizing
    tokens = word_tokenize(text)

    # 3. Filtering (hapus stopwords & non-alphabetic)
    tokens = [w for w in tokens if w.isalpha() and w not in stopwords]

    # 4. Stemming
    stemmed = [stemmer.stem(w) for w in tokens]

    # 5. Lemmatization
    lemmatized = [lemmatizer.lemmatize(w) for w in stemmed]

    # Gabungkan kembali
    return " ".join(lemmatized)

df["ulasan"] = df["ulasan"].apply(preprocessing)

# ============================
# 4. TF-IDF Vectorization
# ============================
X = df["ulasan"]
y = df["sentimen"]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)

# ============================
# 5. Sample Hasil TF-IDF
# ============================

fitur = tfidf.get_feature_names_out()
df_tfidf = pd.DataFrame(X.toarray(), columns=fitur)

print("\n=== SAMPLE HASIL TF-IDF (5 baris × 10 kolom) ===")
print(df_tfidf.iloc[:5, :10])

# ============================
# 6. Training & Testing Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# ============================
# 7. Model Naive Bayes
# ============================
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and TF-IDF vectorizer
with open('model_naive_bayes.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model and TF-IDF vectorizer saved successfully!")

# ============================
# 8. Prediksi & Evaluasi
# ============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Ambil index data training (indeks asli dari DataFrame)
train_idx = y_train.index

# Prediksi data training
y_train_pred = model.predict(X_train)

# Buat tabel hasil prediksi training
df_training = pd.DataFrame({
    "Ulasan Training": df.loc[train_idx, "ulasan"].values,
    "Label Asli": y_train.values,
    "Prediksi Sentimen": y_train_pred
})

print("Akurasi :", accuracy)
print("\n=== Hasil Prediksi Data Training ===")
print(df_training)


# ============================
# 9. Data Test Testing
# ============================

# Load saved model and TF-IDF vectorizer
print("\n=== Loading saved model for testing ===")
with open('model_naive_bayes.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)

print("Model and TF-IDF vectorizer loaded successfully!")

data_baru = [
    "Produk ini sangat mengecewakan",
    "Saya puas dengan kualitas barangnya",
    "Barang ini biasa saja menurut saya",
    "Tidak sesuai keinginan"
]

# Preprocess new data
data_baru_processed = [preprocessing(text) for text in data_baru]

# Transform and predict using loaded model
X_new = loaded_tfidf.transform(data_baru_processed)
y_new_pred = loaded_model.predict(X_new)

# Buat tabel output
df_hasil = pd.DataFrame({
    "Ulasan Baru": data_baru,
    "Prediksi Sentimen": y_new_pred
})

print("\n=== Hasil Prediksi Data Testing ===")
print(df_hasil)
