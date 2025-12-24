import streamlit as st
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize preprocessing tools
@st.cache_resource
def load_preprocessing_tools():
    stemmer = StemmerFactory().create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopwords = set(stopword_factory.get_stop_words())
    lemmatizer = WordNetLemmatizer()
    return stemmer, stopwords, lemmatizer

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model_naive_bayes.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    return model, tfidf

# Preprocessing function
def preprocessing(text, stemmer, stopwords, lemmatizer):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stopwords]
    stemmed = [stemmer.stem(w) for w in tokens]
    lemmatized = [lemmatizer.lemmatize(w) for w in stemmed]
    return ' '.join(lemmatized)

# Load resources
stemmer, stopwords, lemmatizer = load_preprocessing_tools()
model, tfidf = load_model_and_vectorizer()

# Streamlit UI
st.title("Sentiment Analysis App")
st.write(
    """
    Aplikasi ini menggunakan model Multinomial Naive Bayes yang dilatih pada data teks berbahasa Indonesia untuk memprediksi sentimen (positif, negatif, atau netral).

    **Model Details:**
    - Algorithm: Multinomial Naive Bayes
    - Features: TF-IDF
    - Preprocessing: Tokenization, Stemming, Lemmatization
    """
)
# Text input
user_input = st.text_area(
    "Masukkan text disini:",
    height=150,
    placeholder="Contoh: Produk ini sangat bagus dan berkualitas..."
)

# Predict button
if st.button("Prediksi", type="primary"):
    if user_input.strip():
        # Preprocess input
        processed_text = preprocessing(user_input, stemmer, stopwords, lemmatizer)

        # Transform and predict
        X_input = tfidf.transform([processed_text])
        prediction = model.predict(X_input)[0]

        # Display result with color coding
        st.subheader("Hasil Prediksi:")

        if prediction.lower() == "positive":
            st.success(f"Sentiment: **{prediction.upper()}**")
        elif prediction.lower() == "negative":
            st.error(f"Sentiment: **{prediction.upper()}**")
        else:  # neutral
            st.info(f"Sentimen : **{prediction.upper()}**")
    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.title("Contoh Text")
st.sidebar.write("**Positive:**")
st.sidebar.text("Good morning semangat kerja hari moga kasih rezeki berkah sekolah kuliah onlen hari semangat moga ilmu tetap jalan berkah")

st.sidebar.write("**Negative:**")
st.sidebar.text("Benci banget kuliah onlen")

st.sidebar.write("**Neutral:**")
st.sidebar.text("Bangun pagi kuliah onlen")
