import streamlit as st
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Try to load fine-tuned model first
        model = AutoModelForSequenceClassification.from_pretrained("./indobert_sentiment_model")
        tokenizer = AutoTokenizer.from_pretrained("./indobert_sentiment_model")

        # Try to load label mapping
        try:
            with open('label_map.pkl', 'rb') as f:
                label_maps = pickle.load(f)
                label_map_reverse = label_maps['label_map_reverse']
        except:
            # Default label mapping if file doesn't exist
            label_map_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}
            st.warning("Using default label mapping. Run app2.py to create custom mapping.")

        return model, tokenizer, label_map_reverse
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run app2.py first to train and save the IndoBERT model.")
        return None, None, None

# Prediction function
def predict_sentiment(text, model, tokenizer, label_map_reverse):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        # Get label with fallback
        if predicted_class in label_map_reverse:
            predicted_label = label_map_reverse[predicted_class]
        else:
            # Fallback mapping
            fallback_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            predicted_label = fallback_map.get(predicted_class, 'unknown')

        # Get confidence scores
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        confidence = probabilities[predicted_class].item()

    return predicted_label, confidence, probabilities

# Load resources
model, tokenizer, label_map_reverse = load_model_and_tokenizer()

# Streamlit UI
st.title("Sentiment Analysis App (IndoBERT)")
st.write(
    """
    Aplikasi ini menggunakan model **IndoBERT** (BERT yang dilatih pada bahasa Indonesia)
    untuk memprediksi sentimen (positif, negatif, atau netral).

    **Model Details:**
    - Algorithm: IndoBERT (Transformer-based)
    - Model: indobenchmark/indobert-base-p1
    - Fine-tuned on Indonesian sentiment data
    """
)

# Check if model is loaded
if model is None:
    st.stop()

# Text input
user_input = st.text_area(
    "Masukkan text disini:",
    height=150,
    placeholder="Contoh: Produk ini sangat bagus dan berkualitas..."
)

# Predict button
if st.button("Prediksi", type="primary"):
    if user_input.strip():
        with st.spinner('Memprediksi sentimen...'):
            # Predict
            prediction, confidence, probabilities = predict_sentiment(
                user_input, model, tokenizer, label_map_reverse
            )

            # Display result with color coding
            st.subheader("Hasil Prediksi:")

            if prediction.lower() == "positive":
                st.success(f"Sentimen: **{prediction.upper()}**")
            elif prediction.lower() == "negative":
                st.error(f"Sentimen: **{prediction.upper()}**")
            else:  # neutral
                st.info(f"Sentimen: **{prediction.upper()}**")

            # Show confidence
            st.metric("Confidence", f"{confidence*100:.2f}%")

            # Show all probabilities
            with st.expander("Lihat detail probabilitas"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Negative", f"{probabilities[0].item()*100:.2f}%")
                with col2:
                    st.metric("Neutral", f"{probabilities[1].item()*100:.2f}%")
                with col3:
                    st.metric("Positive", f"{probabilities[2].item()*100:.2f}%")
    else:
        st.warning("Silakan masukkan teks untuk dianalisis.")

# Sidebar information
st.sidebar.title("About IndoBERT")
st.sidebar.info(
    """
    **IndoBERT** adalah model BERT yang telah dilatih khusus pada teks bahasa Indonesia.

    Model ini menggunakan arsitektur Transformer yang dapat memahami konteks
    kalimat dengan lebih baik dibandingkan metode tradisional.

    **Keunggulan:**
    - Pemahaman konteks yang lebih baik
    - Akurasi lebih tinggi
    - Pre-trained pada bahasa Indonesia
    - State-of-the-art performance
    """
)

st.sidebar.title("Contoh Text")
st.sidebar.write("**Positive:**")
st.sidebar.text("Good morning semangat kerja\nhari moga kasih rezeki berkah\nsekolah kuliah onlen hari\nsemangat moga ilmu tetap\njalan berkah")

st.sidebar.write("**Negative:**")
st.sidebar.text("Benci banget kuliah onlen\nsangat membosankan")

st.sidebar.write("**Neutral:**")
st.sidebar.text("Bangun pagi kuliah onlen\nseperti biasa")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Powered by IndoBERT 🤖")
