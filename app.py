import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load models
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("sentiment_model.pkl", "rb"))

# Emotion model (light CPU version)
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=-1
)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    
    processed = []
    i = 0
    
    while i < len(tokens):
        if tokens[i] == "not" and i+1 < len(tokens):
            processed.append("not_" + tokens[i+1])
            i += 2
        else:
            if tokens[i] not in stop_words:
                processed.append(lemmatizer.lemmatize(tokens[i]))
            i += 1
            
    return " ".join(processed)

# UI
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🛍️ Product Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("### Analyze Sentiment & Emotion using NLP")

review = st.text_area("✍️ Enter your review:")

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = preprocess(review)
        vector = tfidf.transform([cleaned])

        # Sentiment
        sentiment = model.predict(vector)[0]
        probs = model.predict_proba(vector)
        confidence = round(max(probs[0]), 2)

        # Emotion
        emotions = emotion_model(cleaned[:256], top_k=3)
        final_emotion = emotions[0]['label']

        # Smart correction
        if sentiment == "negative":
            for emo in emotions:
                if emo['label'] in ["anger", "sadness", "fear"]:
                    final_emotion = emo['label']
                    break
        elif sentiment == "positive":
            for emo in emotions:
                if emo['label'] in ["joy", "love"]:
                    final_emotion = emo['label']
                    break

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment")
            if sentiment == "positive":
                st.success(f"😊 {sentiment.upper()} (Confidence: {confidence})")
            elif sentiment == "negative":
                st.error(f"😡 {sentiment.upper()} (Confidence: {confidence})")
            else:
                st.warning(f"😐 {sentiment.upper()} (Confidence: {confidence})")

        with col2:
            st.subheader("🎭 Emotion")
            st.info(f"{final_emotion}")

        # Chart
        labels = [e['label'] for e in emotions]
        scores = [e['score'] for e in emotions]

        fig, ax = plt.subplots()
        ax.bar(labels, scores)
        ax.set_ylim(0,1)
        st.pyplot(fig)

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.info("""
NLP Project:
- Sentiment Analysis (TF-IDF + Logistic Regression)
- Emotion Detection (Transformer Model)
- Negation Handling (not good → negative)

Built by Anagha 🚀
""")