import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load model & vectorizer
model = joblib.load("model/bernoulli_nb.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stemmer.stem(text)
    return text

# Mapping label ke sentimen
label_map = {
    0: "NEGATIF",
    1: "NETRAL",
    2: "POSITIF"
}

st.title("Analisis Sentimen Game Roblox 🇮🇩")
st.write("Model: **Bernoulli Naive Bayes + TF-IDF (3 Kelas)**")

user_input = st.text_area("Masukkan teks:")

if st.button("Prediksi"):
    clean_text = preprocess(user_input)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    sentiment = label_map[prediction]

    if sentiment == "POSITIF":
        st.success(f"Sentimen: {sentiment}")
    elif sentiment == "NETRAL":
        st.warning(f"Sentimen: {sentiment}")
    else:
        st.error(f"Sentimen: {sentiment}")
