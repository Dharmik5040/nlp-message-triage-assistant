import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("triage_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Confidence color/label
def confidence_level(score):
    if score >= 0.85:
        return "🟢 High"
    elif score >= 0.65:
        return "🟠 Medium"
    else:
        return "🔴 Low"

# Page setup
st.set_page_config(page_title="NLP Message Triage Assistant", page_icon="📬")

# Title
st.title("📬 NLP Message Triage Assistant")
st.write("This assistant classifies patient/staff messages into categories like refill request, appointment, or complaint.")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
This app uses an NLP model to help clinics/pharmacies sort incoming messages by intent.

**Confidence** is the model's certainty:
- 🟢 High → confident prediction
- 🟠 Medium → likely but review
- 🔴 Low → manual review advised
    """
)

# Sample message options
samples = [
    "",
    "Can I refill my diabetes meds?",
    "The pharmacy was closed again!",
    "Is Dr. Patel available Monday?",
    "Thanks for the reminder!",
    "Why are my pills always late?"
]
sample = st.selectbox("🧪 Try a sample message:", samples)
message = st.text_area("✍️ Enter or paste message:", value=sample, height=150)

# Predict button
if st.button("🔍 Predict Category"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()
        level = confidence_level(confidence)

        st.success(f"📌 Predicted Tag: **{prediction}**")
        st.metric("📊 Confidence Score", f"{confidence*100:.1f}%", delta=level)
        st.caption("Confidence = how sure the model is. Use it to automate, flag, or review intelligently.")
