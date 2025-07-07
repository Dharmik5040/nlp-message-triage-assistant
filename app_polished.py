
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("triage_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.set_page_config(page_title="NLP Message Triage Assistant", page_icon="📬")

st.title("📬 NLP Message Triage Assistant")
st.write("Paste or type a patient/staff message below. The assistant will predict its category (e.g., Refill, Appointment, Complaint, etc.)")

# Sidebar
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    "This AI assistant reads incoming healthcare-related messages and automatically tags them using NLP.

"
    "It helps clinics and pharmacies triage inquiries faster by classifying intent such as `Refill Request`, `Appointment`, or `Complaint`."
)

# Sample dropdown
examples = [
    "",
    "Can I refill my diabetes meds?",
    "The pharmacy is closed again!",
    "Is Dr. Patel available on Monday?",
    "Thanks for the reminder!",
    "Why are my pills always late?"
]
sample = st.selectbox("🧪 Try a sample message:", examples)
message = st.text_area("✍️ Enter Message", value=sample, height=150)

# Prediction
if st.button("🔍 Predict Category"):
    if message.strip() == "":
        st.warning("⚠️ Please enter or select a message.")
    else:
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector).max()

        st.success(f"📌 Predicted Tag: **{prediction}**")
        st.metric("📊 Confidence", f"{proba*100:.1f}%")
