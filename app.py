import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("triage_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def confidence_level(c):
    if c >= 0.85:
        return "🟢 High"
    elif c >= 0.65:
        return "🟠 Medium"
    else:
        return "🔴 Low"

# App layout
st.set_page_config(page_title="NLP Message Insights", layout="wide")
st.title("📊 NLP Healthcare Message Insights Dashboard")

# Sidebar info
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    "This tool uses NLP to classify healthcare-related messages into categories like:\n"
    "- Refill Request\n"
    "- Appointment Inquiry\n"
    "- Complaint\n\n"
    "Useful for clinics/pharmacies to auto-triage and manage message loads.\n\n"
    "🟢 High Confidence = very likely accurate\n"
    "🟠 Medium = worth reviewing\n"
    "🔴 Low = manual check recommended"
)

mode = st.radio("Choose Mode:", ["🔹 Single Message", "📤 Upload CSV"])

# --- Single message mode ---
if mode == "🔹 Single Message":
    examples = [
        "",
        "Can I refill my diabetes meds?",
        "Is Dr. Mehta available on Friday?",
        "Why are my pills late again?",
        "Thank you for the reminder!",
        "The pharmacy was closed again!",
        "Can you send my prescription to another pharmacy?",
        "Hi",
        "How long will delivery take?"
    ]
    sample = st.selectbox("🧪 Try a sample message:", examples)
    message = st.text_area("✍️ Or enter your own message:", value=sample, height=150)

    if st.button("🔍 Predict"):
        if message.strip() == "":
            st.warning("⚠️ Please enter or select a message.")
        else:
            cleaned = clean_text(message)
            X = vectorizer.transform([cleaned])
            prediction = model.predict(X)[0]
            confidence = model.predict_proba(X).max()
            level = confidence_level(confidence)

            st.success(f"📌 Predicted Tag: **{prediction}**")
            st.metric("📊 Confidence Score", f"{confidence*100:.1f}%", delta=level)

            if confidence < 0.5:
                st.warning("⚠️ Low confidence — message may need manual review.")

# --- Bulk CSV mode ---
else:
    uploaded_file = st.file_uploader("📥 Upload a CSV with a 'message' column", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'message' not in df.columns:
            st.error("CSV must contain a 'message' column.")
        else:
            df['cleaned'] = df['message'].astype(str).apply(clean_text)
            X = vectorizer.transform(df['cleaned'])
            df['Predicted Tag'] = model.predict(X)
            df['Confidence'] = model.predict_proba(X).max(axis=1)
            df['Confidence Level'] = df['Confidence'].apply(confidence_level)

            st.success(f"✅ Processed {len(df)} messages")
            st.dataframe(df[['message', 'Predicted Tag', 'Confidence', 'Confidence Level']])

            st.subheader("📌 Message Tag Distribution")
            tag_counts = df['Predicted Tag'].value_counts()
            fig, ax = plt.subplots()
            tag_counts.plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Tagged Messages", data=csv, file_name="tagged_messages.csv", mime="text/csv")
