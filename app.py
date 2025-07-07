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

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# App layout
st.set_page_config(page_title="📊 Message Insights Dashboard", layout="wide")
st.title("📊 NLP Healthcare Message Insights Dashboard")
st.write("Upload a CSV file with a column `message` to classify and analyze messages automatically.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'message' not in df.columns:
        st.error("CSV must contain a 'message' column.")
    else:
        # Process messages
        df['cleaned'] = df['message'].astype(str).apply(clean_text)
        X = vectorizer.transform(df['cleaned'])
        df['Predicted Tag'] = model.predict(X)
        df['Confidence'] = model.predict_proba(X).max(axis=1)

        # Add emoji confidence levels
        def confidence_level(c):
            if c >= 0.85:
                return "🟢 High"
            elif c >= 0.65:
                return "🟠 Medium"
            else:
                return "🔴 Low"

        df['Confidence Level'] = df['Confidence'].apply(confidence_level)

        # Show result table
        st.success(f"✅ Processed {len(df)} messages")
        st.dataframe(df[['message', 'Predicted Tag', 'Confidence', 'Confidence Level']])

        # Pie chart
        st.subheader("📌 Message Tag Distribution")
        tag_counts = df['Predicted Tag'].value_counts()
        fig, ax = plt.subplots()
        tag_counts.plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Tagged Messages", data=csv, file_name="tagged_messages.csv", mime="text/csv")
