import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("triage_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Message Triage Assistant", page_icon="📬")
st.title("📬 NLP Message Triage Assistant")
st.write("This assistant classifies incoming messages into predefined categories such as refill requests, complaints, or appointment queries.")

# Message input
user_input = st.text_area("Paste or type a message:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess and predict
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec).max()

        # Show result
        st.subheader("🧠 Prediction")
        st.success(f"Tag: `{prediction}`")
        st.metric("Confidence", f"{proba * 100:.2f}%")

        # Optional summary for download
        df = pd.DataFrame({
            "Message": [user_input],
            "Predicted Tag": [prediction],
            "Confidence (%)": [f"{proba * 100:.2f}"]
        })
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Result", data=csv, file_name="triage_prediction.csv", mime="text/csv")
