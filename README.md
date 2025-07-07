# 📬 NLP Message Triage Assistant

This is a Streamlit app that classifies incoming messages (from patients, staff, or systems) into actionable categories such as `Refill Request`, `Complaint`, `Appointment Query`, and more. Built with real NLP logic, it simulates how modern healthcare systems can automate message triage for faster and smarter responses.

🔗 **Live App**: [Try it now](https://nlp-message-triage-assistant-demo.streamlit.app/)  

---

## 💡 Use Case

This tool helps clinics, pharmacies, or care managers automatically categorize messages to:
- Reduce manual effort
- Speed up patient communication
- Triage high-priority requests faster

---

## 🛠️ Features

- 📝 Message input area
- 🤖 ML-powered intent prediction
- 📈 Confidence score output
- 📥 Download result as CSV
- 💬 Real NLP: lowercasing, stopwords, TF-IDF

---

## 📂 Files in This Repo

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI for classification |
| `triage_model.pkl` | Trained logistic regression model |
| `tfidf_vectorizer.pkl` | Vectorizer for converting text to features |
| `requirements.txt` | Libraries for Streamlit Cloud |

---

## 🤖 Model

- **Type**: Logistic Regression (scikit-learn)
- **Input**: Cleaned text messages
- **Text Vectorization**: TF-IDF
- **Training data**: Simulated real-world messages (refill, complaint, appointment, etc.)
- **Accuracy**: ~80% on test set

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
