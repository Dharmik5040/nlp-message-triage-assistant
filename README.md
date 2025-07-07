# 📬 NLP Message Triage Assistant

This Streamlit app uses a trained NLP model to classify incoming messages (from patients or staff) into categories like `Refill Request`, `Appointment Inquiry`, or `Complaint`. It's designed as a practical AI assistant for healthcare or pharmacy operations.

🔗 **Live App**: [Try it now](https://nlp-message-triage-assistant-demo.streamlit.app/)

---

## 🧠 Problem Statement

Clinics and pharmacies receive hundreds of daily messages. Manual triage slows down responses and wastes time. This assistant uses natural language processing to automatically tag messages and help staff prioritize them efficiently.

---

## 💾 Example Inputs

| Message | Predicted Tag |
|--------|----------------|
| “Can I refill my BP meds today?” | Refill Request |
| “Is the doctor available tomorrow?” | Appointment Inquiry |
| “You sent the wrong meds again.” | Complaint |

---

## ⚙️ How It Works

- Cleans and processes text using NLTK
- Converts messages to TF-IDF vectors
- Predicts message category using Logistic Regression
- Shows confidence score
- Downloadable CSV of prediction result

---

## 🧠 Model Details

- Model: Logistic Regression
- Features: TF-IDF vectorization of message text
- Accuracy: ~80% (train/test split on sample dataset)
- Dataset: Simulated healthcare/staff message dataset (200 messages)

---

## 🛠️ Tech Stack

| Component | Tool |
|----------|------|
| UI | Streamlit |
| Model | Scikit-learn |
| Text Processing | NLTK, Regex |
| Hosting | Streamlit Cloud |
| Deployment | GitHub + Streamlit integration |

---

## 📁 Project Files

├── app.py # Streamlit app script
├── triage_model.pkl # Trained NLP model
├── tfidf_vectorizer.pkl # TF-IDF transformer
├── requirements.txt # App dependencies

yaml
Copy

---

## 🙋‍♂️ Created By

**Dharmik Shah**  
Healthcare + Pharmacy + AI + Product Mindset  
📫 dharmik5040@gmail.com  
🔗 [LinkedIn]((https://www.linkedin.com/in/dharmikshah4/)) 

---

## 🔮 Future Ideas

- Support multi-language triage  
- Train on real-world datasets from hospitals  
- Add message priority score and queue visualization
