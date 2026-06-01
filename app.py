import streamlit as st
import pandas as pd
import re
import joblib

st.title("🚀 Sentiment Analysis Dashboard")

# LOAD MODEL
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# SINGLE PREDICTION
st.header("🔮 Single Prediction")

user_text = st.text_input("Enter text:")

def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    mapping = {
        0: "Negative 😞",
        1: "Neutral 😐",
        2: "Positive 😊"
    }

    return mapping[pred]

if st.button("Predict"):
    if user_text:
        st.success(predict(user_text))

# FILE UPLOAD
st.header("📂 Batch Prediction")

file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    if "text" not in df.columns:
        st.error("CSV must have 'text' column")
    else:
        df["clean"] = df["text"].apply(clean_text)

        X = vectorizer.transform(df["clean"])
        preds = model.predict(X)

        mapping = {
            0: "Negative 😞",
            1: "Neutral 😐",
            2: "Positive 😊"
        }

        df["sentiment"] = [mapping[p] for p in preds]

        st.write("## Batch Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Batch Predictions")
            st.dataframe(df, height=400)

        
        with col2:
            st.write("### Sentiment Distribution")
            st.bar_chart(df["sentiment"].value_counts())
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Reviews", len(df))
            col2.metric("Positive 😊", sum(df["sentiment"] == "Positive 😊"))
            col3.metric("Negative 😞", sum(df["sentiment"] == "Negative 😞"))
            col4.metric("Neutral 😐", sum(df["sentiment"] == "Neutral 😐"))