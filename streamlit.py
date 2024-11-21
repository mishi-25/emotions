import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the saved TF-IDF vectorizer and model
@st.cache_resource
def load_model_and_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

tfidf_vectorizer, model = load_model_and_vectorizer()

# Title and description
st.title("Sentiment Analysis App")
st.write("""
This application predicts the sentiment of text (e.g., joy, anger, fear) based on a trained model.
""")

# User input
input_text = st.text_area("Enter your text:", placeholder="Type something...")

# Predict sentiment
if st.button("Predict Sentiment"):
    if input_text.strip() == "":
        st.warning("Please enter some text for analysis!")
    else:
        # Preprocess and predict
        input_text_transformed = tfidf_vectorizer.transform([input_text])
        prediction = model.predict(input_text_transformed)
        st.success(f"Predicted Sentiment: **{prediction[0]}**")

# Sidebar for additional options
st.sidebar.header("About")
st.sidebar.write("""
This app uses a Logistic Regression model trained on a dataset of emotional text comments. 
For more details, check out the [GitHub repository](https://github.com/mishi-25/emotions).
""")
