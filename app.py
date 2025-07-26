import streamlit as st
import pickle

# Load the model and vectorizer
with open("news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="News Category Predictor", layout="centered")
st.title("📰 News Category Prediction")
st.markdown("Enter a news headline and description to predict its category.")

# User input
headline = st.text_input("Enter Headline")
description = st.text_area("Enter Short Description")

# Predict button
if st.button("Predict Category"):
    if headline.strip() == "" and description.strip() == "":
        st.warning("Please enter at least a headline or description.")
    else:
        input_text = headline + " " + description
        text_vector = vectorizer.transform([input_text])
        prediction = model.predict(text_vector)
        st.success(f"**Predicted Category:** `{prediction[0]}`")
