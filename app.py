import streamlit as st
import pickle

# Load the model and vectorizer
with open("news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Set page configuration
st.set_page_config(page_title="News Category Predictor", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://e0.pxfuel.com/wallpapers/981/852/desktop-wallpaper-newspaper-background-black-and-white-newspaper.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #222222;
        font-family: 'Arial', sans-serif;
    }

    .stTextInput, .stTextArea, .stButton > button {
        background-color: #ffffffdd;
        color: #000000 !important;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }

    .stTextInput input, .stTextArea textarea {
        color: #000000 !important;
    }

    .stButton > button {
        transition: 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #444444 !important;
        color: #ffffff !important;
    }

    .stTitle, .stMarkdown {
        background-color: #f9f9f9dd;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }

    .stSuccess {
        background-color: #dff0d8 !important;
        color: #3c763d !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and instructions
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
