import streamlit as st
import pickle

# Load the model and vectorizer
with open("news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Set page configuration
st.set_page_config(page_title="News Category Predictor", layout="centered")

# Inject custom CSS for background
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://st.depositphotos.com/1032463/1373/i/450/depositphotos_13732950-Background-of-old-vintage-newspapers.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: black;
    }

    .stTextInput, .stTextArea, .stButton > button {
        background-color: #ffffffaa;
        border-radius: 8px;
    }

    .stTextInput input, .stTextArea textarea {
        color: black;
    }

    .stTitle, .stMarkdown {
        background-color: #ffffffcc;
        padding: 10px;
        border-radius: 10px;
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
