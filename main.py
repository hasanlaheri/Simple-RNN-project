## import the library and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for (key,value) in word_index.items()}

## load the model
 model = load_model("rnn_imdb_model.h5", compile=False)
## Function to decode review
def decode_review(encodeed_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encodeed_review])

## Function to preprocess the user input
def preprocess_text(text):
    # Tokenize the text
    words = text.lower().split()
    # Convert tokens to their corresponding indices based on the IMDB dataset's word index
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words, +3 for reserved indices
    # Pad the sequence to ensure it has a length of 500
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## prediction function
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

import streamlit as st
## Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")\


##USer input
user_input = st.text_area("Movie Review")
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"

    ## Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")

else:

    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")

