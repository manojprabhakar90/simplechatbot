import streamlit as st
import tensorflow as tf
import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

model = tf.keras.models.load_model("model.h5")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

st.title("Candidtron Chatbot")

user_input = st.text_input("You: ", key="input")

if st.button("Send"):
    bag = bag_of_words(user_input, words)
    result = model.predict(np.array([bag]))[0]
    result_index = np.argmax(result)
    tag = labels[result_index]

    response = "I didn't get that. Can you rephrase?"
    if result[result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                response = random.choice(responses)
                break

    st.text(f"Candidtron: {response}")
