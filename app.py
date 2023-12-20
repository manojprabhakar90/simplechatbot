import streamlit as st
import tensorflow as tf
import numpy as np
import random
import json
import pickle
import nltk
nltk.download('punkt')
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

tags = [intent['tag'] for intent in intents_data['intents'] if intent['responses']]

# Start a conversation list
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Display buttons for each tag
for tag in tags:
    if st.button(tag):
        # Assume the first pattern is the representative one for the tag
        pattern = next((item for item in intents_data['intents'] if item["tag"] == tag), None)['patterns'][0]
        st.session_state['conversation'].append(f"You: {pattern}")
        response = get_response(pattern)  # Assume this function generates the chatbot's response
        st.session_state['conversation'].append(f"Candidtron: {response}")

# Text input for free-form questions
user_input = st.text_input("Ask me anything...", key="input")

if st.button("Send", key="send"):
    st.session_state['conversation'].append(f"You: {user_input}")
    response = get_response(user_input)
    st.session_state['conversation'].append(f"Candidtron: {response}")

# Display the conversation
for line in st.session_state['conversation']:
    st.text(line)

def get_response(user_input):
    # Preprocess the user_input to create a bag-of-words
    bag = bag_of_words(user_input, words)
    
    model_output = model.predict(np.array([bag]))[0]
    predicted_tag_index = np.argmax(model_output)
    predicted_tag = labels[predicted_tag_index]

    responses = [intent['responses'] for intent in data['intents'] if intent['tag'] == predicted_tag]
    
    if model_output[predicted_tag_index] > 0.7:
        response = random.choice(responses[0]) if responses else "I am not sure how to respond to that."
    else:
        response = "I didn't get that. Can you rephrase?"

    return response
