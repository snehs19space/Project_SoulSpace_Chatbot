import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

def home_page():
    st.session_state.page = 'Page 2'
    st.write("""<div style="text-align: center;">
        <h1>Welcome to Soul Space !!</h1><br><hr>
        <p>A safe haven to share emotions freely, fostering connection,
        healing, and growth through compassion, understanding, and the
        power of being truly heard.</p><hr><br><br><br>
        </div>""",unsafe_allow_html=True)
    if st.button("Get Started"):
        pass

def page_2():
    main()

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu\n\n", menu)

    # Home
    if choice == "Home":
        st.title("Share yourself freely at Soul Space")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        intro = "Soulsy: Hi there! I'm Soulsy. 😊 "
        with st.chat_message("assistant"):
            st.write(intro)
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Response', 'Timestamp'])

        counter += 1
        user_input = st.chat_input("How are you feeling today?", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            with st.chat_message("user"):
                st.write(f'You: {user_input_str}')
            st.session_state.messages.append({"role": "user", "content": user_input_str})
            response = chatbot(user_input)
            if "goodbye" in user_input_str.lower() or "bye" in user_input_str.lower():
                with st.chat_message("assistant"):
                    st.write("Soulsy: Take care! I\'m always here to chat whenever you need me. 😊")
                    st.stop()
            else:
                with st.chat_message("assistant"):
                    st.write(f'Soulsy: {response}')
            st.session_state.messages.append({"role": "assistant", "content": response})

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([timestamp, user_input_str, response])

    # Conversation History
    elif choice == "Conversation History":
        st.title("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"Timestamp: {row[0]}")
                    st.text(f"User: {row[1]}")
                    st.text(f"Soulsy: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")
    # About
    elif choice == "About":
        st.title("About Soul Space")
        st.write("I'm a chatbot designed to help you share your emotions and provide uplifting suggestions!")
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface.
           The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Soul Space Interface:")
        inf = """The chatbot interface is built using Streamlit. The interface includes a text input box for users
        to input their text and a chat window to display the chatbot's responses. The interface uses the trained
        model to generate responses to user input."""
        st.write(inf)

        st.subheader("Conclusion:")
        conc = """In this project, a chatbot is built that can understand and respond to user input based on intents.
        The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit.
        This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms."""
        st.write(conc)


if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'Page 2':
    page_2()

