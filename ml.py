# ================== AI COLLEGE CHATBOT (ONE CELL) ==================

import pandas as pd
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load CSV ----------
# CSV format:
# question,answer
# What is college timing?,College timing is 9 AM to 5 PM

data = pd.read_csv("college_chatbot_dataset.csv")

questions = data["question"].astype(str).str.lower()
answers = data["answer"].astype(str)

# ---------- Text Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

questions = questions.apply(clean_text)

# ---------- Vectorization (AI Brain) ----------
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
question_vectors = vectorizer.fit_transform(questions)

# ---------- Chatbot Logic ----------
def chatbot(user_input):
    user_input = clean_text(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()
    confidence = similarity[0][best_match]

    if confidence < 0.3:
        return "Sorry, I am not sure. Please contact the college office."
    else:
        return answers.iloc[best_match]

# ---------- Chat Loop ----------
print("🤖 College Chatbot Started (type 'exit' to stop)\n")

while True:
    user = input("You: ")
    if user.lower() == "exit":
        print("Bot: Thank you! Have a nice day 😊")
        break
    print("Bot:", chatbot(user))

# ================== END ==================
