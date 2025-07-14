import os
import csv
import datetime
import nltk
import ssl
import streamlit as st
import random 
import json
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

with open('intents.json', 'r') as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags=[]
patterns=[]
for intent in intents:
    for pattern in intent['patterns']:
         tags.append(intent['tag'])
         patterns.append(pattern)
