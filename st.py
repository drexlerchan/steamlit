import streamlit as st 

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Flatten,Conv1D,MaxPooling1D,Embedding,Dropout,LSTM
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

# Set the app title 
st.title('My First Streamlit App') 
# Add a welcome message 
st.write('Welcome to my Streamlit app!') 

form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')

#Load model

# Create a text input 
#widgetuser_input = st.text_input('Enter a custom message:', 'Hello, Streamlit!') 
#user_input = st.text_input('Enter a custom message:', 'Hello, Streamlit!')

# Display the customized message 
#st.write('Customized Message:', widgetuser_input)