# text_preprocessing_utils.py

import csv
import re
import sys
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to clean and lowercase text
def preprocess_text(text, max_tokens=256):
    # Converts to lowercase
    text = text.lower()
    # Replaces encoded single quotes/apostrophes and other symbols
    text = text.replace("â\x80\x99", "'")  # Replaces with regular apostrophe
    text = text.replace("â\x80\x9c", "")   # Removes opening quotation-like symbol
    text = text.replace("â\x80\x9d", "")   # Removes closing quotation-like symbol
    text = text.replace("â\x80¦", "...")   # Replacs with ellipsis
    text = text.replace("\\'", "'")        # Replaces backslash before single quotes
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    truncated_tokens = tokens[:max_tokens]
    return " ".join(truncated_tokens).strip()
# Function to load and preprocess the dataset

def load_dataset(file_path):
    df = pd.read_csv(file_path)

    # Dynamically find the appropriate column
    if 'Text' in df.columns:
        text_column = 'Text'
    elif 'article' in df.columns:
        text_column = 'article'
    else:
        raise ValueError("No valid text column found. Expected 'Text' or 'article'.")

    # Preprocess the text and add a new column
    df['Preprocessed_Text'] = df[text_column].apply(preprocess_text)
    return df

