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

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r'\w+')


# Function to clean and lowercase text
def preprocess_text(text, max_tokens=1024):
    # Converts to lowercase
    text = text.lower()
    # Replaces encoded single quotes/apostrophes and other symbols
    text = text.replace("â\x80\x99", "'")  # Replaces with regular apostrophe
    text = text.replace("â\x80\x9c", "")   # Removes opening quotation-like symbol
    text = text.replace("â\x80\x9d", "")   # Removes closing quotation-like symbol
    text = text.replace("â\x80¦", "...")   # Replacs with ellipsis
    text = text.replace("\\'", "'")        # Replaces backslash before single quotes
    text = text.replace("\\n", " ")        # new line
    text = text.replace("\\t", "    ")     # Tab
    text = re.sub(r"\b\d+\b", "", text)  # Remove standalone numbers

    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    truncated_tokens = tokens[:max_tokens]
    return " ".join(truncated_tokens).strip()
# Function to load and preprocess the dataset

def load_dataset(file_path, max_tokens=256, preprocess_column=None):
    df = pd.read_csv(file_path)

    # Dynamically find the appropriate column
    if not preprocess_column:
        if "Text" in df.columns:
            preprocess_column = "Text"
        elif "article" in df.columns:
            preprocess_column = "article"
        else:
            raise ValueError("No valid text column found. Expected 'Text' or 'article'.")

    df["Preprocessed_Text"] = df[preprocess_column].astype(str).apply(
        lambda x: preprocess_text(x, max_tokens=max_tokens)
    )
    return df

