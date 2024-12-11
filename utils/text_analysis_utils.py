# text_analysis_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from itertools import combinations
from collections import Counter
import nltk
import os
from networkx import Graph
import networkx as nx

COLUMNN = "Extracted_Key_Response"

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in STOPWORDS]
    return " ".join(words)

def clean_and_prepare_text(events, column):
    all_text = " ".join(events[column].astype(str))
    return preprocess_text(all_text)

# Word Frequency Analysis
def word_frequency_analysis(events, title, output_file):
    all_text = clean_and_prepare_text(events, COLUMNN)
    word_counts = Counter(all_text.split())
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette="viridis")
    plt.title(f"Top 20 Words in {title}")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.savefig(output_file)
    plt.close()
    print(f"Word frequency plot for {title} saved to {output_file}")
    
# Word Cloud
def generate_wordcloud(events, output_file, year):
    all_text = clean_and_prepare_text(events, COLUMNN)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {year}")
    plt.savefig(output_file)
    plt.close()
    print(f"Word cloud saved to {output_file}")

# Sentiment Analysis
def analyze_sentiment(events):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
    sentiments = []

    for event in events["Generated_Full_Response"].astype(str):
        result = sentiment_analyzer(event[:512])  # Truncate text to the first 512 characters
        sentiments.append({"text": event, "label": result[0]["label"]})
    return sentiments

# Visualize Sentiment
def visualize_sentiment(sentiments, title, output_file):
    df = pd.DataFrame(sentiments)
    if "label" not in df.columns:
        raise ValueError("The DataFrame does not contain a 'label' column for sentiment visualization.")
    sns.countplot(x="label", data=df, palette="coolwarm")
    plt.title(f"Sentiment Distribution in {title}")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(output_file)
    plt.close()
    print(f"Sentiment visualization for {title} saved to {output_file}")

# Keyword Frequency Comparison
def compare_keyword_frequencies(events_2011, events_2021, keywords):
    def count_keywords(events, keywords):
        all_text = " ".join(events[COLUMNN].astype(str))
        word_counts = Counter(all_text.split())
        return {keyword: word_counts[keyword] for keyword in keywords if keyword in word_counts}

    keyword_counts_2011 = count_keywords(events_2011, keywords)
    keyword_counts_2021 = count_keywords(events_2021, keywords)

    print(f"{'Keyword':<15} {'2011 Count':<12} {'2021 Count':<12} {'Difference':<10}")
    print("-" * 45)
    for keyword in keywords:
        count_2011 = keyword_counts_2011.get(keyword, 0)
        count_2021 = keyword_counts_2021.get(keyword, 0)
        difference = count_2021 - count_2011
        print(f"{keyword:<15} {count_2011:<12} {count_2021:<12} {difference:<10}")

def generate_word_association(events, title, output_file, year):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    text = clean_and_prepare_text(events, COLUMNN)
    words = text.split()
    bigrams = zip(words, words[1:])
    graph = nx.Graph()
    graph.add_edges_from(bigrams)
    word_counts = Counter(words)
    node_sizes = [word_counts[word] * 10 for word in graph.nodes]

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        labels={node: node for node in graph.nodes},
        node_size=node_sizes,
        font_size=8,
        font_color="black",
        node_color="skyblue",
        edge_color="gray",
        alpha=0.8,
    )    
    plt.title(title)
    plt.savefig(output_file)
    plt.tight_layout()
    plt.close()
    print(f"Word Association Network saved to {output_file}")