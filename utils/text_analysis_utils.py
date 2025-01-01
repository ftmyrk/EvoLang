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
from collections import Counter
import networkx as nx
import os

COLUMNN = "Extracted_Key_Response"  
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    sentiments = []

    for event in events["Generated_Full_Response"].astype(str):
        # Truncate text to the first 512 characters
        truncated_event = event[:512]
        try:
            result = sentiment_analyzer(truncated_event)  # Analyze sentiment
            sentiments.append({"text": truncated_event, "label": result[0]["label"], "score": result[0]["score"]})
        except Exception as e:
            print(f"Error analyzing sentiment for text: {truncated_event[:50]}... | Error: {e}")
            sentiments.append({"text": truncated_event, "label": "ERROR", "score": 0.0})
    return sentiments

# Visualize Sentiment
def visualize_sentiment(sentiments, title, output_file):
    df = pd.DataFrame(sentiments)
    sns.countplot(x="label", data=df, palette="coolwarm")
    plt.title(f"Sentiment Distribution in {title}")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(output_file)
    plt.close()
    print(f"Sentiment visualization for {title} saved to {output_file}")

# Keyword Frequency Comparison
def compare_keyword_frequencies(events_2013, events_2023, keywords):
    def count_keywords(events, keywords):
        all_text = " ".join(events[COLUMNN].astype(str))
        word_counts = Counter(all_text.split())
        return {keyword: word_counts[keyword] for keyword in keywords if keyword in word_counts}

    keyword_counts_2013 = count_keywords(events_2013, keywords)
    keyword_counts_2023 = count_keywords(events_2023, keywords)

    print(f"{'Keyword':<15} {'2013 Count':<12} {'2023 Count':<12} {'Difference':<10}")
    print("-" * 45)
    for keyword in keywords:
        count_2013 = keyword_counts_2013.get(keyword, 0)
        count_2023 = keyword_counts_2023.get(keyword, 0)
        difference = count_2023 - count_2013
        print(f"{keyword:<15} {count_2013:<12} {count_2023:<12} {difference:<10}")
        
def plot_keyword_frequency(keywords, events_2013, events_2023, output_file):
    def count_keywords(events, keywords):
        all_text = " ".join(events[COLUMNN].astype(str))
        word_counts = Counter(all_text.split())
        return {keyword: word_counts[keyword] for keyword in keywords if keyword in word_counts}

    keyword_counts_2013 = count_keywords(events_2013, keywords)
    keyword_counts_2023 = count_keywords(events_2023, keywords)

    data = []
    for keyword in keywords:
        count_2013 = keyword_counts_2013.get(keyword, 0)
        count_2023 = keyword_counts_2023.get(keyword, 0)
        data.append({"Keyword": keyword, "Year": "2013", "Count": count_2013})
        data.append({"Keyword": keyword, "Year": "2023", "Count": count_2023})

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Keyword", y="Count", hue="Year", data=df, palette="pastel")
    plt.title("Keyword Frequency Comparison (2013 vs 2023)")
    plt.xlabel("Keywords")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Keyword frequency comparison saved to {output_file}")

# Word Association Network
def generate_word_association(events, title, output_file, year, top_n=50, edge_threshold=2):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    text = clean_and_prepare_text(events, COLUMNN)
    words = text.split()

    # Count word frequencies
    word_counts = Counter(words)
    most_common_words = set([word for word, _ in word_counts.most_common(top_n)])

    # Create bigrams and filter by frequency
    bigrams = [(w1, w2) for w1, w2 in zip(words, words[1:]) if w1 in most_common_words and w2 in most_common_words]
    bigram_counts = Counter(bigrams)
    filtered_bigrams = [(w1, w2) for (w1, w2), count in bigram_counts.items() if count >= edge_threshold]

    # Create graph
    graph = nx.Graph()
    graph.add_edges_from(filtered_bigrams)
    node_sizes = [word_counts[word] * 10 for word in graph.nodes]

    # Draw graph
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 8))
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
    plt.title(f"{title} - Word Association Network")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Word Association Network saved to {output_file}")