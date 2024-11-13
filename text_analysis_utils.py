# text_analysis_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from transformers import pipeline
import pandas as pd
import nltk
from nltk.corpus import stopwords
import torch

# Download stopwords if not already available
nltk.download('stopwords')

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Word Frequency Analysis
def word_frequency_analysis(events, title):
    all_text = " ".join([event["text"] for event in events])
    words = all_text.split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette="viridis")
    plt.title(f"Top 20 Words in {title}")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.show()

# Word Cloud
def generate_wordcloud(events, title):
    all_text = " ".join([event["text"] for event in events])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {title}")
    plt.show()

# Sentiment Analysis
def analyze_sentiment(events):
    sentiments = []
    for event in events:
        result = sentiment_analyzer(event["text"][:512])
        sentiments.append(result[0])
    return sentiments

# Visualize Sentiment
def visualize_sentiment(sentiments, title):
    df = pd.DataFrame(sentiments)
    sns.countplot(x='label', data=df, palette="coolwarm")
    plt.title(f"Sentiment Distribution in {title}")
    plt.show()

# Stopword Removal and Unique Words
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return cleaned_words

def get_unique_words(events):
    all_text = " ".join([event["text"] for event in events])
    cleaned_words = clean_text(all_text)
    return set(cleaned_words)

# Keyword Frequency Comparison
def compare_keyword_frequencies(events_2013, events_2023, keywords):
    def count_keywords(events, keywords):
        all_text = " ".join([event["text"] for event in events])
        word_counts = Counter(all_text.split())
        return {keyword: word_counts[keyword] for keyword in keywords if keyword in word_counts}

    keyword_counts_2013 = count_keywords(events_2013, keywords)
    keyword_counts_2023 = count_keywords(events_2023, keywords)

    for keyword in keywords:
        count_2013 = keyword_counts_2013.get(keyword, 0)
        count_2023 = keyword_counts_2023.get(keyword, 0)
        print(f"Keyword: {keyword}, 2013 Count: {count_2013}, 2023 Count: {count_2023}, Difference: {count_2023 - count_2013}")

# Contextual Analysis
def contextual_analysis(word, events, year, model, tokenizer):
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )
    for i, event in enumerate(events[:3]):
        input_text = f"In the year {year}, people commonly discussed {word}. Can you tell me more about {word}?"
        generated_text = generation_pipeline(input_text, max_new_tokens=50)[0]['generated_text']
        print(f"Contextual Response for {year}: {generated_text}\n")
