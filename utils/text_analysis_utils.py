# text_analysis_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from collections import Counter
from wordcloud import WordCloud
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import torch

COLUMNN = "Text"

def ntlk_download():
    nltk.download('stopwords')
    nltk.download('punkt')

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Word Frequency Analysis 
def word_frequency_analysis(events, title, output_file):
    all_text = " ".join(events[COLUMNN].astype(str))
    words = all_text.split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, hue=counts, dodge=False, palette="viridis")
    plt.title(f"Top 20 Words in {title}")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.savefig(output_file)
    plt.close()

# Word Cloud
def generate_wordcloud(events, title, output_file):
    ntlk_download()
    all_text = " ".join(events[COLUMNN].astype(str))
    wordcloud = WordCloud(  
        width=800,
        height=400,
        background_color='white',
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'  
    ).generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {title}")
    plt.savefig(output_file)
    plt.close()


# Sentiment Analysis
def analyze_sentiment(events):
    sentiments = []
    for event in events[COLUMNN].astype(str):
        result = sentiment_analyzer(event[:512])
        sentiments.append(result[0])
    return sentiments

# Visualize Sentiment
def visualize_sentiment(sentiments, title, output_file):
    df = pd.DataFrame(sentiments)
    sns.countplot(x='label', hue='label', data=df, dodge=False, palette="coolwarm")
    plt.title(f"Sentiment Distribution in {title}")
    plt.savefig(output_file) 
    plt.close()  

# Stopword Removal and Unique Words
def clean_text(text):
    ntlk_download()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return cleaned_words

def get_unique_words(events):
    all_text = " ".join(events[COLUMNN].astype(str))
    cleaned_words = clean_text(all_text)
    return set(cleaned_words)

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

# Contextual Analysis
def contextual_analysis(word, events, year, model, tokenizer):
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # Use GPU if its available
    )
    for i, event in enumerate(events[:3]):
        input_text = f"In the year {year}, people commonly discussed {word}. Can you tell me more about {word}?"
        generated_text = generation_pipeline(input_text, max_new_tokens=50)[0]['generated_text']
        print(f"Contextual Response for {year}: {generated_text}\n")


    
    
def plot_bar_chart(data, title, xlabel, ylabel, output_file):
    # Convert keys and values to lists
    keys = list(data.keys())
    values = list(data.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(keys, values, color='skyblue')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_file)
    plt.close()
    print(f"Chart saved to {output_file}")

def plot_keyword_frequency(keywords, tokens_2013, tokens_2023, output_file):
    flat_tokens_2013 = list(chain.from_iterable(tokens_2013))
    flat_tokens_2023 = list(chain.from_iterable(tokens_2023))

    freq_2013 = Counter(flat_tokens_2013)
    freq_2023 = Counter(flat_tokens_2023)

    keyword_frequencies = {keyword: (freq_2013.get(keyword, 0), freq_2023.get(keyword, 0)) for keyword in keywords}
    keywords, freqs_2013, freqs_2023 = zip(*[(k, v[0], v[1]) for k, v in keyword_frequencies.items()])

    x = range(len(keywords))
    plt.figure(figsize=(10, 6))
    plt.bar(x, freqs_2013, width=0.4, label="2013", align="center")
    plt.bar(x, freqs_2023, width=0.4, label="2023", align="edge")
    plt.xticks(x, keywords)
    plt.xlabel("Keywords")
    plt.ylabel("Frequency")
    plt.title("Keyword Frequency Comparison")
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Chart saved to {output_file}")


def extract_keyword_sentences(data, keyword):
    return data[data[COLUMNN].str.contains(keyword, case=False, na=False)]["Text"].tolist()

def generate_heatmap(texts_2013, texts_2023, output_file):
    vectorizer = CountVectorizer(max_features=100, stop_words="english")
    combined_texts = texts_2013 + texts_2023
    matrix = vectorizer.fit_transform(combined_texts).toarray()
    vocab = vectorizer.get_feature_names()
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(matrix, columns=vocab).corr(), cmap="coolwarm")
    plt.title("Refined Word Overlap Heatmap")
    plt.savefig(output_file)
    plt.close()
    