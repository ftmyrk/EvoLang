import pandas as pd
import os
from utils.text_analysis_utils import (
    word_frequency_analysis,
    generate_wordcloud,
    analyze_sentiment,
    visualize_sentiment,
    extract_keyword_sentences,
    generate_heatmap,
)
from utils.macros import GENERATED_DATA_DIR, OUTPUT_DIR, MODEL_DIR, tokenize_data, load_or_train_word2vec, KEYWORDS

print("Loading datasets...")
comparison_2013 = pd.read_csv(os.path.join(GENERATED_DATA_DIR, "comparison_2013.csv"))
comparison_2023 = pd.read_csv(os.path.join(GENERATED_DATA_DIR, "comparison_2023.csv"))

# Detects outliers
threshold = comparison_2013["Difference"].quantile(0.95)
outliers_2013 = comparison_2013[comparison_2013["Difference"] > threshold]
outliers_2023 = comparison_2023[comparison_2023["Difference"] > threshold]

outliers_2013.to_csv(os.path.join(OUTPUT_DIR, "outliers_2013.csv"), index=False)
outliers_2023.to_csv(os.path.join(OUTPUT_DIR, "outliers_2023.csv"), index=False)

# Generates Word Cloud
print("Generating word clouds...")
generate_wordcloud(outliers_2013, "Top 2013 Differences", os.path.join(OUTPUT_DIR, "wordcloud_top_2013.png"))
generate_wordcloud(outliers_2023, "Top 2023 Differences", os.path.join(OUTPUT_DIR, "wordcloud_top_2023.png"))

# Word Frequency Analysis
print("Performing word frequency analysis...")
word_frequency_analysis(outliers_2013, "Top 2013 Differences", os.path.join(OUTPUT_DIR, "word_frequency_2013.png"))
word_frequency_analysis(outliers_2023, "Top 2023 Differences", os.path.join(OUTPUT_DIR, "word_frequency_2023.png"))

# Sentiment Analysis
print("Performing sentiment analysis...")
sentiments_2013 = analyze_sentiment(outliers_2013)
sentiments_2023 = analyze_sentiment(outliers_2023)

# Visualize Sentiment
print("Visualizing sentiment...")
visualize_sentiment(sentiments_2013, "Top 2013 Differences Sentiment", os.path.join(OUTPUT_DIR, "sentiment_2013.png"))
visualize_sentiment(sentiments_2023, "Top 2023 Differences Sentiment", os.path.join(OUTPUT_DIR, "sentiment_2023.png"))

# Word2Vec analysis
tokens_2013 = [text.split() for text in outliers_2013["Text"].tolist()]
tokens_2023 = [text.split() for text in outliers_2023["Text"].tolist()]

model_2013 = load_or_train_word2vec(tokens_2013, 2013)
model_2023 = load_or_train_word2vec(tokens_2023, 2023)

# Heatmap 
generate_heatmap(outliers_2013["Text"].tolist(), outliers_2023["Text"].tolist(), os.path.join(OUTPUT_DIR, "heatmap_outliers.png"))
