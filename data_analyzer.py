import pandas as pd
import os
from utils.text_analysis_utils import word_frequency_analysis, generate_wordcloud, analyze_sentiment, visualize_sentiment, compare_keyword_frequencies, plot_keyword_frequency, generate_word_association
from utils.macros import OUTPUT_DIR, KEYWORDS, GENERATED_DATA_DIR_2013, GENERATED_DATA_DIR_2023

COLUMNN = "Extracted_Key_Response"  

print("Loading datasets...")
data_2013 = pd.read_csv(GENERATED_DATA_DIR_2013)
data_2023 = pd.read_csv(GENERATED_DATA_DIR_2023)
analysis_dir = os.path.join(OUTPUT_DIR, "analysis")
os.makedirs(analysis_dir, exist_ok=True)

# Word Cloud and Word Frequency Analysis
print("Generating word clouds...")
generate_wordcloud(data_2013, os.path.join(analysis_dir, "wordcloud_2013.png"), year=2013)
generate_wordcloud(data_2023, os.path.join(analysis_dir, "wordcloud_2023.png"), year=2023)

print("Performing word frequency analysis...")
word_frequency_analysis(data_2013, "2013 Data", os.path.join(analysis_dir, "word_frequency_2013.png"))
word_frequency_analysis(data_2023, "2023 Data", os.path.join(analysis_dir, "word_frequency_2023.png"))

# Sentiment Analysis
print("Performing sentiment analysis...")
sentiments_2013 = analyze_sentiment(data_2013)
sentiments_2023 = analyze_sentiment(data_2023)

print("Visualizing sentiment...")
visualize_sentiment(sentiments_2013, "Sentiment Analysis for 2013", os.path.join(analysis_dir, "sentiment_2013.png"))
visualize_sentiment(sentiments_2023, "Sentiment Analysis for 2023", os.path.join(analysis_dir, "sentiment_2023.png"))

# Keyword Frequency Comparison
print("Comparing keyword frequencies...")
compare_keyword_frequencies(data_2013, data_2023, KEYWORDS)
plot_keyword_frequency(KEYWORDS, data_2013, data_2023, os.path.join(analysis_dir, "keyword_comparison.png"))

# Word Association Network
print("Generating word association networks...")
generate_word_association(data_2013, "Word Associations for 2013", os.path.join(analysis_dir, "word_association_2013.png"), year=2013)
generate_word_association(data_2023, "Word Associations for 2023", os.path.join(analysis_dir, "word_association_2023.png"), year=2023)

print("Data analysis complete. Outputs saved.")