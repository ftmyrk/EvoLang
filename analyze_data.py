# analyze_data.py

from utils import text_analysis_utils as analysis_utils
import pandas as pd

# Load preprocessed and generated data
old_data = pd.read_csv("generated_responses_2013.csv")
new_data = pd.read_csv("generated_responses_2023.csv")

# Convert data to the expected format for analysis functions
old_events = [{"text": text} for text in old_data["Original_Text"]]
new_events = [{"text": text} for text in new_data["Original_Text"]]

# Perform analysis
analysis_utils.word_frequency_analysis(old_events, "2013 Articles", "/home/otamy001/EvoLang/outputs/word_freq_2013.png")
analysis_utils.word_frequency_analysis(new_events, "2023 Articles", "/home/otamy001/EvoLang/outputs/word_freq_2023.png")
analysis_utils.generate_wordcloud(old_events, "2013 Articles", "/home/otamy001/EvoLang/outputs/wordcloud_2013.png")
analysis_utils.generate_wordcloud(new_events, "2023 Articles", "/home/otamy001/EvoLang/outputs/wordcloud_2023.png")

# Sentiment analysis
sentiment_2013 = analysis_utils.analyze_sentiment(old_events)
sentiment_2023 = analysis_utils.analyze_sentiment(new_events)
analysis_utils.visualize_sentiment(sentiment_2013, "2013 Articles", "/home/otamy001/EvoLang/outputs/visualize_sentiment_2013.png")
analysis_utils.visualize_sentiment(sentiment_2023, "2023 Articles", "/home/otamy001/EvoLang/outputs/visualize_sentiment_2023.png")
