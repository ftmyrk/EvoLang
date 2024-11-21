# analyze_data.py

from utils import text_analysis_utils as analysis_utils
from utils.macros import old_data, new_data, OUTPUT_DIR

# Convert data to expected format
old_events = [{"text": text} for text in old_data["Original_Text"]]
new_events = [{"text": text} for text in new_data["Original_Text"]]

# Perform analysis
analysis_utils.word_frequency_analysis(old_events, "2013 Articles", os.path.join(OUTPUT_DIR, 'word_freq_2013.png'))
analysis_utils.word_frequency_analysis(new_events, "2023 Articles", os.path.join(OUTPUT_DIR, 'word_freq_2023.png'))
analysis_utils.generate_wordcloud(old_events, "2013 Articles", os.path.join(OUTPUT_DIR, 'wordcloud_2013.png'))
analysis_utils.generate_wordcloud(new_events, "2023 Articles", os.path.join(OUTPUT_DIR, 'wordcloud_2023.png'))

# Sentiment analysis
sentiment_2013 = analysis_utils.analyze_sentiment(old_events)
sentiment_2023 = analysis_utils.analyze_sentiment(new_events)
analysis_utils.visualize_sentiment(sentiment_2013, "2013 Articles", os.path.join(OUTPUT_DIR, 'visualize_sentiment_2013.png'))
analysis_utils.visualize_sentiment(sentiment_2023, "2023 Articles", os.path.join(OUTPUT_DIR, 'visualize_sentiment_2023.png'))