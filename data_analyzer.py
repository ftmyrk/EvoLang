import pandas as pd
import os
from utils.text_analysis_utils import (
    word_frequency_analysis,
    generate_wordcloud,
    analyze_sentiment,
    visualize_sentiment,
    compare_keyword_frequencies,
    plot_keyword_frequency,
)
from utils.macros import OUTPUT_DIR, KEYWORDS

print("Loading datasets...")
print("Loading probability data...")
probs_2011 = pd.read_csv(f"{OUTPUT_DIR}/probabilities_2011.csv")
probs_2021 = pd.read_csv(f"{OUTPUT_DIR}/probabilities_2021.csv")

# Identify top differences
threshold_2011 = probs_2011["Log_Probability"].quantile(0.95)
threshold_2021 = probs_2021["Log_Probability"].quantile(0.95)

outliers_2011 = probs_2011[probs_2011["Log_Probability"] > threshold_2011]
outliers_2021 = probs_2021[probs_2021["Log_Probability"] > threshold_2021]

outliers_2011.to_csv(f"{OUTPUT_DIR}/outliers_2011.csv", index=False)
outliers_2021.to_csv(f"{OUTPUT_DIR}/outliers_2021.csv", index=False)

print("Generating word clouds...")
generate_wordcloud(outliers_2011, "Outliers 2011", f"{OUTPUT_DIR}/wordcloud_2011.png")
generate_wordcloud(outliers_2021, "Outliers 2021", f"{OUTPUT_DIR}/wordcloud_2021.png")

print("Performing word frequency analysis...")
word_frequency_analysis(outliers_2011, "Outliers 2011", f"{OUTPUT_DIR}/word_frequency_2011.png")
word_frequency_analysis(outliers_2021, "Outliers 2021", f"{OUTPUT_DIR}/word_frequency_2021.png")

print("Performing sentiment analysis...")
sentiments_2011 = analyze_sentiment(outliers_2011)
sentiments_2021 = analyze_sentiment(outliers_2021)

print("Visualizing sentiment...")
visualize_sentiment(sentiments_2011, "Outliers 2011 Sentiment", f"{OUTPUT_DIR}/sentiment_2011.png")
visualize_sentiment(sentiments_2021, "Outliers 2021 Sentiment", f"{OUTPUT_DIR}/sentiment_2021.png")

print("Comparing keyword frequencies...")
compare_keyword_frequencies(outliers_2011, outliers_2021, KEYWORDS)
plot_keyword_frequency(KEYWORDS, outliers_2011, outliers_2021, f"{OUTPUT_DIR}/keyword_frequencies_comparison.png")

print("Data analysis complete. Outputs saved.")

# tokens_2013 = [text.split() for text in outliers_2013["Text"].tolist()]
# tokens_2023 = [text.split() for text in outliers_2023["Text"].tolist()]

# model_2013 = load_or_train_word2vec(tokens_2013, 2013)
# model_2023 = load_or_train_word2vec(tokens_2023, 2023)
