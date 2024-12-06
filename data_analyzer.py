import pandas as pd
import os
from utils.text_analysis_utils import (
    word_frequency_analysis,
    generate_wordcloud,
    analyze_sentiment,
    visualize_sentiment,
    compare_keyword_frequencies,
    plot_bar_chart,
    plot_keyword_frequency,
    heat_map,
)
from utils.macros import GENERATED_DATA_DIR, OUTPUT_DIR, MODEL_DIR, tokenize_data, load_or_train_word2vec, KEYWORDS

# Load datasets
print("Loading datasets...")
comparison_2013 = pd.read_csv(os.path.join(GENERATED_DATA_DIR, "comparison_2013.csv"))
comparison_2023 = pd.read_csv(os.path.join(GENERATED_DATA_DIR, "comparison_2023.csv"))

# Analyze top differences
print("Identifying top differences...")
top_2013 = comparison_2013.nlargest(50, "Difference")
top_2023 = comparison_2023.nlargest(50, "Difference")

# Save top differences for further analysis
top_2013.to_csv(os.path.join(OUTPUT_DIR, "top_differences_2013.csv"), index=False)
top_2023.to_csv(os.path.join(OUTPUT_DIR, "top_differences_2023.csv"), index=False)

# Generate Word Cloud
print("Generating word clouds...")
generate_wordcloud(top_2013, "Top 2013 Differences", os.path.join(OUTPUT_DIR, "wordcloud_top_2013.png"))
generate_wordcloud(top_2023, "Top 2023 Differences", os.path.join(OUTPUT_DIR, "wordcloud_top_2023.png"))

# Word Frequency Analysis
print("Performing word frequency analysis...")
word_frequency_analysis(top_2013, "Top 2013 Differences", os.path.join(OUTPUT_DIR, "word_frequency_2013.png"))
word_frequency_analysis(top_2023, "Top 2023 Differences", os.path.join(OUTPUT_DIR, "word_frequency_2023.png"))

# Sentiment Analysis
print("Performing sentiment analysis...")
sentiments_2013 = analyze_sentiment(top_2013)
sentiments_2023 = analyze_sentiment(top_2023)

# Visualize Sentiment
print("Visualizing sentiment...")
visualize_sentiment(sentiments_2013, "Top 2013 Differences Sentiment", os.path.join(OUTPUT_DIR, "sentiment_2013.png"))
visualize_sentiment(sentiments_2023, "Top 2023 Differences Sentiment", os.path.join(OUTPUT_DIR, "sentiment_2023.png"))

# Keyword Analysis
print("Comparing keyword frequencies...")
compare_keyword_frequencies(comparison_2013, comparison_2023, KEYWORDS)

# Plot Keyword Frequencies
print("Plotting keyword frequencies...")
tokens_2013 = tokenize_data(comparison_2013)
tokens_2023 = tokenize_data(comparison_2023)
plot_keyword_frequency(
    KEYWORDS,
    tokens_2013,
    tokens_2023,
    os.path.join(OUTPUT_DIR, "keyword_frequencies_comparison.png")
)

# Train or load Word2Vec models
print("Training or loading Word2Vec models...")
word2vec_2013 = load_or_train_word2vec(tokens_2013, "2013")
word2vec_2023 = load_or_train_word2vec(tokens_2023, "2023")

# Heatmap of Differences
print("Generating heatmap...")
heat_map(
    comparison_2013.pivot_table(values="Difference", index="Text", columns="Prob_2013"),
    "Heatmap of 2013 Differences",
    os.path.join(OUTPUT_DIR, "heatmap_2013.png")
)
heat_map(
    comparison_2023.pivot_table(values="Difference", index="Text", columns="Prob_2023"),
    "Heatmap of 2023 Differences",
    os.path.join(OUTPUT_DIR, "heatmap_2023.png")
)

print("Analysis complete. All outputs saved in the output directory.")