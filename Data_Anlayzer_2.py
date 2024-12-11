import pandas as pd
import os
import matplotlib.pyplot as plt
from utils.macros import OUTPUT_DIR, GENERATED_DATA_DIR
from utils.text_analysis_utils import generate_wordcloud, word_frequency_analysis, analyze_sentiment, visualize_sentiment, generate_word_association
from PIL import Image

OUTPUT_DIRR = os.path.join(OUTPUT_DIR, "category_related")
GENERATED_DATA_DIRR = os.path.join(GENERATED_DATA_DIR, "generated_responses")
    
def analyze_category_data(category, year):
    input_file = os.path.join(GENERATED_DATA_DIRR, f"generated_responses_{year}_{category}.csv")
    if not os.path.exists(input_file):
        print(f"Data for {category} {year} not found. Skipping...")
        return None
    
    print(f"Analyzing category: {category} {year}")
    data = pd.read_csv(input_file)

    wordcloud_dir = os.path.join(OUTPUT_DIR, "Word_Cloud")
    sentiment_dir = os.path.join(OUTPUT_DIR, "Sentiment")
    word_frequency_dir = os.path.join(OUTPUT_DIR, "Word_Frequency")
    word_association_dir = os.path.join(OUTPUT_DIR, "Word_Association")
    os.makedirs(wordcloud_dir, exist_ok=True)
    os.makedirs(sentiment_dir, exist_ok=True)
    os.makedirs(word_frequency_dir, exist_ok=True)
    os.makedirs(word_association_dir, exist_ok=True)

    generate_wordcloud(data, os.path.join(wordcloud_dir, f"Word_Cloud_{category}_{year}.png"), year)
    word_frequency_analysis(data, f"{category} {year}", os.path.join(word_frequency_dir, f"Word_Frequency_{category}_{year}.png"))
    sentiments = analyze_sentiment(data)
    visualize_sentiment(sentiments, f"{category} {year}", os.path.join(sentiment_dir, f"Sentiment_{category}_{year}.png"))
    generate_word_association(data, os.path.join(word_association_dir, f"Word_Association_{category}_{year}.png"), year)
    return data


def generate_combined_plots(year, categories, plot_type):
    input_dir = os.path.join(OUTPUT_DIR, plot_type)
    combined_output_dir = OUTPUT_DIR
    os.makedirs(combined_output_dir, exist_ok=True)

    num_categories = len(categories)
    rows = 2
    cols = (num_categories + 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i, category in enumerate(categories):
        file_path = os.path.join(input_dir, f"{plot_type}_{category}_{year}.png")
        if os.path.isfile(file_path):
            img = Image.open(file_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"{category} {year}")
        else:
            axes[i].axis("off")
            axes[i].set_title(f"No data for {category} {year}")

    for j in range(len(categories), len(axes)):
        axes[j].axis("off")

    combined_output = os.path.join(combined_output_dir, f"{plot_type}_{year}.png")
    plt.tight_layout()
    plt.savefig(combined_output)
    plt.close()
    print(f"Combined {plot_type} for {year} saved to {combined_output}")

categories = ["Health", "Technology", "Economy", "Politics", "Sports"]
years = [2011, 2021]

for year in years:
    for category in categories:
        analyze_category_data(category, year)

    for plot_type in ["Word_Cloud", "Sentiment", "Word_Frequency", "Word_Association"]:
        generate_combined_plots(year, categories, plot_type)