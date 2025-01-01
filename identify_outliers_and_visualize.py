import pandas as pd
import numpy as np
import random
from utils.macros import OUTPUT_DIR, GENERATED_DATA_DIR_2013, GENERATED_DATA_DIR_2023
import matplotlib.pyplot as plt
import seaborn as sns
from utils.text_analysis_utils import generate_wordcloud, clean_and_prepare_text
from scipy.stats import entropy
from nltk.tokenize import word_tokenize

generated_2013 = pd.read_csv(GENERATED_DATA_DIR_2013)
generated_2023 = pd.read_csv(GENERATED_DATA_DIR_2023)

log_prob_cols = [
    "LogP(Article | Summary)",
    "LogP(Article | Generated_Response)",
    "LogP(Summary | Generated_Response)"
]

def identify_outliers(data, threshold_quantile=0.05):
    thresholds = {}
    for col in log_prob_cols:
        thresholds[col] = data[col].quantile(threshold_quantile)
        print(f"Threshold for {col}: {thresholds[col]}")

    # Flag rows as outliers if any log probability is below the threshold
    data["Is_Outlier"] = data[log_prob_cols].lt(pd.Series(thresholds)).any(axis=1)
    return data, thresholds

print("Identifying outliers for 2013 data...")
generated_2013, thresholds_2013 = identify_outliers(generated_2013)
print("Identifying outliers for 2023 data...")
generated_2023, thresholds_2023 = identify_outliers(generated_2023)

outliers_2013 = generated_2013[generated_2013["Is_Outlier"]]
outliers_2023 = generated_2023[generated_2023["Is_Outlier"]]

outliers_2013.to_csv(f"{OUTPUT_DIR}/outliers_2013.csv", index=False)
outliers_2023.to_csv(f"{OUTPUT_DIR}/outliers_2023.csv", index=False)
print(f"Outliers saved to {OUTPUT_DIR}/outliers_2013.csv and {OUTPUT_DIR}/outliers_2023.csv.")

def sample_for_human_evaluation(data, outlier_sample_size=10, normal_sample_size=10):
    # Randomly sample outliers and normal rows
    outliers = data[data["Is_Outlier"]].sample(min(outlier_sample_size, len(data[data["Is_Outlier"]])), random_state=42)
    normals = data[~data["Is_Outlier"]].sample(min(normal_sample_size, len(data[~data["Is_Outlier"]])), random_state=42)
    return pd.concat([outliers, normals])

human_eval_2013 = sample_for_human_evaluation(generated_2013)
human_eval_2023 = sample_for_human_evaluation(generated_2023)

human_eval_2013.to_csv(f"{OUTPUT_DIR}/human_eval_2013.csv", index=False)
human_eval_2023.to_csv(f"{OUTPUT_DIR}/human_eval_2023.csv", index=False)
print(f"Human evaluation samples saved to {OUTPUT_DIR}/human_eval_2013.csv and {OUTPUT_DIR}/human_eval_2023.csv.")

def compute_kl_divergence(data1, data2):
    word_counts1 = clean_and_prepare_text(data1, "Extracted_Key_Response").split()
    word_counts2 = clean_and_prepare_text(data2, "Extracted_Key_Response").split()

    freq_dist1 = pd.Series(word_counts1).value_counts(normalize=True)
    freq_dist2 = pd.Series(word_counts2).value_counts(normalize=True)

    aligned = pd.concat([freq_dist1, freq_dist2], axis=1).fillna(0)
    kl_divergence = entropy(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return kl_divergence

def visualize_log_probabilities(data, year, output_file):
    plt.figure(figsize=(10, 6))
    for col in log_prob_cols:
        sns.histplot(data[col], kde=True, label=col)
    plt.title(f"Log Probability Distributions ({year})")
    plt.xlabel("Log Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Log probability visualization saved to {output_file}")

print("Generating Word Clouds...")
generate_wordcloud(outliers_2013, f"{OUTPUT_DIR}/wordcloud_outliers_2013.png", "Outliers 2013")
generate_wordcloud(outliers_2023, f"{OUTPUT_DIR}/wordcloud_outliers_2023.png", "Outliers 2023")

print("Computing KL Divergence...")
kl_divergence = compute_kl_divergence(outliers_2013, outliers_2023)
print(f"KL Divergence between 2013 and 2023: {kl_divergence}")

print("Visualizing Log Probabilities...")
visualize_log_probabilities(generated_2013, 2013, f"{OUTPUT_DIR}/log_probs_2013.png")
visualize_log_probabilities(generated_2023, 2023, f"{OUTPUT_DIR}/log_probs_2023.png")

def get_thresholds(data, col):
    lower_threshold = data[col].quantile(0.10)  # 10th percentile
    return lower_threshold

def filter_low_quality_data(data, thresholds):
    filtered_data = data[
        (data["LogP(Article | Summary)"] >= thresholds["summary_threshold"]) &
        (data["LogP(Article | Generated_Response)"] >= thresholds["response_threshold"]) &
        (data["LogP(Summary | Generated_Response)"] >= thresholds["summary_response_threshold"])
    ]
    return filtered_data


thresholds_2013 = {
    "summary_threshold": get_thresholds(generated_2013, "LogP(Article | Summary)"),
    "response_threshold": get_thresholds(generated_2013, "LogP(Article | Generated_Response)"),
    "summary_response_threshold": get_thresholds(generated_2013, "LogP(Summary | Generated_Response)")
}

thresholds_2023 = {
    "summary_threshold": get_thresholds(generated_2023, "LogP(Article | Summary)"),
    "response_threshold": get_thresholds(generated_2023, "LogP(Article | Generated_Response)"),
    "summary_response_threshold": get_thresholds(generated_2023, "LogP(Summary | Generated_Response)")
}

print("Filtering low-quality data...")
filtered_2013 = filter_low_quality_data(generated_2013, thresholds_2013)
filtered_2023 = filter_low_quality_data(generated_2023, thresholds_2023)

filtered_2013.to_csv(f"{OUTPUT_DIR}/filtered_2013.csv", index=False)
filtered_2023.to_csv(f"{OUTPUT_DIR}/filtered_2023.csv", index=False)

print(f"Filtered 2013 data saved to {OUTPUT_DIR}/filtered_2013.csv")
print(f"Filtered 2023 data saved to {OUTPUT_DIR}/filtered_2023.csv")


style_dictionary = {
    "formal": ["therefore", "moreover", "whereas", "thus"],
    "informal": ["like", "cool", "awesome", "yeah"],
    "positive": ["happy", "good", "excellent", "fantastic"],
    "negative": ["bad", "sad", "terrible", "horrible"],
}

def count_dict_matches(text, dictionary):
    tokens = word_tokenize(text.lower())
    counts = {category: 0 for category in dictionary.keys()}
    for token in tokens:
        for category, words in dictionary.items():
            if token in words:
                counts[category] += 1
    return counts

def analyze_with_dictionary(data, dictionary, text_column="Generated_Full_Response"):
    results = []
    for _, row in data.iterrows():
        text = row[text_column]
        counts = count_dict_matches(text, dictionary)
        counts["Text"] = text  # Keep the original text for reference
        results.append(counts)
    return pd.DataFrame(results)

filtered_2013 = pd.read_csv(f"{OUTPUT_DIR}/filtered_2013.csv")
filtered_2023 = pd.read_csv(f"{OUTPUT_DIR}/filtered_2023.csv")

analysis_2013 = analyze_with_dictionary(filtered_2013, style_dictionary)
analysis_2023 = analyze_with_dictionary(filtered_2023, style_dictionary)

analysis_2013.to_csv(f"{OUTPUT_DIR}/dictionary_analysis_2013.csv", index=False)
analysis_2023.to_csv(f"{OUTPUT_DIR}/dictionary_analysis_2023.csv", index=False)

print(f"Dictionary-based analysis for 2013 saved to {OUTPUT_DIR}/dictionary_analysis_2013.csv")
print(f"Dictionary-based analysis for 2023 saved to {OUTPUT_DIR}/dictionary_analysis_2023.csv")
