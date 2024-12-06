import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.macros import GENERATED_DATA_DIR, download_dataset, generated_events
from utils.text_preprocessing_utils import load_dataset
from utils.text_generation_utils import generate_text, load_model_and_tokenizer
import random
import torch
import matplotlib.pyplot as plt

# Load model and tokenizer
model_id = "meta-llama/Llama-3.2-3B"
model, tokenizer = load_model_and_tokenizer(model_id)


df_2013, df_2023 = generated_events()

# Compute log probabilities
def compute_log_probability(text, context, model, tokenizer):
    prompt = f"{context} {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    log_prob = torch.log_softmax(logits, dim=-1)
    return log_prob.sum().item()

# Compare probabilities for each document
def compare_contexts(df, year_1, year_2):
    """
    Compare log probabilities of each document under two contexts.
    """
    results = []
    for _, row in df.iterrows():
        text = row["Generated_Full_Response"]
        prob_1 = compute_log_probability(text, f"In {year_1},", model, tokenizer)
        prob_2 = compute_log_probability(text, f"In {year_2},", model, tokenizer)
        results.append({
            "Text": text,
            f"Prob_{year_1}": prob_1,
            f"Prob_{year_2}": prob_2,
            "Difference": abs(prob_1 - prob_2)
        })
    return pd.DataFrame(results)

def plot_probability_differences(comparison_df, year_1, year_2, output_path):
    """
    Plot histogram of probability differences between two contexts and save the plot as an image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(comparison_df["Difference"], bins=50, alpha=0.7, label=f"{year_1} vs {year_2}")
    plt.xlabel("Probability Difference")
    plt.ylabel("Frequency")
    plt.title(f"Probability Differences Between {year_1} and {year_2}")
    plt.legend()
    plt.tight_layout()

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid memory issues

# Perform comparisons
comparison_2013 = compare_contexts(df_2013, 2013, 2023)
comparison_2023 = compare_contexts(df_2023, 2013, 2023)

# Save results
comparison_2013.to_csv("/home/otamy001/EvoLang/outputs/comparison_2013.csv", index=False)
comparison_2023.to_csv("/home/otamy001/EvoLang/outputs/comparison_2023.csv", index=False)


# Create output directory for plots
PLOT_OUTPUT_DIR = "/home/otamy001/EvoLang/new_outputs/plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Plot differences and save them
plot_probability_differences(comparison_2013, 2013, 2023, os.path.join(PLOT_OUTPUT_DIR, "probability_differences_2013_vs_2023.png"))

plot_probability_differences(comparison_2023, 2013, 2023, os.path.join(PLOT_OUTPUT_DIR, "probability_differences_2023_vs_2013.png"))