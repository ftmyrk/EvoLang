# analyze_llama_results.py

import os
import pandas as pd
from difflib import SequenceMatcher
from utils.macros import GENERATED_DATA_DIR, OUTPUT_DIR
from matplotlib import pyplot as plt

input_file = os.path.join(GENERATED_DATA_DIR, "llama_experiment_results.csv")
output_file = os.path.join(OUTPUT_DIR, "response_differences.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to compute text similarity
def text_similarity(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0  # Handle cases where responses are missing
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

results = pd.read_csv(input_file)

differences = []
with open(output_file, "w") as f:
    for index, row in results.iterrows():
        question = row["Question"]
        response_2013 = row.get("Response_2013", "")
        response_2023 = row.get("Response_2023", "")
        similarity = text_similarity(response_2013, response_2023)
        differences.append((question, similarity))
        f.write(f"Question: {question}\n")
        f.write(f"Response 2013: {response_2013}\n")
        f.write(f"Response 2023: {response_2023}\n")
        f.write(f"Similarity: {similarity:.4f}\n")
        f.write("\n")

differences.sort(key=lambda x: x[1])

questions, similarities = zip(*differences) if differences else ([], [])
plt.figure(figsize=(10, 6))
plt.barh(questions, similarities, color="skyblue")
plt.xlabel("Similarity Score")
plt.ylabel("Questions")
plt.title("LLaMA Response Similarity: 2013 vs 2023")
plt.tight_layout()
similarity_plot_path = os.path.join(OUTPUT_DIR, "response_similarity_comparison.png")
plt.savefig(similarity_plot_path)
plt.close()

if differences:
    print(f"Differences saved to {output_file}")
    print(f"Similarity comparison plot saved to {similarity_plot_path}")
else:
    print("No differences to analyze. Ensure the input data is correct.")