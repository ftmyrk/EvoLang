# main.py

import os
from utils.macros import download_dataset, generated_events, OUTPUT_DIR, GENERATED_DATA_DIR, KEYWORDS
from generate_data import generate_data
from analyze_data import analyze_data
from apply_topic_modeling import apply_topic_modeling
from semantic_shift import compute_semantic_shifts
from kl_divergence import compute_and_plot_kl_divergence

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENERATED_DATA_DIR, exist_ok=True)

# Not Ready Yet

def main():
    print("=== EvoLang: Main Pipeline ===")
    
    # Step 1: Download Datasets
    print("\nStep 1: Downloading datasets...")
    old_event_csv, new_event_csv = download_dataset()
    print(f"Datasets downloaded to: {old_event_csv} and {new_event_csv}")
    
    # Step 2: Generate Data
    print("\nStep 2: Generating data...")
    generate_data()
    print("Generated data saved to the 'generated_data/' directory.")
    
    # Step 3: Analyze Data
    print("\nStep 3: Analyzing data...")
    analyze_data()
    print("Analysis results saved to the 'outputs/' directory.")
    
    # Step 4: Apply Topic Modeling
    print("\nStep 4: Applying topic modeling...")
    data_2013, data_2023 = generated_events()
    for keyword in KEYWORDS:
        print(f"Processing topics for keyword: {keyword}")
        apply_topic_modeling(data_2013, keyword)
        apply_topic_modeling(data_2023, keyword)
    print("Topic modeling results saved.")
    
    # Step 5: Compute Semantic Shifts
    print("\nStep 5: Computing semantic shifts...")
    compute_semantic_shifts()
    print("Semantic shift analysis completed. Results saved.")
    
    # Step 6: Compute KL Divergence
    print("\nStep 6: Computing KL Divergence...")
    compute_and_plot_kl_divergence()
    print("KL Divergence analysis completed. Results saved.")
    
    print("\n=== EvoLang Pipeline Completed ===")

if __name__ == "__main__":
    main()