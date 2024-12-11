# data_generater.py

import os
import pandas as pd
from utils.macros import GENERATED_DATA_DIR, download_dataset
from utils.text_preprocessing_utils import load_dataset
from utils.text_generation_utils import load_model_and_tokenizer, generate_text, load_summarization_model, summarize_text, save_results_to_csv
import random
import torch

old_csv, new_csv = download_dataset()
df_2013 = load_dataset(old_csv)
df_2023 = load_dataset(new_csv)

model_id = "meta-llama/Llama-3.2-3B"
model, tokenizer = load_model_and_tokenizer(model_id)
summ_model, summ_tokenizer = load_summarization_model()

# prompts = [
#     "Discuss the impact of social media on modern communication.",
#     "Describe recent advancements in renewable energy technologies.",
#     "Explore the challenges of space exploration.",
#     "Analyze the global effects of economic inflation.",
#     "Explain the role of artificial intelligence in healthcare.",
# ]
def extract_only_response(generated_text, year):
    return generated_text.replace(f"In {year},", "").strip()



def generate_responses(df, year, output_file):
    results = []
    print(f"Generating responses for {year}...")
    for i in range(5, 50001, 5): 
        if i % 500 == 0:
            print(i) 
        if i > len(df):
            break
        row = df.iloc[i - 1]
        if year == 2013:
            original_text = row["Text"]
            # preprocessed_text = row["Preprocessed_Text"]
        elif year == 2023:
            original_text = row["article"]
            # preprocessed_text = row["Preprocessed_Text"]
        truncated_text = summarize_text(original_text, summ_model, summ_tokenizer, max_length=150)
        # truncated_text = preprocess_text(preprocessed_text, max_tokens=1024)
        prompt = f"In {year}, {truncated_text}"
        generated_full_response = generate_text(prompt, model, tokenizer)

        core_response = generated_full_response.replace(f"In {year},", "").strip()

        results.append({
            "Original_Text": truncated_text,
            "Generated_Full_Response": generated_full_response,
            "Generated_Only_Response": core_response
        })

    output_path = os.path.join(GENERATED_DATA_DIR, output_file)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Generated responses saved to {output_path}")

# generate_responses(df_2013, 2013, "generated_responses_2013.csv")
generate_responses(df_2023, 2023, "generated_responses_2023.csv")