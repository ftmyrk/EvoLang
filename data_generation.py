import os
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.text_generation_utils import generate_text, extract_key_response

input_dir = "/home/otamy001/EvoLang/generated_data/category_related/"
output_dir = "/home/otamy001/EvoLang/generated_data/generated_responses/"
os.makedirs(output_dir, exist_ok=True)

MODEL_ID = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def format_date(date_str):
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    return date_obj.strftime("%d %B %Y")

def generate_responses(input_file, output_file, year, category, sample_size=1000):
    df = pd.read_csv(input_file)
    if len(df) < sample_size:
        print(f"Warning: The dataset for {category} ({year}) has fewer rows ({len(df)}) than the requested sample size ({sample_size}). Adjusting sample size.")
        sample_size = len(df)
    
    df = df.sample(sample_size, random_state=42)
    results = []

    print(f"Generating responses for {category} ({year})...")
    total_rows = len(df)
    percent_step = max(1, total_rows // 20) 
    progress = 0

    for i, row in df.iterrows():
        if i % percent_step == 0:
            progress += 5
            print(f"Progress: {progress}%")

        formatted_date = format_date(str(row["Date"]))
        prompt = f"{formatted_date} - {row['Headline']}"

        generated_full_response = generate_text(prompt, model, tokenizer, max_length=200)
        extracted_key_response = extract_key_response(generated_full_response, prompt)

        results.append({
            "Original_Text": row["Headline"],
            "Formatted_Date": formatted_date,
            "Generated_Full_Response": generated_full_response,
            "Extracted_Key_Response": extracted_key_response,
            "Category": category
        })

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Generated responses saved to {output_file}")

categories = ["Health", "Technology", "Economy", "Politics", "Sports"]
years = [2011, 2021]

for year in years:
    for category in categories:
        input_path = os.path.join(input_dir, f"{year}_{category}_categorized.csv")
        output_path = os.path.join(output_dir, f"generated_responses_{year}_{category}.csv")
        generate_responses(input_path, output_path, year, category, sample_size=1000)