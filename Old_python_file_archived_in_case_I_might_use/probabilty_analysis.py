import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.macros import GENERATED_DATA_DIR, OUTPUT_DIR
 
GENERATED_DATA_DDIR = os.path.join(GENERATED_DATA_DIR, 'generated_responses')
OUTPUT_DIRR = os.path.join(OUTPUT_DIR, "category_related")
MODEL_ID = "meta-llama/Llama-3.2-3B"
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def compute_log_probability(text, context, model, tokenizer):
    prompt = f"{context} {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    log_prob = torch.log_softmax(logits, dim=-1)
    return log_prob.sum().item()

def analyze_category_probabilities(category, year):
    input_path = os.path.join(GENERATED_DATA_DDIR, f"generated_responses_{year}_{category}.csv")
    output_path = os.path.join(OUTPUT_DIRR, f"probabilities_{year}_{category}.csv")

    if not os.path.exists(input_path):
        print(f"File for category '{category}' and year {year} not found. Skipping...")
        return

    print(f"Analyzing probabilities for {category} ({year})...")
    df = pd.read_csv(input_path)
    context = f"In {year},"
    results = []

    for _, row in df.iterrows():
        text = row["Generated_Full_Response"]
        prob = compute_log_probability(text, context, model, tokenizer)
        results.append({"Text": text, "Year": year, "Category": category, "Log_Probability": prob})

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Probabilities for {category} ({year}) saved to {output_path}")

categories = ["Health", "Technology", "Economy", "Politics", "Sports"]
years = [2011, 2021]

for year in years:
    for category in categories:
        analyze_category_probabilities(category, year)