# text_generation_utils.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import csv
from datetime import datetime, timedelta
import random
import re

categories = {
    "Health": ["vaccine", "COVID-19", "pandemic", "lockdown", "hospital", "virus"],
    "Technology": ["AI", "artificial intelligence", "smartphone", "tablet", "robot", "5G", "cybersecurity", "cloud"],
    "Economy": ["stocks", "market", "economy", "inflation", "finance", "GDP", "real estate", "crypto", "bitcoin"],
    "Politics": ["election", "policy", "government", "senate", "president"],
    "Sports": ["soccer", "football", "Olympics", "basketball", "cricket"]
}

# Load language model and tokenizer
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)  
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_length,
        temperature=0.9,
        num_return_sequences=1,
        top_p=0.85,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,  
        attention_mask=inputs.get("attention_mask", None),  
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_key_response(generated_response, prompt):
    return generated_response.replace(prompt, "")

def assign_category(text):
    text_lower = text.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if re.search(rf"\b{keyword.lower()}\b", text_lower):
                return category
    return "Other"

# # Summarization pipeline
def load_summarization_model():
    model_id = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, tokenizer

def summarize_text(text, model, tokenizer, max_length=150, min_length=30):
    text = text[:1024] 
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=150).to(model.device)
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=2.0, 
        num_beams=4
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.replace("summarize:", "").strip()

def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Original_Text", "Generated_Full_Response", "Generated_Only_Response"])
        writer.writeheader()
        writer.writerows(results)
        
def generate_random_date(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    random_days = random.randint(0, (end_date - start_date).days)
    return start_date + timedelta(days=random_days)
