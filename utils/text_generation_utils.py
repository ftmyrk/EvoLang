# text_generation_utils.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import csv

# Load language model and tokenizer
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).to(model.device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_length,
        temperature=1.0,
        top_p=0.85,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_key_response(generated_response, prompt):
    return generated_response.replace(prompt, "")
# # Summarization pipeline
# def load_summarization_model():
#     model_id = "facebook/bart-large-cnn"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
#     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     return model, tokenizer

# def summarize_text(text, model, tokenizer, max_length=150):
#     text = text[:1024]  # Ensure text isn't too long for summarization
#     inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=150).to(model.device)
#     summary_ids = model.generate(
#         inputs['input_ids'], 
#         max_length=max_length, 
#         min_length=30, 
#         length_penalty=2.0, 
#         num_beams=4
#     )
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Original_Text", "Generated_Full_Response", "Generated_Only_Response"])
        writer.writeheader()
        writer.writerows(results)
        
