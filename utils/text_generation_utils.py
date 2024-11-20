# text_generation_utils.py

import torch
from transformers import pipeline
import csv
# tokenizer = None
# model = None

def truncate_text(input_text, max_length=256):
    tokens = tokenizer.encode(input_text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens)

# Generate text based on truncated input
def generate_text(input_text, model, tokenizer, max_new_tokens=30):
    try:
        generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return generation_pipeline(input_text, max_new_tokens=max_new_tokens)[0]['generated_text']
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def process_in_batches(events, model, tokenizer, batch_size=100):
    results = []
    for i in range(0, len(events), batch_size):
        print(f"Processing batch {i // batch_size + 1} of {len(events) // batch_size + 1}")
        batch = events[i:i + batch_size]
        for event in batch:
            generated_text = generate_text(event["text"], model, tokenizer)
            if generated_text:
                results.append((event["text"], generated_text))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Original_Text", "Generated_Text"])
        for original, generated in results:
            writer.writerow([original, generated])

