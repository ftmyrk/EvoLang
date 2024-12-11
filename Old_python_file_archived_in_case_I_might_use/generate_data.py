from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import text_preprocessing_utils as preprocess_utils
from utils.macros import GENERATED_DATA_DIR, download_dataset
import torch
from tqdm import tqdm  
import os

old_event_csv, new_event_csv = download_dataset()

model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id

old_events = preprocess_utils.load_and_preprocess_dataset(old_event_csv, text_column=0)
new_events = preprocess_utils.load_and_preprocess_dataset(new_event_csv, text_column=0)

import random

def generate_responses(events, target_year, output_file, sample_size=200):
    # Randomly sample `sample_size` articles
    sampled_events = random.sample(events, min(sample_size, len(events)))
    results = []

    print(f"Processing {len(sampled_events)} randomly selected articles for {target_year}...")

    for event in sampled_events:
        input_text = (
            f"Rewrite the following news article as if it was written in the style of {target_year}: {event['text']}"
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Add attention mask
            max_new_tokens=200,  # Limit the generated text to 200 tokens
            temperature=1.2,     # Increase temperature for randomness
            top_p=0.9,           # Nucleus sampling
            top_k=50,            # Limit to the top 50 tokens
            do_sample=True       # Enable sampling for diverse outputs
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"Original_Text": event["text"], "Generated_Text": generated_text})

    # Save results to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Original_Text,Generated_Text\n")
        for result in results:
            f.write(f"{result['Original_Text']},{result['Generated_Text']}\n")

    print(f"Generated responses saved to: {output_file}")
    
generate_responses(old_events, 2013, os.path.join(GENERATED_DATA_DIR, "generated_responses_2013.csv"), 200)
generate_responses(new_events, 2023, os.path.join(GENERATED_DATA_DIR, "generated_responses_2023.csv"), 200)