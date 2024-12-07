# llama_experiment.py

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.macros import GENERATED_DATA_DIR, generated_events, GENERATED_DATA_DIR
from tqdm import tqdm

experiment_output_file = os.path.join(GENERATED_DATA_DIR, "llama_experiment_results.csv")
os.makedirs(GENERATED_DATA_DIR, exist_ok=True)

QUESTIONS = [
    "The president of the United States is...",
    "Popular platforms for sharing music include...",
    "The most discussed technological advancement is...",
    "Global concerns regarding climate change focus on...",
    "The most influential company in the tech industry is...",
    "People primarily use the internet for...",
    "The term 'AI' is commonly associated with...",
    "The leading social media platform is...",
    "Artificial intelligence has...",
    "Space exploration is spearheaded by...",
    "People in twitter canceled ..."
]

print("Loading LLaMA model...")
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id

data_2013, data_2023 = generated_events()
context_2013 = " ".join(data_2013['Original_Text'].tolist()[:5])[:1000]
context_2023 = " ".join(data_2023['Original_Text'].tolist()[:5])[:1000]

results = []
print("Generating responses...")
for question in tqdm(QUESTIONS, desc="Processing Questions"):
    inputs_2013 = f"\nContext:\n{context_2013}\n\nQuestion:\n{question}\n\nAnswer:"
    inputs_2023 = f"\nContext:\n{context_2023}\n\nQuestion:\n{question}\n\nAnswer:"

    tokens_2013 = tokenizer(inputs_2013, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    tokens_2023 = tokenizer(inputs_2023, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    
    response_2013 = model.generate(**tokens_2013, max_new_tokens=50)
    response_2023 = model.generate(**tokens_2023, max_new_tokens=50)

    decoded_2013 = tokenizer.decode(response_2013[0], skip_special_tokens=True).strip()
    decoded_2023 = tokenizer.decode(response_2023[0], skip_special_tokens=True).strip()
    
    decoded_2013 = decoded_2013.split("Answer:")[-1].strip()
    decoded_2023 = decoded_2023.split("Answer:")[-1].strip()

    results.append({
        "Question": question,
        "Response_2013": decoded_2013,
        "Response_2023": decoded_2023
    })

df_results = pd.DataFrame(results)
df_results.to_csv(experiment_output_file, index=False)

print(f"LLaMA experiment results saved to {experiment_output_file}")