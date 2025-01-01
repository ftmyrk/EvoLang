import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.text_generation_utils import generate_text, extract_key_response, assign_category, load_summarization_model, summarize_text, generate_random_date
from utils.macros import load_dataset_event_csv
from utils.text_preprocessing_utils import preprocess_text
import random
from multiprocessing import Process, set_start_method

set_start_method("spawn", force=True)

MODEL_ID = "meta-llama/Llama-3.2-3B"
output_dir = "/home/otamy001/EvoLang/generated_data/"
os.makedirs(output_dir, exist_ok=True)

summarization_model, summarization_tokenizer = load_summarization_model()

def setup_model_on_device(device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto").to(device)
    return model, tokenizer, device

def compute_log_probability(text, model, tokenizer, max_length=1024):
    text = text[:max_length]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    del inputs
    torch.cuda.empty_cache()
    return -outputs.loss.item()  # Negative log-likelihood

def compute_normalized_log_probability(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    num_tokens = len(inputs["input_ids"][0])
    return compute_log_probability(text, model, tokenizer) / num_tokens

def compute_conditional_log_probability(text1, text2, model, tokenizer):
    prompt = f"{text2}\n{text1}"
    return compute_normalized_log_probability(prompt, model, tokenizer)

def generate_responses(input_data, year, device_id, sample_size=5000):
    model, tokenizer, device = setup_model_on_device(device_id)
    input_data = input_data.sample(min(sample_size, len(input_data)), random_state=42)
    results = []
    print(f"Generating responses for {year} on GPU {device_id}...")
    percent_step = max(1, len(input_data) // 20)
    progress = 0

    for i, row in input_data.iterrows():
        if i % percent_step == 0:
            progress += 5
            print(f"Progress on GPU {device_id}: {progress}%")

        try:
            prompt_date = generate_random_date(year).strftime("%d %B %Y")
            text = row.get("Text") if year == 2013 else row.get("article", "")
            words_count = len(text.split())
            min_length_word = words_count / 5
            if min_length_word > 75:
                min_length_word = 75
            elif min_length_word < 30:
                min_length_word = 30
            text = preprocess_text(text ,1024)
            summarized_text = summarize_text(text, summarization_model, summarization_tokenizer, max_length=250, min_length=int(min_length_word))
            prompt = f"On {prompt_date}, {summarized_text}"

            generated_full_response = generate_text(prompt, model, tokenizer, max_length=300)
            extracted_key_response = extract_key_response(generated_full_response, prompt)
            category = assign_category(summarized_text)

            logp_article_summary = compute_conditional_log_probability(text, summarized_text, model, tokenizer)
            logp_article_response = compute_conditional_log_probability(text, extracted_key_response, model, tokenizer)
            logp_summary_response = compute_conditional_log_probability(summarized_text, extracted_key_response, model, tokenizer)

            results.append({
                "Original_Text": text,
                "Summarized_Text": summarized_text,
                "Generated_Full_Response": generated_full_response,
                "Extracted_Key_Response": extracted_key_response,
                "Category": category,
                "LogP(Article | Summary)": logp_article_summary,
                "LogP(Article | Generated_Response)": logp_article_response,
                "LogP(Summary | Generated_Response)": logp_summary_response,
            })

        except RuntimeError as e:
            print(f"Error encountered on GPU {device_id}: {e}")
            torch.cuda.empty_cache()

    return pd.DataFrame(results)

def process_on_device(input_data, year, device_id, output_file, sample_size=10000):
    results_df = generate_responses(input_data, year, device_id, sample_size)
    results_df.to_csv(output_file, index=False)
    print(f"Responses for {year} saved to {output_file}")

if __name__ == '__main__':
    input_2013, input_2023 = load_dataset_event_csv()
    processes = []
    datasets = {
        2013: (input_2013, os.path.join(output_dir, "generated_responses_2013.csv")),
        2023: (input_2023, os.path.join(output_dir, "generated_responses_2023.csv"))
    }
    device_ids = [0, 1]  # GPU IDs

    for i, (year, (input_data, output_file)) in enumerate(datasets.items()):
        if input_data is not None:
            process = Process(target=process_on_device, args=(input_data, year, device_ids[i % len(device_ids)], output_file))
            process.start()
            processes.append(process)
        else:
            print(f"Dataset for {year} not found. Skipping...")

    for process in processes:
        process.join()