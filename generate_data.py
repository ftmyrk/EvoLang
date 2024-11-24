# generate_data.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import text_preprocessing_utils as preprocess_utils, text_generation_utils as gen_utils
from utils.macros import GENERATED_DATA_DIR, download_dataset
import torch

old_event_csv, new_event_csv = download_dataset()

# Model setup
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id


old_events = preprocess_utils.load_and_preprocess_dataset(old_event_csv, text_column=0)
new_events = preprocess_utils.load_and_preprocess_dataset(new_event_csv, text_column=0)

# Generate and save processed data
old_results = gen_utils.process_in_batches(old_events, model, tokenizer)
gen_utils.save_results_to_csv(old_results, f"{GENERATED_DATA_DIR}/generated_responses_2013.csv")
new_results = gen_utils.process_in_batches(new_events, model, tokenizer)
gen_utils.save_results_to_csv(new_results, f"{GENERATED_DATA_DIR}/generated_responses_2023.csv")

