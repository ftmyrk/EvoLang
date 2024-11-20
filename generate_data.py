# generate_data.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import text_preprocessing_utils as preprocess_utils
from utils import text_generation_utils as gen_utils
import torch


model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id
# path = '/home/otamy001/EvoLang/2013_year.csv'
old_event_csv = '/home/otamy001/EvoLang/Dataset/2013_year.csv'
new_event_csv = '/home/otamy001/EvoLang/Dataset/2023_year.csv'
old_events = preprocess_utils.load_and_preprocess_dataset(old_event_csv, text_column=0)
new_events = preprocess_utils.load_and_preprocess_dataset(new_event_csv, text_column=0)

old_results = gen_utils.process_in_batches(old_events, model, tokenizer)
gen_utils.save_results_to_csv(old_results, "/home/otamy001/EvoLang/generated_data/generated_responses_2013.csv")
new_results = gen_utils.process_in_batches(new_events, model, tokenizer)
gen_utils.save_results_to_csv(new_results, "/home/otamy001/EvoLang/generated_data/generated_responses_2023.csv")


