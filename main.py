from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import torch
import csv
from datasets import DatasetDict, Dataset
import sys

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model with 'accelerate' for efficient loading
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model.to(device)