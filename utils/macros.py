import os
import pandas as pd
from gensim.models import Word2Vec
import gdown
from tqdm import tqdm

BASE_DIR = '/home/otamy001/EvoLang'
DATA_DIR = os.path.join(BASE_DIR, 'Dataset')
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'generated_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'word2vec_models')

for directory in [DATA_DIR, GENERATED_DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

OLD_EVENT_LINK = "https://drive.google.com/uc?id=10nzlFF83IGoLDVlFILwtVBVW9TPaeL1m"
NEW_EVENT_LINK = "https://drive.google.com/uc?id=10sLum2gntV-notnNUVsOMOIIqvNvtNrj"

OLD_EVENT_FILE = os.path.join(DATA_DIR, "2013_year.csv")
NEW_EVENT_FILE = os.path.join(DATA_DIR, "2023_year.csv")

GENERATED_DATA_DIR_2013 = "/home/otamy001/EvoLang/generated_data/generated_responses_2013.csv"
GENERATED_DATA_DIR_2023 = "/home/otamy001/EvoLang/generated_data/generated_responses_2023.csv"

KEYWORDS = ['economy', 'policy', 'shares', 'technology', 'market']

def download_with_progress(url, output_file):
    print(f"Downloading {output_file}...")
    with tqdm(total=1, unit="B", unit_scale=True, desc="Downloading") as progress_bar:
        gdown.download(url, output_file, quiet=False)
    print(f"Download completed: {output_file}")

# Function to download the datasets
def download_dataset():
    print("Downloading 2013 dataset...")
    if not os.path.exists(OLD_EVENT_FILE):
        download_with_progress(OLD_EVENT_LINK, OLD_EVENT_FILE)
    else:
        print(f"2013 dataset already exists at {OLD_EVENT_FILE}")

    print("Downloading 2023 dataset...")
    if not os.path.exists(NEW_EVENT_FILE):
        download_with_progress(NEW_EVENT_LINK, NEW_EVENT_FILE)
    else:
        print(f"2023 dataset already exists at {NEW_EVENT_FILE}")

    return OLD_EVENT_FILE, NEW_EVENT_FILE

def load_dataset_event_csv():
    pd1 = pd.read_csv(OLD_EVENT_FILE)
    pd2 = pd.read_csv(NEW_EVENT_FILE)
    event_2013 = pd.DataFrame(pd1, columns=["Text"]) 
    event_2023 = pd.DataFrame(pd2, columns=["article"]) 
    return event_2013, event_2023
# Functions to load datasets
def generated_events():
    old_data_path = "/home/otamy001/EvoLang/generated_data/generated_responses_2013.csv"
    new_data_path = "/home/otamy001/EvoLang/generated_data/generated_responses_2023.csv"
    
    if not os.path.exists(old_data_path):
        print(f"Generated data file not found: {old_data_path}. Ensure you have run 'generate_data.py'.")
        return None, None
    if not os.path.exists(new_data_path):
        print(f"Generated data file not found: {new_data_path}. Ensure you have run 'generate_data.py'.")
        return None, None

    # Load existing data
    return pd.read_csv(old_data_path), pd.read_csv(new_data_path)

# Tokenize text data into word lists
def tokenize_data(data):
    return [text.split() for text in data['Text'].tolist()]

# Word2Vec models
def load_or_train_word2vec(tokens, year):

    model_path = os.path.join(MODEL_DIR, f'word2vec_model_{year}.model')
    if os.path.exists(model_path):
        print(f"Loading Word2Vec model for {year} from {model_path}")
        return Word2Vec.load(model_path)
    
    if tokens is None or len(tokens) == 0:
        print(f"No tokens provided for training Word2Vec model for {year}.")
        return None

    print(f"Training Word2Vec model for {year}...")
    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    return model

# def preload_datasets_and_models():
#     old_data, new_data = generated_events()

#     if old_data is None or new_data is None:
#         print("Generated data is missing. Skipping further initialization.")
#         return None, None, None, None, None, None

#     tokens_2013 = tokenize_data(old_data)
#     tokens_2023 = tokenize_data(new_data)

#     word2vec_model_2013 = load_or_train_word2vec(tokens_2013, "2013")
#     word2vec_model_2023 = load_or_train_word2vec(tokens_2023, "2023")

#     return old_data, new_data, tokens_2013, tokens_2023, word2vec_model_2013, word2vec_model_2023

# old_data, new_data, tokens_2013, tokens_2023, word2vec_model_2013, word2vec_model_2023 = preload_datasets_and_models()