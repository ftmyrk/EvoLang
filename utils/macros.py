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
WORD_ASSOC_GRAPH_DIR = os.path.join(OUTPUT_DIR, "word_associated_graph")

for directory in [DATA_DIR, GENERATED_DATA_DIR, OUTPUT_DIR, MODEL_DIR, WORD_ASSOC_GRAPH_DIR]:
    os.makedirs(directory, exist_ok=True)

OLD_EVENT_LINK = "https://drive.google.com/uc?id=10nzlFF83IGoLDVlFILwtVBVW9TPaeL1m"
NEW_EVENT_LINK = "https://drive.google.com/uc?id=10sLum2gntV-notnNUVsOMOIIqvNvtNrj"

OLD_EVENT_FILE = os.path.join(DATA_DIR, "2013_year.csv")
NEW_EVENT_FILE = os.path.join(DATA_DIR, "2023_year.csv")

KEYWORDS = ['economy', 'policy', 'shares', 'technology', 'market']

def download_with_progress(url, output_file):
    def download_callback(current, total):
        progress_bar.update(current - progress_bar.n)

    print(f"Downloading {output_file}...")
    with tqdm(total=1, unit="B", unit_scale=True, desc="Downloading") as progress_bar:
        gdown.download(url, output_file, quiet=False, callback=download_callback)

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


# Functions to load datasets
def generated_events():
    old_data_path = os.path.join(GENERATED_DATA_DIR, 'generated_responses_2013.csv')
    new_data_path = os.path.join(GENERATED_DATA_DIR, 'generated_responses_2023.csv')
    
    if not os.path.exists(old_data_path):
        raise FileNotFoundError(f"File not found: {old_data_path}")
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"File not found: {new_data_path}")
    
    old_data = pd.read_csv(old_data_path)
    new_data = pd.read_csv(new_data_path)
    
    old_processed = [{"text": text} for text in old_data["Original_Text"]]
    new_processed = [{"text": text} for text in new_data["Original_Text"]]
    
    return old_data, new_data, old_processed, new_processed

# Tokenize text data into word lists
def tokenize_data(data):
    return [text.split() for text in data['Original_Text'].tolist()]

# Word2Vec models
def load_or_train_word2vec(tokens, year):
    model_path = os.path.join(MODEL_DIR, f'word2vec_model_{year}.model')
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)
    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    return model

# Preloads datasets and tokens
def preload_datasets_and_models():
    download_dataset()
    old_data, new_data, _, _ = generated_events()
    tokens_2013 = tokenize_data(old_data)
    tokens_2023 = tokenize_data(new_data)
    word2vec_model_2013 = load_or_train_word2vec(tokens_2013, 2013)
    word2vec_model_2023 = load_or_train_word2vec(tokens_2023, 2023)
    return old_data, new_data, tokens_2013, tokens_2023, word2vec_model_2013, word2vec_model_2023

old_data, new_data, tokens_2013, tokens_2023, word2vec_model_2013, word2vec_model_2023 = preload_datasets_and_models()