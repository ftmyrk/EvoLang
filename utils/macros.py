import os
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

BASE_DIR = '/home/otamy001/EvoLang'
DATA_DIR = os.path.join(BASE_DIR, 'Dataset')
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'generated_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'word2vec_models')
# Keywords for analysis
KEYWORDS = ['economy', 'policy', 'shares', 'technology', 'market']

os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Functions to load raw datasets
def load_raw_data(year):
    """Load raw dataset for a given year (2013 or 2023)."""
    data_path = os.path.join(DATA_DIR, f'{year}_year.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    return pd.read_csv(data_path)

# Functions to load datasets
def old_events():
    """Load preprocessed 2013 dataset."""
    old_data_path = os.path.join(GENERATED_DATA_DIR, 'generated_responses_2013.csv')
    if not os.path.exists(old_data_path):
        raise FileNotFoundError(f"File not found: {old_data_path}")
    old_data = pd.read_csv(old_data_path)
    return [{"text": text} for text in old_data["Original_Text"]]

def new_events():
    """Load preprocessed 2023 dataset."""
    new_data_path = os.path.join(GENERATED_DATA_DIR, 'generated_responses_2023.csv')
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"File not found: {new_data_path}")
    new_data = pd.read_csv(new_data_path)
    return [{"text": text} for text in new_data["Original_Text"]]


# Lazy-loading helper for CSV paths
def dataset_path(year):
    """Get the dataset path for a specific year."""
    return os.path.join(GENERATED_DATA_DIR, f'generated_responses_{year}.csv')

# Tokenize datasets
def tokenize_data(data):
    """Tokenize text data into word lists."""
    return [text.split() for text in data['Original_Text'].tolist()]

# Sentence-BERT model
sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Word2Vec models
def load_or_train_word2vec(tokens, year):
    """Load or train a Word2Vec model for a given year."""
    model_path = os.path.join(MODEL_DIR, f'word2vec_model_{year}.model')
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)
    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    return model

# Preload datasets and tokens
def preload_datasets_and_models():
    """Preload datasets, tokens, and models."""
    try:
        old_data = pd.read_csv(dataset_path(2013))
        new_data = pd.read_csv(dataset_path(2023))
    except FileNotFoundError as e:
        raise RuntimeError(f"Required dataset is missing. Ensure preprocessing has been completed: {e}")
    
    tokens_2013 = tokenize_data(old_data)
    tokens_2023 = tokenize_data(new_data)

    word2vec_model_2013 = load_or_train_word2vec(tokens_2013, 2013)
    word2vec_model_2023 = load_or_train_word2vec(tokens_2023, 2023)

    return old_data, new_data, tokens_2013, tokens_2023, word2vec_model_2013, word2vec_model_2023

# Preload everything for direct access
old_data, new_data, tokens_2013, tokens_2023, word2vec_model_2013, word2vec_model_2023 = preload_datasets_and_models()