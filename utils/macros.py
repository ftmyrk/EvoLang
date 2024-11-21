import os
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = '/home/otamy001/EvoLang'
DATA_DIR = os.path.join(BASE_DIR, 'Dataset')
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'generated_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'word2vec_models')

# Ensure directories exist
os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load preprocessed datasets
def load_data():
    old_data = pd.read_csv(os.path.join(GENERATED_DATA_DIR, 'generated_responses_2013.csv'))
    new_data = pd.read_csv(os.path.join(GENERATED_DATA_DIR, 'generated_responses_2023.csv'))
    return old_data, new_data

# Tokenize datasets
def tokenize_data(data):
    return [text.split() for text in data['Original_Text'].tolist()]

# Preload data and tokens
old_data, new_data = load_data()
tokens_2013 = tokenize_data(old_data)
tokens_2023 = tokenize_data(new_data)

# Sentence-BERT model
sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Word2Vec models
def load_or_train_word2vec(tokens, year):
    model_path = os.path.join(MODEL_DIR, f'word2vec_model_{year}.model')
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)
    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    return model

word2vec_model_2013 = load_or_train_word2vec(tokens_2013, 2013)
word2vec_model_2023 = load_or_train_word2vec(tokens_2023, 2023)