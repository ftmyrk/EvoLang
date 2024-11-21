import pandas as pd
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import os

# Paths
BASE_DIR = '/home/otamy001/EvoLang'
DATA_DIR = os.path.join(BASE_DIR, 'generated_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'word2vec_models')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
def load_data():
    data_2013 = pd.read_csv(os.path.join(DATA_DIR, 'generated_responses_2013.csv'))
    data_2023 = pd.read_csv(os.path.join(DATA_DIR, 'generated_responses_2023.csv'))
    return data_2013, data_2023

# Tokenize data
def tokenize_data(data):
    return [word for text in data['Original_Text'] for word in text.split()]

# Sentence-BERT Model
sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Word2Vec Models
def load_or_train_word2vec(data_tokens, year):
    model_path = os.path.join(MODEL_DIR, f'word2vec_model_{year}.model')
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(sentences=data_tokens, vector_size=100, window=5, min_count=2, workers=4)
        model.save(model_path)
    return model

# Load and preprocess everything
data_2013, data_2023 = load_data()
tokens_2013 = tokenize_data(data_2013)
tokens_2023 = tokenize_data(data_2023)

word2vec_model_2013 = load_or_train_word2vec(tokens_2013, 2013)
word2vec_model_2023 = load_or_train_word2vec(tokens_2023, 2023)