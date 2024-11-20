import pandas as pd
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate and save cosine similarity
def calculate_cosine_similarity(embeddings1, embeddings2, output_file):
    cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
    pd.DataFrame(cosine_similarities.cpu().numpy()).to_csv(output_file, index=False)

# Function to compare keyword vectors between two Word2Vec models
def compare_keyword_vectors(model_2013, model_2023, keywords):
    results = {}
    for keyword in keywords:
        if keyword in model_2013.wv and keyword in model_2023.wv:
            vec_2013 = model_2013.wv[keyword].reshape(1, -1)
            vec_2023 = model_2023.wv[keyword].reshape(1, -1)
            similarity = cosine_similarity(vec_2013, vec_2023)[0][0]
            results[keyword] = similarity
        else:
            results[keyword] = None
    return results

# Function to plot and save a heatmap of similarity data
def plot_similarity_heatmap(similarity_df, output_file, sample_size=1000):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df.sample(n=sample_size), cmap='viridis')
    plt.title('Sample Cosine Similarity Heatmap')
    plt.savefig(output_file)
    plt.close()