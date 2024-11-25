# semantic_analysis.py

from utils.macros import word2vec_model_2013, word2vec_model_2023, tokens_2013, tokens_2023, OUTPUT_DIR, KEYWORDS, GENERATED_DATA_DIR
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')
# Function to calculate and save cosine similarity
def calculate_cosine_similarity(embeddings1, embeddings2):
    output_file = os.path.join(GENERATED_DATA_DIR, 'cosine_similarities.csv')
    cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
    pd.DataFrame(cosine_similarities.cpu().numpy()).to_csv(output_file, index=False)
    print(f"Cosine similarity matrix saved to: {output_file}")
    return output_file

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

    # Automatically save results
    output_file = os.path.join(OUTPUT_DIR, 'keyword_similarities.txt')
    with open(output_file, 'w') as f:
        for keyword, similarity in results.items():
            if similarity is not None:
                result = f"Cosine similarity for '{keyword}' between 2013 and 2023: {similarity:.4f}"
            else:
                result = f"'{keyword}' not found in one of the models."
            print(result)
            f.write(result + '\n')

    print(f"\nKeyword similarities saved to: {output_file}")
    return output_file

# Function to plot and save a heatmap of similarity data
def plot_similarity_heatmap(similarity_df):
    output_file = os.path.join(OUTPUT_DIR, 'cosine_similarity_heatmap.png')
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, cmap='viridis')
    plt.title('Sample Cosine Similarity Heatmap')
    plt.savefig(output_file)
    plt.close()
    print(f"Cosine similarity heatmap saved to: {output_file}")
    return output_file


# Convert token lists into strings
texts_2013 = [' '.join(tokens) for tokens in tokens_2013]
texts_2023 = [' '.join(tokens) for tokens in tokens_2023]

# Measure cosine similarity between 2013 and 2023 token embeddings
cosine_similarity_file = calculate_cosine_similarity(
    sentence_bert_model.encode(texts_2013, convert_to_tensor=True, show_progress_bar=True),
    sentence_bert_model.encode(texts_2023, convert_to_tensor=True, show_progress_bar=True),
)

# Load similarity results and plot heatmap
similarity_df = pd.read_csv(cosine_similarity_file)
plot_similarity_heatmap(similarity_df)

# Keyword evolution with Word2Vec
print("\nAnalyzing keyword evolution with Word2Vec:")
compare_keyword_vectors(word2vec_model_2013, word2vec_model_2023, KEYWORDS)