# semantic_analysis.py

from utils import measurement_utils
from utils.macros import word2vec_model_2013, word2vec_model_2023, sentence_bert_model, tokens_2013, tokens_2023, OUTPUT_DIR, KEYWORDS
import os
import pandas as pd

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Measure cosine similarity between 2013 and 2023 token embeddings
cosine_similarity_file = os.path.join(OUTPUT_DIR, 'cosine_similarities.csv')
measurement_utils.calculate_cosine_similarity(
    sentence_bert_model.encode(tokens_2013, convert_to_tensor=True),
    sentence_bert_model.encode(tokens_2023, convert_to_tensor=True),
    cosine_similarity_file,
)
print(f"Cosine similarity matrix saved to: {cosine_similarity_file}")

# Load similarity results and plot heatmap
similarity_df = pd.read_csv(cosine_similarity_file)
heatmap_file = os.path.join(OUTPUT_DIR, 'cosine_similarity_heatmap.png')
measurement_utils.plot_similarity_heatmap(similarity_df, OUTPUT_DIR, 'cosine_similarity_heatmap.png')
print(f"Cosine similarity heatmap saved to: {heatmap_file}")

# Keyword evolution with Word2Vec
print("\nAnalyzing keyword evolution with Word2Vec:")
keyword_similarities = measurement_utils.compare_keyword_vectors(word2vec_model_2013, word2vec_model_2023, KEYWORDS)

# Print and save similarities
results_file = os.path.join(OUTPUT_DIR, 'keyword_similarities.txt')
with open(results_file, 'w') as f:
    for keyword, similarity in keyword_similarities.items():
        if similarity is not None:
            result = f"Cosine similarity for '{keyword}' between 2013 and 2023: {similarity:.4f}"
        else:
            result = f"'{keyword}' not found in one of the models."
        print(result)
        f.write(result + '\n')

print(f"\nKeyword similarities saved to: {results_file}")