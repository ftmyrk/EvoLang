# visualize_word_associations.py

import os
import matplotlib.pyplot as plt
from utils.macros import word2vec_model_2013, word2vec_model_2023, OUTPUT_DIR, KEYWORDS

output_dir = os.path.join(OUTPUT_DIR, 'word_associated_graph')
os.makedirs(output_dir, exist_ok=True)

# Function to plot word associations
def plot_word_associations(model, keyword, top_n=10, output_file=None):
    """Generate a bar chart for top word associations for a given keyword."""
    if keyword in model.wv:
        similar_words = model.wv.most_similar(keyword, topn=top_n)
        words, similarities = zip(*similar_words)
        
        plt.figure(figsize=(10, 6))
        plt.barh(words, similarities, color='skyblue')
        plt.gca().invert_yaxis()
        plt.title(f"Top {top_n} Words Associated with '{keyword}'")
        plt.xlabel("Cosine Similarity")
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
            print(f"Saved plot: {output_file}")
        else:
            plt.show()
    else:
        print(f"'{keyword}' not found in the model.")

# Generate plots for each keyword
for keyword in KEYWORDS:
    print(f"\nWord associations for '{keyword}' in 2013:")
    plot_word_associations(
        word2vec_model_2013,
        keyword,
        output_file=os.path.join(output_dir, f'word_associations_2013_{keyword}.png')
    )
    
    print(f"\nWord associations for '{keyword}' in 2023:")
    plot_word_associations(
        word2vec_model_2023,
        keyword,
        output_file=os.path.join(output_dir, f'word_associations_2023_{keyword}.png')
    )