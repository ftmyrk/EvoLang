# kl divergence.py

from scipy.stats import entropy
from collections import Counter
import numpy as np
from utils.macros import tokens_2013, tokens_2023, OUTPUT_DIR
from utils.text_analysis_utils import plot_bar_chart

# KL Divergence Calculation
def compute_kl_divergence(distribution1, distribution2):
    return entropy(distribution1, distribution2)

# Word Distribution
def get_word_distribution(tokens):
    # Flatten the list of lists
    flat_tokens = [word for sentence in tokens for word in sentence]
    word_counts = Counter(flat_tokens)
    total_count = sum(word_counts.values())
    return {word: count / total_count for word, count in word_counts.items()}

# Compute distributions
dist_2013 = get_word_distribution(tokens_2013)
dist_2023 = get_word_distribution(tokens_2023)

# Smooth and compute KL
all_words = set(dist_2013.keys()).union(set(dist_2023.keys()))
epsilon = 1e-10
dist1 = np.array([dist_2013.get(word, epsilon) for word in all_words])
dist2 = np.array([dist_2023.get(word, epsilon) for word in all_words])
kl_div = compute_kl_divergence(dist1, dist2)

# Save results
with open(f'{OUTPUT_DIR}/kl_divergence.txt', 'w') as f:
    f.write(f"KL Divergence: {kl_div}\n")

# Plot top contributors
contributions = np.abs(dist1 - dist2) * np.log(dist1 / dist2)
top_contributors = dict(sorted(zip(all_words, contributions), key=lambda x: x[1], reverse=True)[:10])
plot_bar_chart(top_contributors, 'Top Words Contributing to KL Divergence', 'Contribution', 'Words', f'{OUTPUT_DIR}/kl_divergence_contributions.png')