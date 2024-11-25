# sementic_shift.py

from utils.macros import tokens_2013, tokens_2023, generated_events, OUTPUT_DIR, KEYWORDS
from sentence_transformers import util, SentenceTransformer
from utils.text_analysis_utils import plot_keyword_frequency
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)
sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')
data_2013, data_2023, _, _ = generated_events()
# Function to compute semantic similarity for a keyword
def compute_semantic_shift(keyword):
    """
    Compute the semantic similarity between sentences containing the keyword in 2013 and 2023 datasets.
    """
    # Filter sentences containing the keyword
    sentences_2013 = [text for text in data_2013['Original_Text'] if keyword in text]
    sentences_2023 = [text for text in data_2023['Original_Text'] if keyword in text]
    
    if sentences_2013 and sentences_2023:
        # Encode sentences using Sentence-BERT
        emb_2013 = sentence_bert_model.encode(sentences_2013, convert_to_tensor=True)
        emb_2023 = sentence_bert_model.encode(sentences_2023, convert_to_tensor=True)
        
        # Compute the cosine similarity between averaged embeddings
        return util.pytorch_cos_sim(emb_2013.mean(dim=0), emb_2023.mean(dim=0)).item()
    else:
        return None  # Return None if no sentences contain the keyword

# Compute semantic shifts
results = {}
for keyword in KEYWORDS:
    shift = compute_semantic_shift(keyword)
    results[keyword] = shift
    print(f"Semantic similarity for '{keyword}': {shift}")

# Save results to a text file
results_file = os.path.join(OUTPUT_DIR, 'semantic_shifts.txt')
with open(results_file, 'w') as f:
    for keyword, shift in results.items():
        f.write(f"{keyword}: {shift}\n")

print(f"\nSemantic shifts saved to: {results_file}")

# Plot keyword frequency comparison
plot_file = os.path.join(OUTPUT_DIR, 'keyword_frequency_comparison.png')
plot_keyword_frequency(KEYWORDS, tokens_2013, tokens_2023, plot_file)
print(f"\nKeyword frequency comparison plot saved to: {plot_file}")