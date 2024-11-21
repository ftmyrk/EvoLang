from utils.macros import sentence_bert_model, data_2013, data_2023, tokens_2013, tokens_2023, OUTPUT_DIR
from sentence_transformers import util
from utils.text_analysis_utils import plot_keyword_frequency

# Function to compute semantic similarity for a keyword
def compute_semantic_shift(keyword):
    sentences_2013 = [text for text in data_2013['Original_Text'] if keyword in text]
    sentences_2023 = [text for text in data_2023['Original_Text'] if keyword in text]
    if sentences_2013 and sentences_2023:
        emb_2013 = sentence_bert_model.encode(sentences_2013, convert_to_tensor=True)
        emb_2023 = sentence_bert_model.encode(sentences_2023, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb_2013.mean(dim=0), emb_2023.mean(dim=0)).item()
    return None

# Keywords to analyze
keywords = ['economy', 'policy', 'shares', 'technology', 'market']

# Compute semantic shifts
results = {}
for keyword in keywords:
    shift = compute_semantic_shift(keyword)
    results[keyword] = shift
    print(f"Semantic similarity for '{keyword}': {shift}")

# Save results
with open(f'{OUTPUT_DIR}/semantic_shifts.txt', 'w') as f:
    for keyword, shift in results.items():
        f.write(f"{keyword}: {shift}\n")
        
        
plot_keyword_frequency(keywords, tokens_2013, tokens_2023, f'{OUTPUT_DIR}/keyword_frequency_comparison.png')