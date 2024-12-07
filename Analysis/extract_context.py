# extract_context.py

import os
import random
from utils.macros import generated_events, OUTPUT_DIR, KEYWORDS


data_2013, data_2023 = generated_events()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to extract contexts for a given keyword
def extract_contexts(data, keyword, num_samples=5):
    contexts = [text for text in data['Original_Text'].tolist() if keyword in text]
    if len(contexts) > num_samples:
        contexts = random.sample(contexts, num_samples)
    return contexts

output_file = os.path.join(OUTPUT_DIR, 'extracted_contexts.txt')

with open(output_file, 'w') as f:
    for keyword in KEYWORDS:
        f.write(f"\nContexts for '{keyword}' in 2013:\n")
        print(f"\nContexts for '{keyword}' in 2013:")
        contexts_2013 = extract_contexts(data_2013, keyword)
        for i, context in enumerate(contexts_2013, 1):
            snippet = context[:300]  
            print(f"{i}. {snippet}...")
            f.write(f"{i}. {snippet}...\n")
        
        f.write(f"\nContexts for '{keyword}' in 2023:\n")
        print(f"\nContexts for '{keyword}' in 2023:")
        contexts_2023 = extract_contexts(data_2023, keyword)
        for i, context in enumerate(contexts_2023, 1):
            snippet = context[:300]
            print(f"{i}. {snippet}...")
            f.write(f"{i}. {snippet}...\n")

print(f"\nExtracted contexts saved to: {output_file}")