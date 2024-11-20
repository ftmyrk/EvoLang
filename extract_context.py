import pandas as pd
import random

def extract_contexts(data, keyword, num_samples=5):
    contexts = [text for text in data['Original_Text'].tolist() if keyword in text]
    if len(contexts) > num_samples:
        contexts = random.sample(contexts, num_samples)
    return contexts

old_data = pd.read_csv('/home/otamy001/EvoLang/generated_data/generated_responses_2013.csv')
new_data = pd.read_csv('/home/otamy001/EvoLang/generated_data/generated_responses_2023.csv')

keywords = ['economy', 'policy', 'shares', 'technology', 'market']

for keyword in keywords:
    print(f"\nContexts for '{keyword}' in 2013:")
    contexts_2013 = extract_contexts(old_data, keyword)
    for i, context in enumerate(contexts_2013, 1):
        print(f"{i}. {context[:300]}...")  # first 300 characters for brevity

    print(f"\nContexts for '{keyword}' in 2023:")
    contexts_2023 = extract_contexts(new_data, keyword)
    for i, context in enumerate(contexts_2023, 1):
        print(f"{i}. {context[:300]}...")