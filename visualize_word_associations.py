import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import os
import pandas as pd

# Load data
old_data = pd.read_csv('/home/otamy001/EvoLang/Dataset/generated_responses_2013.csv')
new_data = pd.read_csv('/home/otamy001/EvoLang/Dataset/generated_responses_2023.csv')

# Tokenize the articles for each year
old_data_tokens = [text.split() for text in old_data['Original_Text'].tolist()]
new_data_tokens = [text.split() for text in new_data['Original_Text'].tolist()]

# Train and save Word2Vec models if they don't already exist
if not os.path.exists('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2013.model'):
    model_2013 = Word2Vec(sentences=old_data_tokens, vector_size=100, window=5, min_count=2, workers=4)
    model_2013.save('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2013.model')
else:
    model_2013 = Word2Vec.load('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2013.model')

if not os.path.exists('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2023.model'):
    model_2023 = Word2Vec(sentences=new_data_tokens, vector_size=100, window=5, min_count=2, workers=4)
    model_2023.save('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2023.model')
else:
    model_2023 = Word2Vec.load('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2023.model')

# Function to plot word associations
def plot_word_associations(model, keyword, top_n=10, output_file=None):
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
        else:
            plt.show()
    else:
        print(f"'{keyword}' not found in the model.")

model_2013 = Word2Vec.load('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2013.model')
model_2023 = Word2Vec.load('/home/otamy001/EvoLang/word2vec_models/word2vec_model_2023.model')

# List of keywords to analyze
keywords = ['economy', 'policy', 'shares', 'technology', 'market']

for keyword in keywords:
    print(f"\nWord associations for '{keyword}' in 2013:")
    plot_word_associations(model_2013, keyword, output_file=f'/home/otamy001/EvoLang/word_associatioted_graph/word_associations_2013_{keyword}.png')
    
    print(f"\nWord associations for '{keyword}' in 2023:")
    plot_word_associations(model_2023, keyword, output_file=f'/home/otamy001/EvoLang/word_associatioted_graph/word_associations_2023_{keyword}.png')