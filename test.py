from utils import measurement_utils
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

old_data = pd.read_csv('/home/otamy001/EvoLang/generated_data/generated_responses_2013.csv')
new_data = pd.read_csv('/home/otamy001/EvoLang/generated_data/generated_responses_2023.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')
old_embeddings = model.encode(old_data['Original_Text'].tolist(), convert_to_tensor=True)
new_embeddings = model.encode(new_data['Original_Text'].tolist(), convert_to_tensor=True)

measurement_utils.calculate_cosine_similarity(old_embeddings, new_embeddings, '/home/otamy001/EvoLang/generated_data/cosine_similarities.csv')

similarity_df = pd.read_csv('/home/otamy001/EvoLang/generated_data/cosine_similarities.csv')
measurement_utils.plot_similarity_heatmap(similarity_df, '/home/otamy001/EvoLang/outputs/' ,'cosine_similarity_heatmap.png')

# Keyword Evolution Analysis with Word2Vec
old_data_tokens = [text.split() for text in old_data['Original_Text'].tolist()]
new_data_tokens = [text.split() for text in new_data['Original_Text'].tolist()]

model_2013 = Word2Vec(sentences=old_data_tokens, vector_size=100, window=5, min_count=2, workers=4)
model_2023 = Word2Vec(sentences=new_data_tokens, vector_size=100, window=5, min_count=2, workers=4)

keywords = ['economy', 'policy', 'shares', 'technology', 'market']
keyword_similarities = measurement_utils.compare_keyword_vectors(model_2013, model_2023, keywords)

for keyword, similarity in keyword_similarities.items():
    if similarity is not None:
        print(f"Cosine similarity for '{keyword}' between 2013 and 2023: {similarity:.4f}")
    else:
        print(f"'{keyword}' not found in one of the models.")
