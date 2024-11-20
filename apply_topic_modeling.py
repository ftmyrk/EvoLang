from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Function to apply LDA and display topics
def apply_topic_modeling(data, keyword, num_topics=3, num_words=5):
    keyword_texts = [text for text in data['Original_Text'].tolist() if keyword in text]
    if keyword_texts:
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(keyword_texts)
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix)
        
        # Use get_feature_names() for older versions of scikit-learn
        words = vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(lda_model.components_):
            print(f"Topic {topic_idx + 1}:")
            print(" ".join([words[i] for i in topic.argsort()[-num_words:]]))
    else:
        print(f"No articles found for '{keyword}'.")

old_data = pd.read_csv('/home/otamy001/EvoLang/Dataset/generated_responses_2013.csv')
new_data = pd.read_csv('/home/otamy001/EvoLang/Dataset/generated_responses_2023.csv')

# List of keywords to analyze
keywords = ['economy', 'policy', 'shares', 'technology', 'market']

# Apply topic modeling for each keyword in both datasets
for keyword in keywords:
    print(f"\nTopics for '{keyword}' in 2013:")
    apply_topic_modeling(old_data, keyword)

    print(f"\nTopics for '{keyword}' in 2023:")
    apply_topic_modeling(new_data, keyword)