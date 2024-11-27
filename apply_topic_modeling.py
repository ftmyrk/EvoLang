# apply_topic_modeling.py

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from utils.macros import generated_events, OUTPUT_DIR, KEYWORDS

os.makedirs(OUTPUT_DIR, exist_ok=True)
data_2013, data_2023, _, _ = generated_events()

# Function to apply Latent Dirichlet Allocation and display topics
def apply_topic_modeling(data, keyword, num_topics=3, num_words=5):
    # Perform LDA on texts containing a keyword and display the top topics
    keyword_texts = [text for text in data['Original_Text'].tolist() if keyword in text]
    if keyword_texts:
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(keyword_texts)
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix)
        
        # Gets feature names for topic words
        words = vectorizer.get_feature_names() 
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            topic_words = " ".join([words[i] for i in topic.argsort()[-num_words:]])
            print(f"Topic {topic_idx + 1}: {topic_words}")
            topics.append(topic_words)
        
        return topics
    else:
        print(f"No articles found for '{keyword}'.")
        return []

results = {}

for keyword in KEYWORDS:
    print(f"\nTopics for '{keyword}' in 2013:")
    topics_2013 = apply_topic_modeling(data_2013, keyword)
    results[f'{keyword}_2013'] = topics_2013

    print(f"\nTopics for '{keyword}' in 2023:")
    topics_2023 = apply_topic_modeling(data_2023, keyword)
    results[f'{keyword}_2023'] = topics_2023

output_file = os.path.join(OUTPUT_DIR, 'topic_modeling_results.txt')
with open(output_file, 'w') as f:
    for key, topics in results.items():
        f.write(f"\nTopics for {key}:\n")
        for topic in topics:
            f.write(f"- {topic}\n")

print(f"\nTopic modeling results saved to: {output_file}")