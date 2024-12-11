# EvoLang: Exploring the Evolution of Language Through News Articles

This project explores how language evolves over time by analyzing datasets of news articles from 2013 and 2023. By examining word usage, sentiment, semantic similarity, keyword associations, and topic modeling, this repository provides insights into linguistic and contextual shifts in news content over a decade.

---

## **Features**

### 1. **Data Collection and Preprocessing**
- Collected news articles from 2013 and 2023.
- Preprocessed the text by cleaning, tokenizing, and preparing it for analysis.

### 2. **Text Generation with LLaMA**
- Utilized the LLaMA language model to generate responses based on predefined questions and context from the datasets.
- Stored generated responses for both years for further analysis.

### 3. **Sentiment Analysis**
- Analyzed the sentiment (positive vs. negative) of articles using a pre-trained `DistilBERT` model.
- Visualized sentiment distributions for 2013 and 2023.

### 4. **Word Frequency and Keyword Comparison**
- Generated word frequency charts and word clouds for each year.
- Compared keyword frequencies to identify trends in the focus of topics.

### 5. **Semantic Similarity**
- Measured semantic shifts in keywords (e.g., "economy," "policy") using `Sentence-BERT`.
- Quantified contextual changes in language over the years.

### 6. **KL Divergence Analysis**
- Calculated KL Divergence to measure shifts in word distributions between 2013 and 2023.
- Identified top words contributing to divergence and visualized their impact.

### 7. **Topic Modeling**
- Applied Latent Dirichlet Allocation (LDA) to extract key topics in 2013 and 2023 datasets.
- Compared topics to analyze thematic evolution.

### 8. **Word Association Visualizations**
- Trained Word2Vec models to explore associations for specific keywords.
- Created bar charts highlighting the most similar words for each keyword in 2013 and 2023.

### 9. **Temporal Question-Answering with LLaMA**
- Designed a set of contextually relevant questions reflecting societal and knowledge-based themes (e.g., “The president of the United States is…”).
- Highlighted temporal influences on the model’s generated answers

### 10. **Analysis of Temporal Divergences**
- Compared LLaMA-generated responses across years using text similarity metrics.
- Identified questions with the most significant temporal divergence in responses.
- Visualized similarities and differences in generated content to reveal context-based variations.

---
## **Dataset**

### **Sources**
1. **2013 Dataset**: [Link to Download](https://drive.google.com/file/d/10nzlFF83IGoLDVlFILwtVBVW9TPaeL1m/view?usp=drive_link)
2. **2023 Dataset**: [Link to Download](https://drive.google.com/file/d/10sLum2gntV-notnNUVsOMOIIqvNvtNrj/view?usp=drive_link)

---

### **Automated Download**

The datasets are automatically downloaded when you run any script that requires them. The files are saved in the `Dataset/` directory.

---
## **Execution Order and Purposes**

### **1. Generate Data**

- **Purpose**: Preprocesses the datasets and uses LLaMA to generate text data for analysis.
  ```bash
  python generate_data.py
  ```
- **Output**:
    - generated_data/generated_responses_2013.csv
    - generated_data/generated_responses_2023.csv

### **2. Analyze Data**

- **Purpose**: Performs word frequency analysis, generates word clouds, and conducts sentiment analysis.
  ```bash
  python analyze_data.py
  ```
- **Output**:
    - outputs/word_freq_2013.png
    - outputs/word_freq_2023.png
    - outputs/wordcloud_2013.png
    - outputs/wordcloud_2023.png
    - outputs/visualize_sentiment_2013.png
    - outputs/visualize_sentiment_2023.png

### **3. Measure Semantic Shifts**

- **Purpose**: Measures cosine similarity between word embeddings and compares keyword evolution using Word2Vec.
  ```bash
   python semantic_analysis.py
  ```
- **Output**:
    - outputs/cosine_similarities.csv
    - outputs/cosine_similarity_heatmap.png
    - outputs/keyword_similarities.txt

### **4. Apply Topic Modeling**
   - **Purpose**: Extracts topics related to specific keywords using Latent Dirichlet Allocation (LDA).
   ```bash
   python apply_topic_modeling.py
   ```
   - **Output**:
       - Topics printed in the console for 2013 and 2023 datasets.

### **5. Extract Contexts**
   - **Purpose**: Extracts and saves contextual sentences containing specific keywords from the datasets.
   ```bash
   python extract_context.py
   ```
   - **Output**:
       - Contexts printed in the console for 2013 and 2023 datasets.

### **6. Visualize Word Associations**
   - **Purpose**: Trains Word2Vec models and visualizes word associations for each keyword in 2013 and 2023.
  ```bash
  python visualize_word_associations.py
  ```
   - **Output**:
       - outputs/word_associated_graph/word_associations_2013_<keyword>.png
       - outputs/word_associated_graph/word_associations_2023_<keyword>.png

### **7. Semantic Shift Analysis**
   - **Purpose**: Computes semantic similarity shifts for keywords between 2013 and 2023 using Sentence-BERT.
   ```bash
   python semantic_shift.py
   ```
   - **Output**:
       - outputs/semantic_shifts.txt
       - outputs/keyword_frequency_comparison.png

### **8. KL Divergence Analysis**

- **Purpose**: 
  - Quantifies the differences between the word distributions of 2013 and 2023 datasets.
  - Identifies the top words contributing to the divergence.
  ```bash
  python kl_divergence.py
  ```
- Output:
	-	outputs/kl_divergence.txt: Contains the KL divergence value.
	-	outputs/kl_divergence_contributions.png: A bar chart showing the top contributors to the KL divergence.
### **9. LLaMA Experiment**

- **Purpose**: Uses the LLaMA language model to generate responses to a predefined set of questions. The context for these responses is extracted from news articles from 2013 and 2023, allowing for analysis of how the model interprets knowledge and language in different time frames.
  ```bash
  python llama_experiment.py
  ```
- Output:

### **10. Analyze LLaMA Experiment Results**

- **Purpose**: Compares responses generated by the LLaMA Experiment for 2013 and 2023 contexts. It calculates similarities between responses using a text similarity metric, highlights variations in generated answers, and visualizes differences.
  ```bash
  python analyze_llama_results.py
  ```
- Output:
  	- outputs/response_differences.txt: Detailed analysis of response similarities.
	- outputs/response_similarity_comparison.png
---
## **Visual Output** 
 - Question response output comparisation between 2013 and 2023
<div style="display: flex; justify-content: space-between;">
	<img src = "https://github.com/ftmyrk/EvoLang/blob/main/outputs/response_similarity_comparison.png" style = width: 20%;"/>
</div>

- Most used words
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/ftmyrk/EvoLang/blob/main/outputs/wordcloud_2013.png" alt="alt-text-1" style="width: 48%;"/>
    <img src="https://github.com/ftmyrk/EvoLang/blob/main/outputs/wordcloud_2023.png" alt="alt-text-2" style="width: 48%;"/>
</div>
