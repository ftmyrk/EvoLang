# EvoLang: Exploring the Evolution of Language Through News Articles

This project explores how language evolves over time by analyzing datasets of news articles from 2013 and 2023. By examining word usage, sentiment, semantic similarity, keyword associations, and topic modeling, this repository provides insights into linguistic and contextual shifts in news content over a decade.

---

## **Features**

### 1. **Data Collection and Preprocessing**
- Collected news articles from 2013 and 2023.
- Preprocessed the text by cleaning, tokenizing, and preparing it for analysis.

### 2. **Text Generation**
- Used the `LLaMA` language model to generate expanded text from the datasets.
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


---

## **Repository Structure**

```plaintext
EvoLang/
├── Dataset/                     # Raw datasets (2013 and 2023 articles)
├── generated_data/              # Generated text and intermediate data
├── outputs/                     # Output plots and analysis results
├── utils/                       # Utility scripts and macros
│   ├── text_analysis_utils.py   # Functions for sentiment and word frequency analysis
│   ├── text_generation_utils.py # Utilities for LLaMA text generation
│   ├── text_preprocessing_utils.py # Functions for data cleaning and preprocessing
│   ├── macros.py                # Centralized macros for data and model initialization
├── analyze_data.py              # Script for word frequency and sentiment analysis
├── generate_data.py             # Script for generating text using LLaMA
├── kl_divergence.py             # KL Divergence analysis
├── semantic_shift.py            # Semantic similarity analysis
├── visualize_word_associations.py # Word association visualizations
├── apply_topic_modeling.py      # Topic modeling with LDA
├── README.md                    # Project documentation
```
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

- **Script**: generate_data.py
- **Purpose**: Preprocesses the datasets and uses LLaMA to generate text data for analysis.
  ```bash
  python generate_data.py
  ```
- **Output**:
    - generated_data/generated_responses_2013.csv
    - generated_data/generated_responses_2023.csv

### **2. Analyze Data**

- **Script**: analyze_data.py
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

- **Script**: semantic_analysis.py
- **Purpose**: Measures cosine similarity between word embeddings and compares keyword evolution using Word2Vec.
  ```bash
   python semantic_analysis.py
  ```
- **Output**:
    - outputs/cosine_similarities.csv
    - outputs/cosine_similarity_heatmap.png
    - outputs/keyword_similarities.txt

### **4. Apply Topic Modeling**
   - **Script**: apply_topic_modeling.py
   - **Purpose**: Extracts topics related to specific keywords using Latent Dirichlet Allocation (LDA).
   ```bash
   python apply_topic_modeling.py
   ```
   - **Output**:
       - Topics printed in the console for 2013 and 2023 datasets.

### **5. Extract Contexts**
   - **Script**: extract_context.py
   - **Purpose**: Extracts and saves contextual sentences containing specific keywords from the datasets.
   ```bash
   python extract_context.py
   ```
   - **Output**:
       - Contexts printed in the console for 2013 and 2023 datasets.

### **6. Visualize Word Associations**
   - **Script**: visualize_word_associations.py
   - **Purpose**: Trains Word2Vec models and visualizes word associations for each keyword in 2013 and 2023.
  ```bash
  python visualize_word_associations.py
  ```
   - **Output**:
       - outputs/word_associated_graph/word_associations_2013_<keyword>.png
       - outputs/word_associated_graph/word_associations_2023_<keyword>.png

### **7. Semantic Shift Analysis**
   - **Script**: semantic_shift.py
   - **Purpose**: Computes semantic similarity shifts for keywords between 2013 and 2023 using Sentence-BERT.
   ```bash
   python semantic_shift.py
   ```
   - **Output**:
       - outputs/semantic_shifts.txt
       - outputs/keyword_frequency_comparison.png

### **8. KL Divergence Analysis**

- **Script**: `kl_divergence.py`
- **Purpose**: 
  - Quantifies the differences between the word distributions of 2013 and 2023 datasets.
  - Identifies the top words contributing to the divergence.
  ```bash
  python kl_divergence.py
  ```
- Output:
	-	outputs/kl_divergence.txt: Contains the KL divergence value.
	-	outputs/kl_divergence_contributions.png: A bar chart showing the top contributors to the KL divergence.
---
## **Visual Output** 
 - Question response output comparisation between 2013 and 2023
<div style="display: flex; justify-content: space-between;">
	<img src = "https://github.com/ftmyrk/EvoLang/blob/main/outputs/response_similarity_comparison.png" style = width: 20%;"/>
</div>

- Most used words
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/ftmyrk/EvoLang/blob/main/outputs/wordcloud_2013.png" alt="alt-text-1" style="width: 40%;"/>
    <img src="https://github.com/ftmyrk/EvoLang/blob/main/outputs/wordcloud_2023.png" alt="alt-text-2" style="width: 40%;"/>
</div>
