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
│   ├── measurement_utils.py     # Functions for semantic similarity and KL Divergence
│   ├── macros.py                # Centralized macros for data and model initialization
├── analyze_data.py              # Script for word frequency and sentiment analysis
├── generate_data.py             # Script for generating text using LLaMA
├── kl_divergence.py             # KL Divergence analysis
├── semantic_shift.py            # Semantic similarity analysis
├── visualize_word_associations.py # Word association visualizations
├── apply_topic_modeling.py      # Topic modeling with LDA
├── README.md                    # Project documentation
