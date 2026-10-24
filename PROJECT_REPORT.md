# Real-Time Sentiment Analysis System with Web Scraping

**Academic Project Report**  
**Department of Computer Science and Engineering**  
**Aditya University**

---

## 📄 Abstract
In the modern digital era, online consumer feedback, e-commerce product reviews, and social media platforms generate millions of textual opinions daily. Manually parsing, categorizing, and interpreting these unstructured textual datasets is computationally expensive and practically unfeasible. 

This project presents an end-to-end **Real-Time Sentiment Analysis System with Web Scraping** powered by Natural Language Processing (NLP) and Machine Learning (ML). The system features an automated Web Scraping engine (`BeautifulSoup4`), a 6-stage NLP cleaning and preprocessing pipeline, Sublinear TF-IDF trigram feature vectorization ($1 + \log(\text{tf})$), and a **Hybrid Soft-Voting Ensemble Classifier** trained on **47,276 Amazon Consumer Reviews** extracted from a 67,992 entry dataset archive. 

Experimental benchmarks demonstrate that our Hybrid Soft-Voting Ensemble Classifier achieves a state-of-the-art **94.12% Accuracy** and **93.96% F1-Score**, significantly outperforming traditional rule-based approaches (VADER: 90.21%) and standalone Naive Bayes models (89.30%).

---

## 🎯 1. Introduction & Problem Statement

### 1.1 Problem Statement
Online platforms generate massive volumes of user opinions every day. Organizations and research teams face the following challenges:
1. **High Data Volume & Velocity**: Web opinion content expands continuously in real time.
2. **Manual Processing Bottlenecks**: Manually reading customer feedback is slow, prone to human bias, and non-scalable.
3. **Unstructured HTML Noise**: Web pages contain scripts, styling, headers, footers, and advertisement noise that degrade text classifier accuracy.
4. **Class Imbalance & Multi-Gram Nuances**: Sentiment text contains subtle negation phrases (*"not worth buying"*, *"never fails to impress"*) that simple bag-of-words models fail to capture.

### 1.2 Key Objectives
- **Web Data Extraction**: Automatically fetch web pages and strip HTML noise using BeautifulSoup.
- **NLP Text Preprocessing**: Implement HTML stripping, lowercasing, punctuation filtering, tokenization, stop-words removal, and suffix stemming.
- **Feature Engineering**: Extract 35,000 sublinear TF-IDF unigrams, bigrams, and trigrams $(1, 3)$ concatenated with review titles and VADER polarity features.
- **Machine Learning Classification**: Train and evaluate Logistic Regression, Linear Support Vector Machine (LinearSVC), Complement Naive Bayes, VADER Analyzer, and a Soft-Voting Ensemble Model.
- **Real-Time Dashboard & API**: Serve a web dashboard (`FastAPI` + HTML5/CSS3/JS) for real-time inference, live stream simulation, and web scraping.

---

## 🏗️ 2. Proposed System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 WEB SOURCES & REVIEWS                             |
|              (Amazon Consumer Reviews Dataset: 67,992 Raw / 47,276 Clean)          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                WEB SCRAPING ENGINE                                |
|             (Requests + BeautifulSoup4 HTML Noise & Script Stripper)             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             6-STAGE NLP PREPROCESSING                             |
|    HTML Strip -> Lowercase -> Noise Filter -> Tokenize -> Stopwords -> Stemming   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        SUBLINEAR TF-IDF FEATURE VECTORIZER                        |
|       (35,000 Trigrams (1, 3) + Title-Text Concatenation + VADER Polarity)        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         HYBRID SOFT-VOTING ENSEMBLE CLASSIFIER                     |
|         (Soft Fusion: Logistic Regression + LinearSVC + ComplementNB + VADER)     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        REAL-TIME DASHBOARD & FASTAPI REST API                     |
|           (Text Tester, URL Intelligence, Live Stream, Benchmarks Dashboard)      |
+-----------------------------------------------------------------------------------+
```

---

## 🔬 3. Methodology & Implementation Details

### 3.1 Dataset Preparation
The model was trained on **47,276 unique clean consumer review records** combined across three CSV archives inside `archive (2).zip`:
1. `1429_1.csv` (34,660 records)
2. `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` (28,332 records)
3. `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` (5,000 records)

**Star Rating to Sentiment Class Mapping**:
- Ratings **1 & 2** $\rightarrow$ **`negative`**
- Rating **3** $\rightarrow$ **`neutral`**
- Ratings **4 & 5** $\rightarrow$ **`positive`**

### 3.2 Feature Vectorization
Text features are extracted using **Sublinear TF-IDF** with logarithmic term frequency weighting:
$$\text{tf}_{\text{sublinear}} = 1 + \log(\text{tf})$$
$$\text{TF-IDF}(t, d) = \text{tf}_{\text{sublinear}}(t, d) \times \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$

---

## 📊 4. Experimental Results & Performance Analysis

### 4.1 Comparative Model Benchmark Table (47,276 Records)

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|---|---|---|---|---|---|
| **Hybrid Soft-Voting Ensemble** | **94.12%** | **93.85%** | **94.12%** | **93.96%** | **< 10 ms** |
| **Linear Support Vector Machine (LinearSVC)** | **94.00%** | **93.38%** | **94.00%** | **93.68%** | **< 6 ms** |
| **Logistic Regression (TF-IDF)** | **92.73%** | **93.71%** | **92.73%** | **93.21%** | **< 8 ms** |
| **VADER Rule-Based Analyzer** | **90.21%** | **90.10%** | **90.21%** | **90.09%** | **< 12 ms** |
| **Complement Naive Bayes** | **89.30%** | **92.15%** | **89.30%** | **90.70%** | **< 4 ms** |

---

## 🎓 5. Conclusion & Future Work
The implemented system successfully demonstrates real-time web scraping, multi-stage NLP text preprocessing, and high-accuracy sentiment classification. Achieving **94.12% Accuracy** on a 47,276 record dataset proves the effectiveness of hybrid ensemble learning for real-world opinion mining.

**Future Enhancements**:
1. Integration of Deep Learning Transformer architectures (BERT / RoBERTa).
2. Expansion to multilingual sentiment classification across regional languages.
