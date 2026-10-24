# Real-Time Sentiment Analysis System with Web Scraping

An end-to-end Machine Learning & NLP system for real-time web scraping, text preprocessing, and sentiment classification.

---

## 📌 Project Overview
This project presents an end-to-end **Real-Time Sentiment Analysis System with Web Scraping** built using Python, Scikit-Learn, NLTK/VADER, BeautifulSoup, and FastAPI.

The system automatically extracts online textual content (user reviews, news paragraphs, social comments), cleans raw HTML noise, applies NLP preprocessing, and classifies sentiment into **Positive**, **Neutral**, or **Negative** categories with confidence scores and interactive visualizations.

---

## 🏗️ System Architecture

```
+------------------+       +--------------------+       +------------------------------+
|   Web Sources    | ----> |    Web Scraper     | ----> |    Data Cleaning & NLP       |
|  (URLs/Reviews)  |       |  (BeautifulSoup4)  |       |  (HTML noise, Lowercasing,  |
+------------------+       +--------------------+       |  Tokenization, Stop-words)   |
                                                        +------------------------------+
                                                                       |
+------------------+       +--------------------+                      v
|  Dashboard UI &  | <---- |   Sentiment ML     | <---- +------------------------------+
|   REST API       |       |   Classifier       |       |     Feature Representation   |
| (FastAPI Server) |       | (LR, NB, SVC, VADER)|      |      (TF-IDF N-Gram Vector)  |
+------------------+       +--------------------+       +------------------------------+
```

---

## 🚀 Key Features & Modules

1. **Web Scraping Engine (`app/scraper.py`)**
   - Automatically fetches web pages and extracts paragraph text blocks.
   - Cleans HTML noise (scripts, styles, headers, footers, navigation tags).
   - Provides live request handling with fallback simulated streams.

2. **NLP Preprocessing Engine (`app/preprocessing.py`)**
   - **Step 1:** HTML & URL removal
   - **Step 2:** Lowercasing
   - **Step 3:** Punctuation and special character removal
   - **Step 4:** Tokenization
   - **Step 5:** Stop-word filtering (with negation preservation)
   - **Step 6:** Suffix normalization & Stemming

3. **Machine Learning Classifiers (`app/model.py`)**
   - **Multinomial / Complement Naive Bayes**
   - **Logistic Regression (TF-IDF)**
   - **Linear Support Vector Machine (LinearSVC)**
   - **VADER Rule-Based Sentiment Analyzer**
   - **Hybrid Ensemble Classifier** (Weighted ML + VADER sentiment fusion)

4. **FastAPI REST API & Interactive Dashboard (`app/api.py`, `app/templates/dashboard.html`)**
   - Single Text Real-Time Sentiment Tester with confidence gauge & NLP trace.
   - Live URL Web Content Scraper & Paragraph Analyzer.
   - Real-Time Opinion Stream Feed Simulator.
   - Model Evaluation Comparison Matrix (Accuracy, Precision, Recall, F1-Score).
   - Custom Dataset Upload and on-the-fly sentiment annotation.

---

## 🔧 Installation & Setup

### 1. Prerequisites
- Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Server & Launch Dashboard
```bash
python main.py
```
Open your browser and navigate to:
- **Interactive Dashboard:** `http://127.0.0.1:8000/`
- **FastAPI OpenAPI Swagger Documentation:** `http://127.0.0.1:8000/docs`

---

## 🧪 Running Automated Tests

```bash
python -m unittest discover tests
```

---

## 📡 API Endpoint Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves Interactive Dashboard UI |
| `POST` | `/api/analyze/text` | Classifies single raw text string & returns NLP trace |
| `POST` | `/api/analyze/batch` | Analyzes list of text items with aggregate summary statistics |
| `POST` | `/api/analyze/url` | Scrapes target URL, extracts text, and returns sentiment metrics |
| `GET` | `/api/preprocessing/inspect` | Returns step-by-step trace of NLP cleaning steps |
| `GET` | `/api/model/metrics` | Returns evaluation metrics (Accuracy, Precision, Recall, F1) |
| `POST` | `/api/dataset/upload` | Uploads and processes custom CSV/JSON dataset |
| `GET` | `/api/dataset/list` | Lists all available datasets |
| `POST` | `/api/dataset/evaluate` | Evaluates sentiment across selected dataset |
