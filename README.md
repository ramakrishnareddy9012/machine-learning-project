# Real-Time Sentiment Analysis System with Web Scraping

**Department of Computer Science and Engineering | Aditya University**  
*Machine Learning & NLP Project Review Implementation*

---

## 👥 Team Members & Roll Numbers
- **I. Harshil Preetham** – 24B11CS156 *(Project Lead)*
- **D. V. S. Samrudh** – 24B11CS114
- **M. C. S. Vikas Reddy** – 24B11CS274
- **J. Lakshman Swamy** – 24B11CS163
- **K. Rama Krishna Reddy** – 24B11CS233

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
   - **Step 5:** Stop-word filtering
   - **Step 6:** Suffix normalization & Stemming

3. **Machine Learning Classifiers (`app/model.py`)**
   - **Multinomial Naive Bayes**
   - **Logistic Regression (TF-IDF)**
   - **Linear Support Vector Machine (LinearSVC)**
   - **VADER Rule-Based Sentiment Analyzer**
   - **Hybrid Ensemble Classifier** (Weighted ML + VADER sentiment fusion)

4. **FastAPI REST API & Interactive Dashboard (`app/api.py`, `app/templates/dashboard.html`)**
   - Single Text Real-Time Sentiment Tester with confidence gauge & NLP trace.
   - Live URL Web Content Scraper & Paragraph Analyzer.
   - Real-Time Opinion Stream Feed Simulator.
   - Model Evaluation Comparison Matrix (Accuracy, Precision, Recall, F1-Score).
   - Embedded Project Review Presentation Viewer (Slides 1 to 10).

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

---

## 🎓 University Review Details
- **Department:** Computer Science & Engineering
- **Institution:** Aditya University
- **Project Topic:** Real-Time Sentiment Analysis System with Web Scraping
