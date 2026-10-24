import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Any, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.preprocessing import TextPreprocessor
from app.dataset import get_train_test_data

class SentimentModelPipeline:
    """
    Enhanced ML Sentiment Analysis System Pipeline:
    Optimized TF-IDF (1-3 ngrams, sublinear TF) + Calibrated LinearSVC,
    Logistic Regression, Complement Naive Bayes, and Hybrid Ensemble Classifier.
    """
    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=35000,
            sublinear_tf=True
        )
        self.vader = SentimentIntensityAnalyzer()

        self.lr_model = LogisticRegression(C=5.0, max_iter=500, class_weight="balanced")
        self.svc_raw = LinearSVC(C=1.2, class_weight="balanced", dual="auto")
        self.cnb_model = ComplementNB(alpha=0.05)
        self.calibrated_svc = CalibratedClassifierCV(self.svc_raw)

        self.ensemble_model = VotingClassifier(
            estimators=[
                ("lr", self.lr_model),
                ("svc", self.calibrated_svc),
                ("cnb", self.cnb_model)
            ],
            voting="soft"
        )

        self.is_trained = False
        self.evaluation_metrics = {}

    def _get_vader_features(self, texts: List[str]) -> np.ndarray:
        """Extract scaled VADER polarity scores in range [0, 1]."""
        vader_scores = []
        for text in texts:
            comp = self.vader.polarity_scores(text)["compound"]
            vader_scores.append((comp + 1.0) / 2.0)
        return np.array(vader_scores).reshape(-1, 1)

    def train_and_evaluate(self) -> Dict[str, Any]:
        """Train models on dataset and compute accuracy, precision, recall, f1, and confusion matrix."""
        train_df, test_df = get_train_test_data()

        # Clean text
        X_train_clean = [self.preprocessor.preprocess(t) for t in train_df["text"]]
        y_train = train_df["sentiment"].values

        X_test_clean = [self.preprocessor.preprocess(t) for t in test_df["text"]]
        y_test = test_df["sentiment"].values

        # TF-IDF Feature Extraction
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        X_test_vec = self.vectorizer.transform(X_test_clean)

        # VADER feature column
        v_train_col = sp.csr_matrix(self._get_vader_features(list(train_df["text"])))
        v_test_col = sp.csr_matrix(self._get_vader_features(list(test_df["text"])))

        X_train_stacked = sp.hstack([X_train_vec, v_train_col])
        X_test_stacked = sp.hstack([X_test_vec, v_test_col])

        # Train individual models on stacked features
        self.lr_model.fit(X_train_stacked, y_train)
        self.svc_raw.fit(X_train_vec, y_train)
        self.cnb_model.fit(X_train_stacked, y_train)

        # Train Ensemble
        self.ensemble_model.fit(X_train_stacked, y_train)
        self.is_trained = True

        # Evaluate models
        models = {
            "Hybrid Ensemble Classifier": (self.ensemble_model, X_test_stacked),
            "Linear Support Vector Machine (LinearSVC)": (self.svc_raw, X_test_vec),
            "Logistic Regression (TF-IDF)": (self.lr_model, X_test_stacked),
            "Complement Naive Bayes": (self.cnb_model, X_test_stacked)
        }

        labels = ["positive", "neutral", "negative"]
        metrics_summary = {}

        for name, (model, test_features) in models.items():
            y_pred = model.predict(test_features)
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=labels, average="weighted", zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

            metrics_summary[name] = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "labels": labels,
                "confusion_matrix": cm
            }

        # Evaluate VADER as rule-based benchmark
        vader_preds = []
        for text in test_df["text"]:
            v_score = self.vader.polarity_scores(text)["compound"]
            if v_score >= 0.05:
                vader_preds.append("positive")
            elif v_score <= -0.05:
                vader_preds.append("negative")
            else:
                vader_preds.append("neutral")

        v_acc = accuracy_score(y_test, vader_preds)
        v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(
            y_test, vader_preds, labels=labels, average="weighted", zero_division=0
        )
        metrics_summary["VADER Rule-Based Analyzer"] = {
            "accuracy": round(float(v_acc), 4),
            "precision": round(float(v_prec), 4),
            "recall": round(float(v_rec), 4),
            "f1_score": round(float(v_f1), 4),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_test, vader_preds, labels=labels).tolist()
        }

        self.evaluation_metrics = metrics_summary
        return metrics_summary

    def predict(self, raw_text: str, selected_model: str = "Ensemble") -> Dict[str, Any]:
        """
        Classify input raw text and return sentiment prediction,
        confidence scores, emotion highlights, and processing details.
        """
        if not self.is_trained:
            self.train_and_evaluate()

        clean_text = self.preprocessor.preprocess(raw_text)
        if not clean_text:
            return {
                "raw_text": raw_text,
                "clean_text": "",
                "sentiment": "neutral",
                "confidence": 0.5,
                "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                "vader_compound": 0.0,
                "model_used": selected_model
            }

        text_vec = self.vectorizer.transform([clean_text])
        v_col = sp.csr_matrix(self._get_vader_features([raw_text]))
        text_stacked = sp.hstack([text_vec, v_col])

        vader_scores = self.vader.polarity_scores(raw_text)
        v_compound = vader_scores["compound"]

        # Get Ensemble probabilities
        ens_probs = self.ensemble_model.predict_proba(text_stacked)[0]
        ens_classes = list(self.ensemble_model.classes_)
        scores = {cls: round(float(prob), 4) for cls, prob in zip(ens_classes, ens_probs)}

        if selected_model in ["Logistic Regression", "Logistic Regression (TF-IDF)"]:
            lr_probs = self.lr_model.predict_proba(text_stacked)[0]
            lr_classes = list(self.lr_model.classes_)
            best_sentiment = self.lr_model.predict(text_stacked)[0]
            confidence = round(float(max(lr_probs)), 4)
            scores = {cls: round(float(prob), 4) for cls, prob in zip(lr_classes, lr_probs)}
        elif selected_model in ["Linear SVC", "Linear Support Vector Machine (LinearSVC)"]:
            best_sentiment = self.svc_raw.predict(text_vec)[0]
            confidence = round(scores.get(best_sentiment, 0.82), 4)
        elif selected_model in ["Complement Naive Bayes", "Naive Bayes"]:
            cnb_probs = self.cnb_model.predict_proba(text_stacked)[0]
            cnb_classes = list(self.cnb_model.classes_)
            best_sentiment = self.cnb_model.predict(text_stacked)[0]
            confidence = round(float(max(cnb_probs)), 4)
            scores = {cls: round(float(prob), 4) for cls, prob in zip(cnb_classes, cnb_probs)}
        elif selected_model == "VADER":
            if v_compound >= 0.05:
                best_sentiment = "positive"
            elif v_compound <= -0.05:
                best_sentiment = "negative"
            else:
                best_sentiment = "neutral"
            confidence = round(abs(v_compound) if abs(v_compound) > 0.5 else 0.65, 4)
        else: # Ensemble
            best_sentiment = max(scores, key=scores.get)
            confidence = scores[best_sentiment]

        return {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "sentiment": best_sentiment,
            "confidence": confidence,
            "scores": scores,
            "vader_compound": round(v_compound, 4),
            "model_used": selected_model
        }

    def get_top_features(self, top_n: int = 12) -> Dict[str, List[Dict[str, Any]]]:
        """Return top TF-IDF trigram features with highest weights for each sentiment class."""
        if not self.is_trained:
            self.train_and_evaluate()
        
        feature_names = list(self.vectorizer.get_feature_names_out())
        feature_names.append("vader_polarity_score")
        classes = list(self.lr_model.classes_)
        
        res = {}
        if hasattr(self.lr_model, "coef_"):
            for i, cls in enumerate(classes):
                coefs = self.lr_model.coef_[i]
                top_idx = np.argsort(coefs)[-top_n:][::-1]
                res[cls] = [
                    {"word": feature_names[j], "weight": round(float(coefs[j]), 3)}
                    for j in top_idx if j < len(feature_names)
                ]
        return res

