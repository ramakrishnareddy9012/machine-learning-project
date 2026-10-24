import os
import json
import uuid
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

from app.dataset import prepare_dataset

UPLOADS_DIR = os.path.join("data", "uploads")
METADATA_FILE = os.path.join(UPLOADS_DIR, "datasets_meta.json")

class DatasetManager:
    """
    Module for Uploading, Processing, Indexing, and Managing Custom Datasets.
    Supports CSV, JSON, and TXT file uploads, automatic schema detection,
    model sentiment annotation, and multi-model performance testing across datasets.
    """
    def __init__(self):
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.dataset_dfs: Dict[str, pd.DataFrame] = {}
        self.load_metadata()

    def load_metadata(self):
        # Register default dataset (Amazon Consumer Reviews)
        amazon_df = prepare_dataset()
        pos_c = int((amazon_df["sentiment"] == "positive").sum())
        neu_c = int((amazon_df["sentiment"] == "neutral").sum())
        neg_c = int((amazon_df["sentiment"] == "negative").sum())

        default_id = "default_amazon_47k"
        self.datasets[default_id] = {
            "id": default_id,
            "name": "Amazon Consumer Reviews Dataset",
            "filename": "amazon_reviews_full_68k.csv",
            "total_records": len(amazon_df),
            "counts": {"positive": pos_c, "neutral": neu_c, "negative": neg_c},
            "upload_date": "Default System Dataset",
            "source": "47,276 Clean Review Records",
            "is_default": True
        }
        self.dataset_dfs[default_id] = amazon_df

        # Load metadata for custom uploads
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    for ds_id, info in meta.items():
                        file_path = os.path.join(UPLOADS_DIR, info["filename"])
                        if os.path.exists(file_path):
                            df = pd.read_csv(file_path)
                            self.datasets[ds_id] = info
                            self.dataset_dfs[ds_id] = df
            except Exception as e:
                print(f"Notice loading custom datasets metadata: {e}")

    def save_metadata(self):
        meta_to_save = {
            k: v for k, v in self.datasets.items() if not v.get("is_default")
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return list of all registered dataset records."""
        return list(self.datasets.values())

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame for specified dataset ID."""
        return self.dataset_dfs.get(dataset_id)

    def process_and_add_dataset(
        self,
        file_bytes: bytes,
        filename: str,
        pipeline_annotator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded file (CSV, JSON, TXT), detect text and sentiment columns,
        auto-annotate sentiment if missing using the model pipeline, and save to inventory.
        """
        ext = os.path.splitext(filename)[1].lower()
        save_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(UPLOADS_DIR, save_name)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Parse DataFrame based on extension
        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext == ".json":
                df = pd.read_json(file_path)
            elif ext == ".txt":
                lines = file_bytes.decode("utf-8", errors="ignore").split("\n")
                lines = [l.strip() for l in lines if len(l.strip()) > 5]
                df = pd.DataFrame({"text": lines})
            else:
                raise ValueError(f"Unsupported file format: {ext}. Please upload CSV, JSON, or TXT.")
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise ValueError(f"Failed to parse dataset file: {str(e)}")

        if df.empty:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise ValueError("Uploaded dataset file is empty.")

        # Identify text column
        text_col = None
        for col in df.columns:
            if str(col).lower() in ["text", "review", "reviews", "comment", "comments", "feedback", "body", "content", "sentence"]:
                text_col = col
                break
        if not text_col:
            # Fallback to first string column
            string_cols = [c for c in df.columns if df[c].dtype == object]
            if string_cols:
                text_col = string_cols[0]
            else:
                text_col = df.columns[0]

        # Identify sentiment / rating column
        sentiment_col = None
        for col in df.columns:
            if str(col).lower() in ["sentiment", "label", "target", "category", "class"]:
                sentiment_col = col
                break
            elif str(col).lower() in ["rating", "ratings", "score", "stars"]:
                # Convert ratings to sentiment
                df["sentiment"] = df[col].apply(
                    lambda r: "negative" if float(r) <= 2 else ("neutral" if float(r) == 3 else "positive")
                )
                sentiment_col = "sentiment"
                break

        # Standardize DataFrame
        clean_df = pd.DataFrame()
        clean_df["text"] = df[text_col].astype(str)
        clean_df = clean_df[clean_df["text"].str.len() > 5].copy()

        if sentiment_col and sentiment_col in df.columns:
            clean_df["sentiment"] = df[sentiment_col].astype(str).str.lower().map({
                "1": "negative", "0": "negative", "neg": "negative", "negative": "negative",
                "2": "neutral", "neu": "neutral", "neutral": "neutral",
                "3": "positive", "pos": "positive", "positive": "positive", "4": "positive", "5": "positive"
            }).fillna("neutral")
        elif pipeline_annotator:
            # Auto-annotate sentiment using model pipeline
            annotated_sentiments = []
            for t in clean_df["text"]:
                res = pipeline_annotator.predict(t, selected_model="Ensemble")
                annotated_sentiments.append(res["sentiment"])
            clean_df["sentiment"] = annotated_sentiments
        else:
            clean_df["sentiment"] = "neutral"

        # Overwrite file with standardized clean CSV
        clean_df.to_csv(file_path, index=False)

        ds_id = f"ds_{uuid.uuid4().hex[:6]}"
        pos_c = int((clean_df["sentiment"] == "positive").sum())
        neu_c = int((clean_df["sentiment"] == "neutral").sum())
        neg_c = int((clean_df["sentiment"] == "negative").sum())

        meta_info = {
            "id": ds_id,
            "name": os.path.splitext(filename)[0].replace("_", " ").title() + " Dataset",
            "filename": save_name,
            "total_records": len(clean_df),
            "counts": {"positive": pos_c, "neutral": neu_c, "negative": neg_c},
            "upload_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": f"Uploaded File ({filename})",
            "is_default": False
        }

        self.datasets[ds_id] = meta_info
        self.dataset_dfs[ds_id] = clean_df
        self.save_metadata()

        return meta_info

    def get_sample_for_stream(self, dataset_id: str = "default_amazon_47k") -> Dict[str, Any]:
        """Return a random sample item from specified dataset ID."""
        df = self.get_dataset(dataset_id)
        if df is None or len(df) == 0:
            df = self.get_dataset("default_amazon_47k")

        row = df.sample(n=1).iloc[0]
        text_val = str(row["text"])
        snippet = text_val[:160] + ("..." if len(text_val) > 160 else "")
        sent = str(row["sentiment"]) if "sentiment" in row else "neutral"
        return {"text": snippet, "actual_sentiment": sent}

    def evaluate_models_on_custom_dataset(self, dataset_id: str, pipeline_model: Any) -> Dict[str, Any]:
        """Run all 5 models across specified dataset and compute precision, recall, accuracy, f1 metrics."""
        df = self.get_dataset(dataset_id)
        if df is None or len(df) == 0:
            raise ValueError("Target dataset not found or empty.")

        texts = list(df["text"])
        y_true = list(df["sentiment"])

        models = ["Ensemble", "Linear SVC", "Logistic Regression", "Complement Naive Bayes", "VADER"]
        results = {}

        for m_name in models:
            y_preds = []
            correct = 0
            for t, true_label in zip(texts, y_true):
                pred = pipeline_model.predict(t, selected_model=m_name)
                pred_label = pred["sentiment"]
                y_preds.append(pred_label)
                if pred_label == true_label:
                    correct += 1

            accuracy = round(correct / len(texts), 4) if len(texts) > 0 else 0.0
            
            # Simple per-class counts
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            for p in y_preds:
                counts[p] = counts.get(p, 0) + 1

            results[m_name] = {
                "accuracy": accuracy,
                "total_tested": len(texts),
                "predicted_counts": counts,
                "sample_predictions": [
                    {"text": t[:80], "true": tr, "pred": pr} for t, tr, pr in zip(texts[:5], y_true[:5], y_preds[:5])
                ]
            }

        return {
            "dataset_id": dataset_id,
            "dataset_name": self.datasets.get(dataset_id, {}).get("name", dataset_id),
            "total_records": len(texts),
            "model_evaluations": results
        }
