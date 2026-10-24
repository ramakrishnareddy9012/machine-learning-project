import unittest
import io
from fastapi.testclient import TestClient

from app.api import app
from app.scraper import WebScraper
from app.preprocessing import TextPreprocessor
from app.model import SentimentModelPipeline
from app.dataset_manager import DatasetManager

class TestSentimentPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.preprocessor = TextPreprocessor()
        self.pipeline = SentimentModelPipeline()
        self.scraper = WebScraper()
        self.ds_manager = DatasetManager()

    def test_preprocessor(self):
        raw = "<h1>WOW!!!</h1> This product is <b>AMAZING</b>! Check http://example.com"
        cleaned = self.preprocessor.preprocess(raw)
        self.assertNotIn("<h1>", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertIn("amaz", cleaned)

        trace = self.preprocessor.inspect_pipeline(raw)
        self.assertIn("step_1_html_urls_removed", trace)
        self.assertIn("final_processed_text", trace)

    def test_scraper_clean_html_and_raw_text(self):
        html = "<html><head><title>Test Page</title></head><body><script>var x=10;</script><p>This product is great and fast performance.</p></body></html>"
        res = self.scraper.clean_html(html)
        self.assertEqual(res["title"], "Test Page")
        self.assertIn("This product is great and fast performance.", res["full_text"])

        raw_res = self.scraper.scrape_raw_html_or_text(html, source_title="Test Direct")
        self.assertTrue(raw_res["success"])
        self.assertIn("This product is great and fast performance.", raw_res["text_snippets"])

    def test_model_training_and_prediction(self):
        metrics = self.pipeline.train_and_evaluate()
        self.assertIn("Hybrid Ensemble Classifier", metrics)
        self.assertIn("Logistic Regression (TF-IDF)", metrics)
        self.assertIn("Complement Naive Bayes", metrics)

        pos_res = self.pipeline.predict("This product is brilliant and awesome!")
        self.assertEqual(pos_res["sentiment"], "positive")

        neg_res = self.pipeline.predict("Worst item ever! Terrible quality and broke immediately.")
        self.assertEqual(neg_res["sentiment"], "negative")

    def test_dataset_upload_and_management(self):
        csv_content = b"text,rating\nThis camera is awesome and takes crisp photos!,5\nBattery died instantly and case broke.,1\nAverage product nothing special.,3\n"
        file_obj = ("custom_test.csv", io.BytesIO(csv_content), "text/csv")
        
        # Upload endpoint
        upload_res = self.client.post("/api/dataset/upload", files={"file": file_obj})
        self.assertEqual(upload_res.status_code, 200)
        json_data = upload_res.json()
        self.assertTrue(json_data["success"])
        ds_id = json_data["dataset"]["id"]

        # List datasets
        list_res = self.client.get("/api/dataset/list")
        self.assertEqual(list_res.status_code, 200)
        self.assertGreaterEqual(list_res.json()["total_datasets"], 2)

        # Multi-model evaluation on uploaded dataset
        eval_res = self.client.post(f"/api/dataset/{ds_id}/evaluate")
        self.assertEqual(eval_res.status_code, 200)
        self.assertIn("model_evaluations", eval_res.json())
        self.assertIn("Ensemble", eval_res.json()["model_evaluations"])

        # Stream sample with dataset selection
        stream_res = self.client.get(f"/api/stream/sample?dataset_id={ds_id}&model_name=Ensemble")
        self.assertEqual(stream_res.status_code, 200)
        self.assertEqual(stream_res.json()["dataset_id"], ds_id)

    def test_api_endpoints(self):
        # Health
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

        # Text analysis
        res_text = self.client.post("/api/analyze/text", json={"text": "Super fast delivery and fantastic item!"})
        self.assertEqual(res_text.status_code, 200)
        self.assertEqual(res_text.json()["sentiment"], "positive")

        # Raw web text analysis
        res_raw = self.client.post("/api/analyze/rawtext", json={"html_or_text": "<div>The build quality is outstanding and crystal clear audio!</div>"})
        self.assertEqual(res_raw.status_code, 200)
        self.assertTrue(res_raw.json()["success"])

        # Batch analysis
        res_batch = self.client.post("/api/analyze/batch", json={"texts": ["Great quality!", "Horrible item!"]})
        self.assertEqual(res_batch.status_code, 200)
        self.assertEqual(res_batch.json()["total_analyzed"], 2)

        # Metrics
        res_metrics = self.client.get("/api/model/metrics")
        self.assertEqual(res_metrics.status_code, 200)
        self.assertIn("Hybrid Ensemble Classifier", res_metrics.json())

if __name__ == "__main__":
    unittest.main()

