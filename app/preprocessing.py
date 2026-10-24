import re
import string
from typing import List, Dict, Any

# Standard English Stopwords list
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

class TextPreprocessor:
    """
    NLP Preprocessing Engine:
    Cleans raw text data step-by-step for feature extraction and ML modeling.
    """
    def __init__(self, stop_words: set = None):
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS

    def remove_html_and_urls(self, text: str) -> str:
        """Step 1: Remove HTML tags and web links/URLs."""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
        return text

    def to_lowercase(self, text: str) -> str:
        """Step 2: Convert text to lower case."""
        return text.lower()

    def remove_special_characters(self, text: str) -> str:
        """Step 3: Remove punctuation, numbers, and non-alphabetic characters."""
        # Replace non-alphanumeric (except spaces) with space
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Step 4: Tokenize text into individual words."""
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Step 5: Filter out common English stop words."""
        return [word for word in tokens if word not in self.stop_words and len(word) > 1]

    def normalize(self, tokens: List[str]) -> List[str]:
        """Step 6: Simple suffix normalization (stemming/lemmatization simulation)."""
        normalized = []
        for word in tokens:
            # Simple rule-based suffix normalization
            if word.endswith("ing") and len(word) > 5:
                word = word[:-3]
            elif word.endswith("ed") and len(word) > 4:
                word = word[:-2]
            elif word.endswith("es") and len(word) > 4:
                word = word[:-2]
            elif word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
                word = word[:-1]
            normalized.append(word)
        return normalized

    def preprocess(self, raw_text: str) -> str:
        """Complete pipeline: Raw text -> Processed clean string."""
        if not raw_text or not raw_text.strip():
            return ""

        step1 = self.remove_html_and_urls(raw_text)
        step2 = self.to_lowercase(step1)
        step3 = self.remove_special_characters(step2)
        tokens = self.tokenize(step3)
        filtered = self.remove_stopwords(tokens)
        normalized = self.normalize(filtered)
        
        return " ".join(normalized)

    def inspect_pipeline(self, raw_text: str) -> Dict[str, Any]:
        """
        Return step-by-step trace of preprocessing pipeline for demonstration
        and project review visual inspection.
        """
        step1_no_html = self.remove_html_and_urls(raw_text)
        step2_lowercase = self.to_lowercase(step1_no_html)
        step3_no_special = self.remove_special_characters(step2_lowercase)
        step4_tokens = self.tokenize(step3_no_special)
        step5_no_stopwords = self.remove_stopwords(step4_tokens)
        step6_normalized = self.normalize(step5_no_stopwords)
        final_clean = " ".join(step6_normalized)

        return {
            "raw_text": raw_text,
            "step_1_html_urls_removed": step1_no_html,
            "step_2_lowercased": step2_lowercase,
            "step_3_special_chars_removed": step3_no_special,
            "step_4_tokenized": step4_tokens,
            "step_5_stopwords_removed": step5_no_stopwords,
            "step_6_normalized_tokens": step6_normalized,
            "final_processed_text": final_clean
        }
