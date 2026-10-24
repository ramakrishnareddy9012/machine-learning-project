import re
import urllib.parse
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup

class WebScraper:
    """
    Enhanced Web Scraping & Content Extraction Engine.
    Extracts text, articles, customer reviews, blog content from live URLs or raw HTML text,
    filtering out script noise, headers, footers, and HTML markup.
    """
    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
        ]

    def _get_headers(self, index: int = 0) -> Dict[str, str]:
        ua = self.user_agents[index % len(self.user_agents)]
        return {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "max-age=0",
            "Upgrade-Insecure-Requests": "1"
        }

    def fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch URL content safely with user-agent rotation and SSL verification fallback."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            # Try normal GET with User-Agent 0
            response = requests.get(url, headers=self._get_headers(0), timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            return {
                "success": True,
                "status_code": response.status_code,
                "url": response.url,
                "html": response.text,
                "error": None
            }
        except Exception as e:
            # Fallback 1: Try with SSL verify=False if SSL error occurred
            err_str = str(e)
            if "SSLError" in err_str or "certificate" in err_str.lower():
                try:
                    response = requests.get(url, headers=self._get_headers(1), timeout=self.timeout, verify=False)
                    response.raise_for_status()
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "url": response.url,
                        "html": response.text,
                        "error": None
                    }
                except Exception as inner_e:
                    err_str = str(inner_e)

            # Fallback 2: Try User-Agent 2
            try:
                response = requests.get(url, headers=self._get_headers(2), timeout=self.timeout)
                response.raise_for_status()
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "url": response.url,
                    "html": response.text,
                    "error": None
                }
            except Exception as final_e:
                return {
                    "success": False,
                    "status_code": getattr(final_e.response, "status_code", None) if hasattr(final_e, "response") else None,
                    "url": url,
                    "html": "",
                    "error": str(final_e)
                }

    def clean_html(self, html_content: str) -> Dict[str, Any]:
        """
        Remove HTML noise (scripts, styles, nav, footers, forms) and extract readable text snippets.
        Supports article tags, divs, paragraphs, lists, headings, and quotes.
        """
        if not html_content or not html_content.strip():
            return {"title": "", "extracted_texts": [], "full_text": ""}

        soup = BeautifulSoup(html_content, "html.parser")

        # Decompose noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form", "iframe", "svg", "button", "input"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        else:
            title = "Extracted Web Page"

        text_elements = []
        seen = set()

        # Target tags: main content containers first, then general content tags
        content_containers = soup.find_all(["article", "main", "section"]) or [soup]

        for container in content_containers:
            for elem in container.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote", "div", "span"]):
                # Skip if tag contains other block children to avoid duplicate text
                if elem.name in ["div", "span"] and elem.find(["p", "div", "h1", "h2", "h3"]):
                    continue

                raw_t = elem.get_text()
                cleaned_text = re.sub(r'\s+', ' ', raw_t).strip()
                
                # Check line quality: length > 12 chars, not pure punctuation/numbers
                if len(cleaned_text) >= 12 and cleaned_text not in seen:
                    # Exclude typical menu or cookie consent phrases
                    lowered = cleaned_text.lower()
                    if any(ignore_phrase in lowered for ignore_phrase in ["cookie policy", "accept all cookies", "all rights reserved", "privacy policy", "terms of use"]):
                        continue

                    seen.add(cleaned_text)
                    text_elements.append(cleaned_text)

        full_text = " ".join(text_elements)
        return {
            "title": title,
            "extracted_texts": text_elements,
            "full_text": full_text
        }

    def scrape_and_extract(self, url: str) -> Dict[str, Any]:
        """Scrape target web page URL and extract cleaned text content."""
        result = self.fetch_url(url)
        if not result["success"]:
            return {
                "success": False,
                "url": result["url"],
                "title": "URL Fetch Failed",
                "total_snippets": 0,
                "text_snippets": [],
                "full_text": "",
                "fetch_notice": f"Unable to fetch URL directly ({result['error']}). You can paste raw HTML or web page text in Direct Web Scraper mode.",
                "error": result["error"]
            }

        extracted = self.clean_html(result["html"])
        if not extracted["extracted_texts"]:
            return {
                "success": False,
                "url": result["url"],
                "title": extracted["title"] or "No readable text",
                "total_snippets": 0,
                "text_snippets": [],
                "full_text": "",
                "fetch_notice": "Page loaded but no main readable text paragraphs were extracted. Try pasting webpage content into Direct Web Scraper mode.",
                "error": "No text elements found"
            }

        return {
            "success": True,
            "url": result["url"],
            "title": extracted["title"],
            "total_snippets": len(extracted["extracted_texts"]),
            "text_snippets": extracted["extracted_texts"],
            "full_text": extracted["full_text"],
            "fetch_notice": "Successfully scraped and extracted live web page content."
        }

    def scrape_raw_html_or_text(self, raw_input: str, source_title: str = "Pasted Web Content") -> Dict[str, Any]:
        """Scrape and extract text snippets directly from raw HTML string or plain text block."""
        if not raw_input.strip():
            return {
                "success": False,
                "url": "direct_input",
                "title": source_title,
                "total_snippets": 0,
                "text_snippets": [],
                "full_text": "",
                "fetch_notice": "Input content is empty."
            }

        if "<html" in raw_input.lower() or "<body" in raw_input.lower() or "<p>" in raw_input.lower() or "<div>" in raw_input.lower():
            extracted = self.clean_html(raw_input)
            snippets = extracted["extracted_texts"]
            title = extracted["title"] if extracted["title"] else source_title
        else:
            # Plain text paragraphs
            snippets = [re.sub(r'\s+', ' ', line).strip() for line in raw_input.split("\n") if len(line.strip()) >= 10]
            title = source_title

        full_text = " ".join(snippets)
        return {
            "success": True,
            "url": "Direct Input / Pasted Content",
            "title": title,
            "total_snippets": len(snippets),
            "text_snippets": snippets,
            "full_text": full_text,
            "fetch_notice": "Successfully parsed and extracted text snippets from input content."
        }

