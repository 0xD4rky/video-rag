import os
from typing import Optional

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _GENAI_AVAILABLE = False

from transformers import pipeline


class QueryExpander:
    """Expand user queries using Gemini if available, otherwise a tiny local model."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key and _GENAI_AVAILABLE:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-pro")
            self.backend = "gemini"
        else:
            # Fallback to a very small text generation model
            self.generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")
            self.backend = "local"

    def expand(self, query: str) -> str:
        if self.backend == "gemini":
            try:
                resp = self.model.generate_content(f"Provide a detailed search query for: {query}")
                return resp.text.strip()
            except Exception:
                pass
        result = self.generator(f"Expand: {query}", max_new_tokens=20)[0]["generated_text"]
        return result.strip()
