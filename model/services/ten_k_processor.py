import json
import logging
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from model.thesis_agents.llm_helper import LLMHelperMixin
from model.prompts.ten_k_prompts import EXTRACTION_PROMPT
from model.prompts.ten_k_schemas import TEN_K_EXTRACTION_SCHEMA

logger = logging.getLogger(__name__)


class TenKProcessor(LLMHelperMixin):
    """Processes 10-K documents and extracts signals using structured LLM calls."""

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".htm", ".html")
    EXTRACTION_FILE_EXT: str = ".json"
    DEFAULT_MAX_PAGES: int = 100
    DEFAULT_MAX_TOKENS: int = 8000
    DEFAULT_TEMPERATURE: float = 0.0
    
    def __init__(self, stock_ticker: str):
        """Initialize processor.

        Args:
            stock_ticker: Stock ticker symbol
        """
        self.stock_ticker = stock_ticker
        logger.debug("Initialized TenKProcessor for %s", stock_ticker)
    
    def extract_text_from_pdf(self, filepath: Path, max_pages: Optional[int] = None) -> str:
        """Extract text from PDF file.

        Args:
            filepath: Path to PDF file
            max_pages: Maximum pages to extract

        Returns:
            Extracted text
        """
        if max_pages is None:
            max_pages = self.DEFAULT_MAX_PAGES

        try:
            reader = PdfReader(filepath)
            texts = []
            logger.debug("Extracting text from %s (%s pages, max %s)", filepath.name, len(reader.pages), max_pages)
            
            for i, page in enumerate(reader.pages[:max_pages]):
                try:
                    text = page.extract_text() or ""
                    texts.append(text)
                except Exception as e:
                    logger.warning("Could not extract page %s from %s: %s", i + 1, filepath.name, e)
                    continue

            result = "\n\n".join(texts)
            logger.info("Extracted %s chars from %s", len(result), filepath.name)
            return result
        except Exception as e:
            logger.error("Failed to extract PDF text from %s: %s", filepath.name, e)
            return ""
    
    def extract_text_from_html(self, filepath: Path) -> str:
        """Extract text from HTML file.

        Args:
            filepath: Path to HTML file

        Returns:
            Extracted text
        """
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            logger.info("Extracted %s chars from HTML %s", len(text), filepath.name)
            return text
        except Exception as e:
            logger.error("Failed to extract HTML text from %s: %s", filepath.name, e)
            return ""
    
    def extract_from_file(
        self,
        filepath: Path,
        max_pages: Optional[int] = None,
        call_model: bool = True,
    ) -> Dict[str, Any]:
        """Extract text and optionally call LLM for signal extraction.

        Args:
            filepath: Path to 10-K file
            max_pages: Maximum pages to extract from PDF
            call_model: Whether to call LLM for extraction

        Returns:
            Extraction dict with metadata, text length, and signals
        """
        if max_pages is None:
            max_pages = self.DEFAULT_MAX_PAGES

        filename = filepath.name

        if filename.endswith(".pdf"):
            text = self.extract_text_from_pdf(filepath, max_pages=max_pages)
        elif filename.endswith((".htm", ".html")):
            text = self.extract_text_from_html(filepath)
        else:
            text = self._extract_text_by_detection(filepath, max_pages)

        text_length = len(text)

        result: Dict[str, Any] = {
            "meta": {
                "file": filename,
                "filepath": str(filepath),
                "extracted_at": datetime.now().isoformat(),
                "text_length": text_length,
            },
            "text_length": text_length,
        }

        if not call_model:
            result["text_preview"] = text[:500]
            logger.debug("Text extraction only for %s (%s chars)", filename, text_length)
            return result

        logger.info("Calling LLM for %s (%s chars)", filename, text_length)
        try:
            prompt = f"{EXTRACTION_PROMPT}\n\n10-K TEXT (first {max_pages} pages):\n\n{text}"
            parsed = self._call_llm_structured(
                prompt=prompt,
                schema=TEN_K_EXTRACTION_SCHEMA,
                max_tokens=self.DEFAULT_MAX_TOKENS,
                temperature=self.DEFAULT_TEMPERATURE
            )
            result["signals"] = parsed
            logger.info("Extracted %s signal categories from %s", len(parsed) if parsed else 0, filename)
        except Exception as e:
            logger.error("Signal extraction failed for %s: %s", filename, e)
            result["signals"] = None
            result["extraction_error"] = str(e)

        return result
    
    def _extract_text_by_detection(self, filepath: Path, max_pages: int) -> str:
        """Detect file type by content and extract text.

        Args:
            filepath: Path to file
            max_pages: Max pages for PDF extraction

        Returns:
            Extracted text
        """
        try:
            with open(filepath, "rb") as f:
                header = f.read(10)
            if header.startswith(b"%PDF"):
                return self.extract_text_from_pdf(filepath, max_pages=max_pages)
            else:
                return self.extract_text_from_html(filepath)
        except Exception as e:
            logger.error("Could not detect file type for %s: %s", filepath.name, e)
            return ""
    
    def _get_extraction_filename(self, source_filename: str) -> str:
        """Generate extraction JSON filename from source document filename.

        Args:
            source_filename: Original document filename

        Returns:
            JSON filename for extraction
        """
        for ext in self.SUPPORTED_EXTENSIONS:
            if source_filename.endswith(ext):
                return source_filename.replace(ext, self.EXTRACTION_FILE_EXT)
        return source_filename + self.EXTRACTION_FILE_EXT
    
    def save_extraction(self, extraction: Dict[str, Any], output_dir: Path) -> Path:
        """Save extraction to JSON file.

        Args:
            extraction: Extraction dict
            output_dir: Directory to save to

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        source_filename = extraction["meta"]["file"]
        filename = self._get_extraction_filename(source_filename)
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(extraction, f, indent=2)

        logger.info("Saved extraction to %s", filepath)
        return filepath
