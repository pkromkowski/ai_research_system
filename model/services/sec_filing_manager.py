import json
import logging
import requests
import yfinance as yf
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from model.core.config import FilingConfig

logger = logging.getLogger(__name__)


class SecFilingManager:
    """Manages SEC 10-K filings for a company."""

    PRIMARY_FILING_TYPE: str = "10-K"
    FALLBACK_FILING_TYPE: str = "10-Q"
    EXHIBIT_KEYS: Tuple[str, ...] = (
        "10-K",
        "10-Q",
        "10-K/A",
        "10-K405",
        "10-Q/A",
        "10-QSB",
    )
    DEFAULT_MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    
    def __init__(
        self,
        ticker: str,
        config: Optional[FilingConfig] = None,
        base_dir: Optional[str] = None,
        ticker_factory: Optional[Callable[[str], Any]] = None,
    ):
        """Initialize filing manager for ticker."""
        self.ticker = ticker.upper()
        if config is None:
            if base_dir:
                config = FilingConfig(base_dir=Path(base_dir))
            else:
                config = FilingConfig()

        self.config = config
        self.base_dir = self.config.base_dir
        self.raw_dir = self.config.get_raw_dir(self.ticker)
        self.metadata_dir = self.config.get_metadata_dir(self.ticker)
        self._ticker_factory: Callable[[str], Any] = ticker_factory or (lambda t: yf.Ticker(t))
        self._stock: Optional[Any] = None
        self._metadata: Optional[Dict[str, Any]] = None
        logger.debug("Initializing SecFilingManager for %s", self.ticker, extra={"ticker": self.ticker, "raw_dir": str(self.raw_dir)})
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.metadata_dir / "filings_metadata.json"
    
    @property
    def stock(self) -> Any:
        """Lazy-load ticker object."""
        if self._stock is None:
            try:
                self._stock = self._ticker_factory(self.ticker)
            except Exception as e:
                logger.warning("Could not create ticker object for %s: %s", self.ticker, e)
                self._stock = None
        return self._stock
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Lazy-load filings metadata."""
        if self._metadata is None:
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file, "r") as f:
                        self._metadata = json.load(f)
                except Exception as e:
                    logger.warning("Could not read metadata file %s: %s", self.metadata_file, e)
                    self._metadata = {
                        "ticker": self.ticker,
                        "last_updated": None,
                        "processed_filings": {},
                    }
            else:
                self._metadata = {
                    "ticker": self.ticker,
                    "last_updated": None,
                    "processed_filings": {},
                }
        return self._metadata
    
    def _save_metadata(self) -> None:
        """Save filings metadata to disk."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error("Failed to save metadata to %s: %s", self.metadata_file, e)
    
    def _get_filename_from_url(self, url: str, extension: str = ".pdf") -> str:
        """Extract filename from URL or generate one."""
        try:
            parts = url.split("/")
            if len(parts) > 0:
                accession_id = parts[-2] if parts[-1].endswith("index.htm") else parts[-1]
                if accession_id.endswith((".pdf", ".htm", ".html")):
                    accession_id = accession_id.rsplit(".", 1)[0]
                return f"{self.ticker}_{accession_id}{extension}"
        except Exception:
            pass
        return f"{self.ticker}_{datetime.now().timestamp()}{extension}"
    
    def _normalize_date(self, date_val: Any) -> Optional[str]:
        """Normalize date to ISO format string."""
        if date_val is None:
            return None
        if hasattr(date_val, "isoformat"):
            return date_val.isoformat()
        return str(date_val)
    
    def _extract_filing_id(self, filing: Dict[str, Any]) -> Optional[str]:
        """Extract unique filing ID from filing data."""
        if filing.get("accession"):
            return filing["accession"]
        document_url = filing.get("document_url", "")
        if document_url:
            parts = document_url.split("/")
            for part in reversed(parts):
                if part and part[0].isdigit() and len(part) > 10:
                    return part
        url = filing.get("url", "")
        if url:
            parts = url.split("/")
            if len(parts) > 1:
                return parts[-2]
        if document_url:
            return f"{self.ticker}_{document_url.split('/')[-1]}"
        return None
    
    def _get_document_url_from_filing(self, filing: Dict[str, Any]) -> Optional[str]:
        """Extract document URL from filing exhibits."""
        try:
            exhibits = filing.get("exhibits", {})
            if not isinstance(exhibits, dict):
                return None
            for key in self.EXHIBIT_KEYS:
                if key in exhibits:
                    return exhibits[key]
            filing_url = filing.get("url")
            if filing_url and "sec.gov" in filing_url:
                return filing_url
            return None
        except Exception as e:
            logger.debug(f"Error extracting document URL: {e}")
            return None
    
    def fetch_10k_filings(self) -> List[Dict[str, Any]]:
        """Fetch 10-K filings with 10-Q fallback for missing years."""
        try:
            filings = self.stock.sec_filings
            if not filings:
                return []
            primary_filings = [
                f for f in filings 
                if f.get("type") == self.PRIMARY_FILING_TYPE
            ]
            fallback_filings = [
                f for f in filings 
                if f.get("type") == self.FALLBACK_FILING_TYPE
            ]
            result = []
            for filing in primary_filings:
                document_url = self._get_document_url_from_filing(filing)
                if document_url:
                    filing_dict = {
                        "type": filing.get("type"),
                        "date": self._normalize_date(filing.get("date")),
                        "accession": filing.get("accession"),
                        "url": filing.get("url"),
                        "exhibits": filing.get("exhibits", {}),
                        "document_url": document_url,
                    }
                    result.append(filing_dict)
            if primary_filings and fallback_filings:
                primary_years = {
                    filing.get("date").year 
                    for filing in primary_filings 
                    if filing.get("date") and hasattr(filing.get("date"), "year")
                }
                for q_filing in fallback_filings:
                    q_date = q_filing.get("date")
                    if not q_date or not hasattr(q_date, "year"):
                        continue
                    if q_date.year in primary_years:
                        continue
                    has_fallback_for_year = any(
                        r.get("type") == self.FALLBACK_FILING_TYPE
                        and str(q_date.year) in str(r.get("date", ""))
                        for r in result
                    )
                    if has_fallback_for_year:
                        continue
                    document_url = self._get_document_url_from_filing(q_filing)
                    if document_url:
                        filing_dict = {
                            "type": q_filing.get("type"),
                            "date": self._normalize_date(q_date),
                            "accession": q_filing.get("accession"),
                            "url": q_filing.get("url"),
                            "exhibits": q_filing.get("exhibits", {}),
                            "document_url": document_url,
                        }
                        result.append(filing_dict)
            return sorted(
                result,
                key=lambda f: self._parse_date_for_sort(f.get("date")),
                reverse=True,
            )
        except Exception as e:
            logger.error("Error fetching filings for %s: %s", self.ticker, e)
            return []
    
    def _parse_date_for_sort(self, date_val: Any) -> datetime:
        """Parse date value for sorting purposes."""
        if isinstance(date_val, str):
            try:
                return datetime.fromisoformat(date_val)
            except ValueError:
                return datetime.min
        elif hasattr(date_val, "isoformat"):
            return date_val if isinstance(date_val, datetime) else datetime.min
        return datetime.min
    
    def check_new_filings(self) -> List[Dict[str, Any]]:
        """Check for new unprocessed 10-K filings."""
        all_filings = self.fetch_10k_filings()
        if not all_filings:
            logger.warning("No filings found for %s", self.ticker)
            return []
        logger.info("Checking %s filings for %s", len(all_filings), self.ticker, extra={"ticker": self.ticker, "total_filings": len(all_filings)})
        processed_ids = set(self.metadata.get("processed_filings", {}).keys())
        new_filings = []
        for filing in all_filings:
            if not isinstance(filing, dict):
                continue
            filing_id = self._extract_filing_id(filing)
            if filing_id and filing_id not in processed_ids:
                new_filings.append(filing)
            elif not filing_id:
                logger.warning("Could not generate ID for filing", extra={"ticker": self.ticker, "type": filing.get("type"), "date": filing.get("date")})
        logger.info("Found %s new filings for %s", len(new_filings), self.ticker)
        return new_filings
    
    def download_filing(
        self,
        filing: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> Tuple[bool, Optional[Path]]:
        """Download filing document, trying PDF first then HTML."""
        if max_retries is None:
            max_retries = getattr(self.config, "max_retries", self.DEFAULT_MAX_RETRIES)
        timeout = getattr(self.config, "timeout_seconds", self.REQUEST_TIMEOUT)
        url = filing.get("document_url")
        if not url:
            logger.error("No document URL for filing: %s", self.ticker)
            return False, None
        logger.debug("Starting download for %s: %s", self.ticker, url)
        urls_to_try = []
        if url.endswith((".htm", ".html")):
            pdf_url = url.replace(".htm", ".pdf").replace(".html", ".pdf")
            urls_to_try.append((pdf_url, ".pdf"))
            urls_to_try.append((url, ".htm"))
        else:
            urls_to_try.append((url, ".pdf"))
        for download_url, extension in urls_to_try:
            filename = self._get_filename_from_url(
                filing.get("url", download_url), 
                extension,
            )
            filepath = self.raw_dir / filename
            if filepath.exists():
                logger.debug("File already exists: %s", filename)
                return True, filepath
            logger.info("Downloading %s filing (%s)", self.ticker, extension)
            for attempt in range(max_retries):
                try:
                    response = requests.get(download_url, timeout=timeout)
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    logger.info("Successfully downloaded: %s", filename)
                    return True, filepath
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        logger.warning("Download failed after %s retries: %s - %s", max_retries, download_url, e)
                    else:
                        logger.debug("Download attempt %s/%s failed for %s, retrying", attempt + 1, max_retries, download_url)
        logger.error("Failed to download filing after trying all formats: %s", url)
        return False, None
    
    def mark_filing_processed(
        self,
        filing: Dict[str, Any],
        status: str = "processed",
    ) -> None:
        """Mark filing as processed."""
        filing_id = self._extract_filing_id(filing)

        if not filing_id:
            filing_id = f"{self.ticker}_{datetime.now().timestamp()}"
        self.metadata["processed_filings"][filing_id] = {
            "date": self._normalize_date(filing.get("date")),
            "url": filing.get("url"),
            "filing_type": filing.get("type", "unknown"),
            "status": status,
            "processed_at": datetime.now().isoformat(),
        }
        self._save_metadata()

    def get_processed_filings(self) -> List[str]:
        """Return list of processed filing accession IDs."""
        return list(self.metadata.get("processed_filings", {}).keys())

    def clear_cache(self) -> None:
        """Clear cached ticker and metadata."""
        self._stock = None
        self._metadata = None
        logger.debug("Cleared SecFilingManager cache for %s", self.ticker)
