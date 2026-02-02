import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

from model.core.config import FilingConfig
from model.services.ten_k_processor import TenKProcessor
from model.services.sec_filing_manager import SecFilingManager

logger = logging.getLogger(__name__)


class TenKOrchestrator:
    """Orchestrates 10-K filing download, processing, and storage."""

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".htm", ".html")
    EXTRACTION_FILE_EXT: str = ".json"
    
    def __init__(
        self,
        ticker: str,
        config: Optional[FilingConfig] = None,
        base_dir: Optional[str] = None,
        filing_manager_factory: Optional[Callable[[str, Optional[FilingConfig]], SecFilingManager]] = None,
        processor_factory: Optional[Callable[[], TenKProcessor]] = None,
    ):
        """Initialize orchestrator.

        Args:
            ticker: Stock ticker symbol
            config: FilingConfig instance
            base_dir: Base directory override
            filing_manager_factory: Factory for SecFilingManager
            processor_factory: Factory for TenKProcessor
        """
        self.ticker = ticker.upper()

        if config is None:
            if base_dir:
                config = FilingConfig(base_dir=Path(base_dir))
            else:
                config = FilingConfig()

        self.config = config
        self.base_dir = self.config.base_dir

        self._filing_manager_factory = filing_manager_factory or (lambda t, cfg: SecFilingManager(t, config=cfg))
        self._processor_factory = processor_factory or (lambda: TenKProcessor(stock_ticker=self.ticker))

        self._filing_manager: Optional[SecFilingManager] = None
        self._processor: Optional[TenKProcessor] = None

        self.raw_dir = self.config.get_raw_dir(self.ticker)
        self.extractions_dir = self.config.get_extractions_dir(self.ticker)
        self.metadata_dir = self.config.get_metadata_dir(self.ticker)

        self.extractions_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Initialized TenKOrchestrator for %s", self.ticker)
    
    @property
    def filing_manager(self) -> SecFilingManager:
        """Lazy-load filing manager."""
        if self._filing_manager is None:
            self._filing_manager = self._filing_manager_factory(self.ticker, self.config)
        return self._filing_manager

    @property
    def processor(self) -> TenKProcessor:
        """Lazy-load processor."""
        if self._processor is None:
            self._processor = self._processor_factory()
        return self._processor
    
    def sync_and_process(self, force_reprocess: bool = False, call_model: bool = True) -> Dict[str, Any]:
        """Sync filings and process any new ones.

        Args:
            force_reprocess: Reprocess all filings even if cached
            call_model: Call the model during extraction

        Returns:
            Dict with status and results
        """
        result = {
            "ticker": self.ticker,
            "timestamp": datetime.now().isoformat(),
            "new_filings_checked": 0,
            "new_filings_downloaded": 0,
            "new_filings_processed": 0,
            "errors": [],
            "extractions": []
        }

        try:
            logger.info("Starting 10-K orchestration", extra={"ticker": self.ticker, "force_reprocess": force_reprocess, "call_model": call_model})

            logger.debug("Checking for new 10-K filings for %s", self.ticker)
            new_filings = self.filing_manager.check_new_filings()
            result["new_filings_checked"] = len(new_filings)
            logger.info("New filings check completed", extra={"ticker": self.ticker, "new_filings_count": len(new_filings)})

            if new_filings:
                logger.info("Processing %s new filing(s)", len(new_filings), extra={"ticker": self.ticker, "count": len(new_filings)})
                for filing in new_filings:
                    try:
                        success, filepath = self.filing_manager.download_filing(filing)
                        if success and filepath:
                            result["new_filings_downloaded"] += 1
                            logger.debug("Processing filing %s", filepath.name, extra={"ticker": self.ticker, "file": filepath.name})

                            extraction = self.processor.extract_from_file(filepath, call_model=call_model)

                            self.processor.save_extraction(extraction, self.extractions_dir)
                            result["new_filings_processed"] += 1

                            json_name = self._get_extraction_filename(filepath.name)

                            result["extractions"].append({
                                "file": filepath.name,
                                "extraction_path": str(self.extractions_dir / json_name),
                                "text_length": extraction.get("text_length", 0),
                                "has_signals": extraction.get("signals") is not None
                            })

                            self.filing_manager.mark_filing_processed(filing, status="processed")
                            logger.info("Successfully processed filing", extra={"ticker": self.ticker, "file": filepath.name, "text_length": extraction.get("text_length", 0)})
                        else:
                            error = f"Failed to download filing: {filing.get('url')}"
                            result["errors"].append(error)
                            logger.warning("Failed to download filing", extra={"ticker": self.ticker, "url": filing.get('url')})
                    except Exception as e:
                        error = f"Error processing filing: {str(e)}"
                        result["errors"].append(error)
                        logger.error("Error processing filing", extra={"ticker": self.ticker, "error": str(e)}, exc_info=True)
            else:
                logger.info("No new filings found", extra={"ticker": self.ticker})

            logger.debug("Loading cached extractions for %s", self.ticker)
            cached = self.load_all_extractions()
            result["total_cached_extractions"] = len(cached)

            logger.info(
                "10-K orchestration completed",
                extra={
                    "ticker": self.ticker,
                    "new_filings_checked": result["new_filings_checked"],
                    "new_filings_downloaded": result["new_filings_downloaded"],
                    "new_filings_processed": result["new_filings_processed"],
                    "total_cached": result["total_cached_extractions"],
                    "error_count": len(result["errors"])
                }
            )

        except Exception as e:
            error = f"Unexpected error in sync_and_process: {str(e)}"
            result["errors"].append(error)
            logger.error("Unexpected error in sync_and_process", extra={"ticker": self.ticker, "error": str(e)}, exc_info=True)

        return result

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

    def load_all_extractions(self) -> List[Dict[str, Any]]:
        """Load all cached 10-K extractions for this ticker.

        Returns:
            List of extraction dicts, sorted by date
        """
        if not self.extractions_dir.exists():
            return []

        extractions = []
        pattern = f"*{self.EXTRACTION_FILE_EXT}"

        for json_file in sorted(self.extractions_dir.glob(pattern)):
            try:
                with open(json_file, "r") as f:
                    extraction = json.load(f)
                    extractions.append(extraction)
            except Exception as e:
                logger.warning("Error loading %s: %s", json_file.name, e)

        return extractions

    def get_latest_extraction(self) -> Optional[Dict[str, Any]]:
        """Get the most recent 10-K extraction."""
        extractions = self.load_all_extractions()
        return extractions[-1] if extractions else None

    def get_extraction_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of all 10-K extractions with signals summary.

        Returns:
            List of dicts with extraction metadata and key signals
        """
        extractions = self.load_all_extractions()
        timeline = []

        for extraction in extractions:
            meta = extraction.get("meta", {})
            signals = extraction.get("signals", {})

            timeline_entry = {
                "file": meta.get("file"),
                "extracted_at": meta.get("extracted_at"),
                "text_length": extraction.get("text_length", 0),
                "tone": signals.get("tone_analysis", {}).get("sentiment_baseline", "unknown"),
                "credibility_score": signals.get("credibility_scorecard", {}).get("overall_score", None),
                "material_risks": signals.get("business_risks", {}).get("material_risks", []),
            }
            timeline.append(timeline_entry)

        return timeline

    def compute_temporal_trends(self) -> Dict[str, Any]:
        """Analyze trends across multiple 10-K filings.

        Returns:
            Dict with trend analysis
        """
        extractions = self.load_all_extractions()

        if not extractions:
            return {"error": "No extractions found"}

        trends = {
            "ticker": self.ticker,
            "total_filings": len(extractions),
            "tone_progression": [],
            "credibility_progression": [],
            "risk_evolution": {},
            "strategic_shifts": {}
        }

        for extraction in extractions:
            meta = extraction.get("meta", {})
            signals = extraction.get("signals", {})

            tone = signals.get("tone_analysis", {})
            trends["tone_progression"].append({
                "file": meta.get("file"),
                "sentiment": tone.get("sentiment_baseline"),
                "hedging": tone.get("hedging_intensity")
            })

            credibility = signals.get("credibility_scorecard", {})
            trends["credibility_progression"].append({
                "file": meta.get("file"),
                "score": credibility.get("overall_score"),
                "transparency": credibility.get("transparency_on_risks")
            })

            risks = signals.get("business_risks", {}).get("material_risks", [])
            for risk in risks:
                if risk not in trends["risk_evolution"]:
                    trends["risk_evolution"][risk] = []
                trends["risk_evolution"][risk].append(meta.get("file"))

            pillars = signals.get("strategic_narrative", {}).get("core_pillars", [])
            for pillar in pillars:
                if pillar not in trends["strategic_shifts"]:
                    trends["strategic_shifts"][pillar] = []
                trends["strategic_shifts"][pillar].append(meta.get("file"))

        return trends

    def format_timeline(self) -> str:
        """Format timeline of 10-K extractions as string.

        Returns:
            Formatted timeline string
        """
        timeline = self.get_extraction_timeline()

        if not timeline:
            return f"No 10-K extractions found for {self.ticker}"

        lines = [
            f"10-K EXTRACTION TIMELINE - {self.ticker}",
            "",
        ]

        for entry in timeline:
            lines.append(entry['file'])
            lines.append(f"  Extracted: {entry['extracted_at']}")
            lines.append(f"  Text Length: {entry['text_length']:,} characters")
            lines.append(f"  Tone: {entry['tone']}")
            lines.append(f"  Credibility Score: {entry['credibility_score']}/100")
            if entry["material_risks"]:
                lines.append(f"  Key Risks: {', '.join(entry['material_risks'][:3])}")
            lines.append("")

        return "\n".join(lines)

    def format_trends(self) -> str:
        """Format trend analysis as string.

        Returns:
            Formatted trends string
        """
        trends = self.compute_temporal_trends()

        if "error" in trends:
            return trends["error"]

        lines = [
            f"10-K TEMPORAL TREND ANALYSIS - {self.ticker}",
            "",
            f"Total 10-K Filings Analyzed: {trends['total_filings']}",
            "",
            "[TONE PROGRESSION]",
        ]

        for item in trends["tone_progression"]:
            sentiment = item["sentiment"] or "unknown"
            lines.append(f"  {item['file']:40} | {sentiment:12} | Hedging: {item['hedging']}")

        lines.append("")
        lines.append("[CREDIBILITY PROGRESSION]")
        for item in trends["credibility_progression"]:
            score = item["score"] if item["score"] else "?"
            lines.append(f"  {item['file']:40} | Score: {score:>3}/100 | {item['transparency']}")

        lines.append("")
        lines.append("[RISK EVOLUTION]")
        for risk, filings in sorted(trends["risk_evolution"].items()):
            lines.append(f"  {risk:50} | {', '.join(filings)}")

        lines.append("")
        lines.append("[STRATEGIC PILLARS]")
        for pillar, filings in sorted(trends["strategic_shifts"].items()):
            lines.append(f"  {pillar:50} | {', '.join(filings)}")

        return "\n".join(lines)

    def clear_cache(self) -> None:
        """Clear cached components and delegate to underlying managers."""
        if self._filing_manager is not None:
            try:
                self._filing_manager.clear_cache()
            except Exception:
                pass
        self._filing_manager = None
        self._processor = None
        logger.debug("Cleared TenKOrchestrator caches for %s", self.ticker)