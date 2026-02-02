"""
Run a single end-to-end thesis validation pipeline for a sample thesis (SNOW)
and save the full `FinalThesisReport` to a JSON file.

This script will attempt to call the live agent implementations (LLMs). If the
environment is not configured (API keys missing), the run may fail with an
exception; the exception will be written to `run_error.log` for debugging.

Usage:
    python scripts/run_thesis_example.py
"""
import os
import sys
import json
import textwrap
from datetime import datetime
from dataclasses import is_dataclass, asdict

from model.orchestration.thesis_validation_orchestrator import ThesisValidationOrchestrator

OUT_DIR = ""
OUT_PATH = os.path.join(OUT_DIR, "snowflake_thesis_run.json")
ERR_PATH = os.path.join(OUT_DIR, "snowflake_thesis_run_error.log")

# Example thesis narrative for SNOW (you can customize)
THESIS = textwrap.dedent('''
    Snowflake (SNOW) will continue to outgrow the cloud data market as customers
    consolidate data workloads onto its platform, driving 20%+ revenue CAGR and
    margin expansion from scale and product mix. Management's execution
    roadmap and the stickiness of platform adoption will sustain above-market
    growth for the next 5 years.
''').strip()

COMPANY_CONTEXT = textwrap.dedent('''
    Snowflake Inc. operates a cloud-native data platform for data engineering,
    data lakes, data warehousing, data application development, and secure data
    sharing. Large enterprise customers primarily drive revenue with consumption-based pricing.
''').strip()

def dataclass_to_dict(obj):
    """Recursively convert dataclasses to dicts for JSON serialization."""
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def main():
    print("Starting thesis validation run for SNOW — this may call LLMs and take a minute or two.")
    try:
        runner = ThesisValidationOrchestrator(stock_ticker="SNOW")
        report = runner.run(
            thesis_narrative=THESIS,
            company_context=COMPANY_CONTEXT,
            stated_conviction=0.65,
            evidence_items=[],
            current_regime="Transitioning",
            quantitative_context=None,
        )

        serializable = dataclass_to_dict(report)

        with open(OUT_PATH, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"Saved run output to {OUT_PATH}")
    except Exception as e:
        print("Run failed:", e)
        with open(ERR_PATH, "w") as ef:
            ef.write(str(e))
        print(f"Wrote error to {ERR_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
