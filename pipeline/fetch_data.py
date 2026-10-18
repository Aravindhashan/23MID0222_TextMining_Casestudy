"""
pipeline.fetch_data
-------------------
Phase 1, Step 1 — Download the Skytrax airline reviews CSV (static dataset,
no API key) into data/raw_airline_reviews.csv, with a light profile printout.

Re-running is safe: an existing raw CSV is reused unless --force is given.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import requests

from .common import RAW_CSV, SOURCE_URL, logger, ensure_dirs


def download(url: str = SOURCE_URL, timeout: int = 60, retries: int = 3) -> bytes:
    """Retry-safe download of the source CSV bytes."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading (attempt %d/%d): %s", attempt, retries, url)
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_exc = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
    raise RuntimeError(f"Download failed after {retries} attempts: {last_exc}")


def fetch(force: bool = False) -> pd.DataFrame:
    """Return the raw DataFrame, downloading it if necessary."""
    ensure_dirs()
    if RAW_CSV.exists() and not force:
        logger.info("Raw CSV already present (%s) - reusing. Use --force to refetch.", RAW_CSV)
        return pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)

    content = download()
    RAW_CSV.write_bytes(content)
    logger.info("Saved raw dataset -> %s (%.1f MB)", RAW_CSV, len(content) / 1048576)

    return pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)


def profile(df: pd.DataFrame) -> None:
    """Print a quick sanity profile of the raw dataset."""
    logger.info("Shape: %s rows x %s cols", *df.shape)
    logger.info("Columns: %s", list(df.columns))
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        logger.info("Review dates span: %s -> %s (unparseable: %d)",
                    dates.min().date(), dates.max().date(), dates.isna().sum())
    if "airline_name" in df.columns:
        top = df["airline_name"].value_counts().head(8)
        logger.info("Top airlines by review count:\n%s", top.to_string())
    if "recommended" in df.columns:
        logger.info("recommended value counts:\n%s",
                    df["recommended"].value_counts(dropna=False).to_string())
    sample = df["content"].iloc[0] if "content" in df.columns else ""
    logger.info("First review snippet: %r", str(sample)[:160])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch raw Skytrax airline reviews.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args(argv)

    df = fetch(force=args.force)
    profile(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
