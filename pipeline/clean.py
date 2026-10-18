"""
pipeline.clean
--------------
Phase 1, Step 2 — Data cleaning for the Skytrax airline reviews.

What this stage does
--------------------
1.  Rating columns:  "n" / "N/A" / "" placeholders  ->  NaN, then numeric.
2.  Deduplicate on (author, date, normalized content); drop empty-content rows.
3.  Date parsing -> datetime, year, month.
4.  Three levels of text cleaning are stored per review:
      * content            - untouched original text
      * content_clean      - HTML entities/tags/URLs removed, whitespace fixed
      * content_processed  - lowercased, contractions expanded, tokenized,
                             stopwords removed, lemmatized, negation-marked
                             (e.g. "not_good") - this is what Phase 2 feasts on.
5.  Cleaning statistics are recorded per step -> data/cleaning_stats.json

Outputs: data/clean_reviews.csv, data/cleaning_stats.json
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .common import (
    ASPECT_RATING_COLS,
    CLEANING_STATS_JSON,
    CLEAN_CSV,
    RAW_CSV,
    logger,
    save_csv,
    save_json,
)
from .fetch_data import fetch

# --------------------------------------------------------------------------
# Regexes & lexicons
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTMLTAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z']+")
_WS_RE = re.compile(r"\s+")

NEGATION_WORDS = {"not", "no", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere"}
NEGATION_WINDOW = 3  # negate the next N tokens after a negation trigger

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not", "ain't": "is not",
    "let's": "let us", "gonna": "going to", "wanna": "want to", "gotta": "got to",
}
_SUFFIX_RULES = [  # applied in order: (regex, replacement)
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'s\b"), ""),  # ambiguous possessive/contraction -> drop
]

_STOPWORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


def _get_stopwords() -> set[str]:
    """Lazy-load stopwords, minus the negation words we must keep."""
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS
    return _STOPWORDS


def _get_lemmatizer() -> WordNetLemmatizer:
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


# --------------------------------------------------------------------------
# Text cleaning levels
# --------------------------------------------------------------------------
def basic_clean(text: str) -> str:
    """Level 2: decode HTML entities, strip tags/URLs, fix whitespace."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = _HTMLTAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("'", "'").replace("\xa0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def expand_contractions(text: str) -> str:
    lowered = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        lowered = lowered.replace(contraction, expansion)
    for pattern, replacement in _SUFFIX_RULES:
        lowered = pattern.sub(replacement, lowered)
    return lowered


def nlp_process(text: str) -> str:
    """Level 3: full pipeline -> lemmatized, negation-marked token string."""
    stops = _get_stopwords()
    lemmatizer = _get_lemmatizer()

    tokens = _TOKEN_RE.findall(expand_contractions(text))

    # stopword removal (negation triggers survive; they are consumed below)
    tokens = [t for t in tokens if len(t) >= 2 and t not in stops and not t.isdigit()]

    # lemmatize (verb sense first, then noun)
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # negation window: after a trigger, prefix the next N tokens with not_
    out: list[str] = []
    neg_remaining = 0
    for tok in tokens:
        if tok in NEGATION_WORDS:
            neg_remaining = NEGATION_WINDOW
            continue  # trigger itself is dropped; its effect lives in the prefixes
        if neg_remaining > 0:
            out.append(f"not_{tok}")
            neg_remaining -= 1
        else:
            out.append(tok)
    return " ".join(out)


# --------------------------------------------------------------------------
# Main cleaning routine
# --------------------------------------------------------------------------
PLACEHOLDERS = {"", "n", "n/a", "na", "nan", "-", "none"}

def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"rows_in": int(len(raw)), "steps": {}}
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # ---- 1. Ratings: placeholders -> NaN, then numeric -------------------
    rating_cols = [c for c in ASPECT_RATING_COLS + ["overall_rating"] if c in df.columns]
    placeholder_hits = 0
    for col in rating_cols:
        s = df[col].astype(str).str.strip().str.lower()
        placeholder_hits += int(s.isin(PLACEHOLDERS).sum())
        df[col] = pd.to_numeric(s, errors="coerce")
    stats["steps"]["rating_placeholders_to_nan"] = placeholder_hits

    # ---- 2. recommended: normalise + binary label ------------------------
    # NB: the source stores this as "1"/"0" in some copies and "yes"/"no" in
    # others, so accept every known encoding.
    if "recommended" in df.columns:
        df["recommended"] = df["recommended"].astype(str).str.strip().str.lower()
        rec_map = {"yes": 1, "no": 0, "1": 1, "0": 0,
                   "true": 1, "false": 0, "y": 1, "n": 0}
        df["recommended_binary"] = df["recommended"].map(rec_map)
        stats["steps"]["recommended_value_counts"] = (
            df["recommended"].value_counts(dropna=False).to_dict()
        )
        stats["steps"]["recommended_unmapped"] = int(df["recommended_binary"].isna().sum())

    # ---- 3. Dates ---------------------------------------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        stats["steps"]["date_parse_failures"] = int(df["date"].isna().sum())
        # Guard against epoch sentinels / typos (source reviews start in 2002).
        implausible = df["date"].notna() & (df["date"] < pd.Timestamp("2000-01-01"))
        stats["steps"]["implausible_dates_to_nat"] = int(implausible.sum())
        df.loc[implausible, "date"] = pd.NaT
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df.loc[df["date"].isna(), "month"] = None

    # ---- 4. Drop empty-content & duplicate rows ---------------------------
    if "content" in df.columns:
        df["content"] = df["content"].astype(str)
        non_empty = df["content"].str.strip().ne("") & df["content"].ne("nan")
        stats["steps"]["dropped_empty_content"] = int((~non_empty).sum())
        df = df[non_empty]

        dup_mask = df.duplicated(subset=["author", "date", "content"], keep="first")
        stats["steps"]["dropped_duplicates"] = int(dup_mask.sum())
        df = df[~dup_mask]

    # ---- 5. Text cleaning (3 levels) --------------------------------------
    logger.info("Cleaning text for %d reviews ...", len(df))
    df["content_clean"] = df["content"].map(basic_clean)
    df["content_processed"] = df["content_clean"].map(nlp_process)

    # drop reviews left with zero usable tokens
    empty_proc = df["content_processed"].str.strip().eq("")
    stats["steps"]["dropped_no_tokens_after_processing"] = int(empty_proc.sum())
    df = df[~empty_proc]

    # ---- 6. Final tidy ------------------------------------------------------
    if "author_country" in df.columns:
        df["author_country"] = df["author_country"].replace({"": None}).str.strip().str.title()
    df = df.reset_index(drop=True)
    stats["rows_out"] = int(len(df))
    stats["rows_dropped_total"] = stats["rows_in"] - stats["rows_out"]
    return df, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean the raw Skytrax reviews.")
    parser.add_argument("--force-fetch", action="store_true", help="Re-download raw data first.")
    args = parser.parse_args(argv)

    if not RAW_CSV.exists() or args.force_fetch:
        fetch(force=args.force_fetch)

    raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)
    cleaned, stats = clean(raw)

    save_csv(cleaned, CLEAN_CSV)
    save_json(stats, CLEANING_STATS_JSON)

    logger.info("Cleaning stats: %s in -> %s out", stats["rows_in"], stats["rows_out"])
    logger.info("Steps: %s", stats["steps"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
