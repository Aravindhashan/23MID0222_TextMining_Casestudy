"""
pipeline.features
------------------
Phase 2, Step 1 — Feature extraction.

Stages:
1. TF-IDF (unigrams + bigrams) on content_processed -> DTM + vectorizer
2. Collocation/phrase mining (top frequency + top PMI bigrams)
3. Engineered per-review features:
   length_chars, word_count, exclamation_count, caps_ratio,
   vader_compound (lexicon baseline)
4. Top-20 terms (positive vs negative), correlations with overall_rating

Outputs:
   data/features/dtm.pkl                 (vectorizer + TF-IDF sparse matrix)
   data/features/engineered.csv
   data/features/features_meta.json
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text as sk_text

from .common import (
    DTM_PKL,
    ENGINEERED_CSV,
    FEATURES_DIR,
    FEATURES_META_JSON,
    logger,
    save_json,
    ensure_dirs,
    CLEAN_CSV,
)


def load_clean():
    from .common import load_csv

    df = load_csv(CLEAN_CSV)
    df = df.dropna(subset=["content_processed"]).reset_index(drop=True)
    df["content_processed"] = df["content_processed"].astype(str)
    df["content_clean"] = df["content_clean"].astype(str)
    df["content"] = df["content"].astype(str)
    return df


def build_tfidf(df):
    vect = TfidfVectorizer(
        min_df=5, max_df=0.95, ngram_range=(1, 2),
        max_features=8000, sublinear_tf=True, stop_words=None,
    )
    matrix = vect.fit_transform(df["content_processed"])
    logger.info("TF-IDF shape: %s (vocab=%d)", matrix.shape, len(vect.vocabulary_))
    return vect, matrix


VADER_OK = False
try:
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    VADER_OK = True
except Exception as e:  # pragma: no cover
    logger.warning("VADER unavailable, sentiment feature will be 0: %s", e)


def top_terms_per_prefix(matrix, vect, labels, topn=20):
    features = np.asarray(vect.get_feature_names_out())
    out = {}
    labels = np.asarray(labels)
    for tag, lbl in [("positive", 1), ("negative", 0)]:
        mask = labels == lbl
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        group_w = np.asarray(matrix[mask].mean(axis=0)).ravel()
        other_w = np.asarray(matrix[~mask].mean(axis=0)).ravel()
        diff = group_w - other_w
        order = np.argsort(diff)[::-1][:topn]
        out[tag] = [
            {"term": str(features[i]), "weight": round(float(diff[i]), 5)}
            for i in order
        ]
    return out


def collocations(df, topn=30):
    toks = []
    for txt in df["content_processed"].head(5000):
        toks.extend(t for t in txt.split() if len(t) >= 3)
    finder = BigramCollocationFinder.from_words(toks)
    finder.apply_freq_filter(20)
    bigram_measures = BigramAssocMeasures()
    ranked = sorted(
        finder.score_ngrams(bigram_measures.pmi)[:topn * 2],
        key=lambda x: (-x[1], -finder.ngram_fd[x[0]]),
    )
    pmi_list = [
        {"bigram": " ".join(bg), "pmi": round(float(score), 4),
         "freq": int(finder.ngram_fd[bg])}
        for (bg, score) in ranked[:topn]
        if finder.ngram_fd[bg] >= 20
    ]
    freq_rank = finder.ngram_fd.most_common(topn * 2)
    freq_list = [{"bigram": " ".join(bg), "freq": int(n)} for bg, n in freq_rank[:topn]]
    return {"pmi": pmi_list, "frequent": freq_list}


def engineered_features(df):
    sia = SentimentIntensityAnalyzer() if VADER_OK else None

    rows = []
    for _, r in df.iterrows():
        content = str(r["content"])
        clean = str(r["content_clean"])
        proc = str(r["content_processed"])
        words = proc.split()
        alpha_chars = [c for c in clean if c.isalpha()]
        excl = content.count("!")
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
        polarity = float(sia.polarity_scores(content)["compound"]) if sia else 0.0
        rows.append({
            "length_chars": int(len(content)),
            "word_count": int(len(words)),
            "exclamation_count": int(excl),
            "caps_ratio": round(float(caps_ratio), 4),
            "vader_compound": round(float(polarity), 4),
        })
    eng = pd.DataFrame(rows)
    out = df[["airline_name", "year", "cabin_flown", "recommended_binary",
              "overall_rating"]].reset_index(drop=True)
    out = pd.concat([eng.reset_index(drop=True), out], axis=1)
    return out


def run(force: bool = False):
    ensure_dirs()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    # 1. TF-IDF
    vect, matrix = build_tfidf(df)
    labels = df["recommended_binary"].fillna(-1).astype(int).values
    labels = np.where(labels == 1, 1, 0)
    collocations_top = collocations(df)
    top_terms = top_terms_per_prefix(matrix, vect, labels, topn=20)

    joblib.dump((vect, matrix), DTM_PKL)
    logger.info("Saved DTM -> %s", DTM_PKL)

    # 2. engineered
    eng = engineered_features(df)
    eng.to_csv(ENGINEERED_CSV, index=False)
    logger.info("Saved engineered -> %s (%d rows)", ENGINEERED_CSV, len(eng))

    # 3. correlations
    ccols = ["length_chars", "word_count", "exclamation_count", "caps_ratio",
             "vader_compound"]
    corr_with_rating = {}
    joined = eng.assign(overall_rating=df["overall_rating"].values)
    for cc in ccols:
        try:
            corr_with_rating[cc] = round(float(joined[cc].corr(joined["overall_rating"])), 3)
        except Exception:
            corr_with_rating[cc] = None

    meta = {
        "n_reviews": int(len(df)),
        "n_features": int(matrix.shape[1]),
        "vocab_size": int(len(vect.vocabulary_)),
        "tfidf_shape": list(matrix.shape),
        "tfidf_params": {"min_df": 5, "max_df": 0.95, "ngram_range": [1, 2],
                         "max_features": 8000, "sublinear_tf": True},
        "top_terms": top_terms,
        "top_collocations": collocations_top,
        "engineered_correlations_with_overall_rating": corr_with_rating,
    }
    save_json(meta, FEATURES_META_JSON)
    logger.info("Saved features_meta -> %s", FEATURES_META_JSON)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Step 1: feature extraction.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv if argv is not None else [])
    return run(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
