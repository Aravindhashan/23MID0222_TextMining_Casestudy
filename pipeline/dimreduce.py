"""
pipeline.dimreduce
--------------------
Phase 2, Step 2 — Dimensionality reduction + topic modeling.

1. LSA / TruncatedSVD on TF-IDF -> (scree data, top terms per component)
2. LDA topic modeling on BoW counts, grid search n_components in [6..10]
   by perplexity (lower is better). Pick best. Topic labels heuristically
   assigned from candidate aspect labels via keyword overlap; fall back to
   "Topic {k}: top-term". Persist top terms per topic.
3. t-SNE on SVD-reduced (20d) TF-IDF, ≤20k subsample, perplexity 30.
   (PCA fallback if subsample small)
4. PCA 2D for additional map.

Outputs:
   data/features/lsa.pkl        (SVD model + transformed matrix + scree)
   models/lda_model.pkl + vec
   data/tsne_coords.csv
   data/topic_terms.json
   data/topic_mix.csv
   data/lda_perplexity.json
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

from .common import *
from .features import load_clean, build_tfidf, VADER_OK


def _choose_svd(n_features, desired=20):
    return min(desired, n_features - 1)


CANDIDATE_TOPIC_LABELS = [
    ("crew_service", ["crew", "flight", "service", "attendant", "staff", "friendly",
                    "cabin", "crew"]),
    ("delays_baggage", ["delay", "delayed", "baggage", "bag", "luggage", "cancel",
                        "gate", "ontime", "late"]),
    ("food_drinks", ["food", "meal", "drink", "breakfast", "snack", "dinner",
                   "lunch", "beverage"]),
    ("seats_comfort", ["seat", "comfort", "legroom", "leg", "recline", "space"]),
    ("entertainment", ["entertainment", "movie", "screen", "wifi", "inflight"]),
    ("ground_service", ["ground", "check", "boarding", "airport", "lounge",
                      "counter"]),
    ("value_money", ["value", "price", "money", "cheap", "expensive", "fare"]),
    ("aircraft_cleanliness", ["aircraft", "clean", "dirty", "condition", "plane"]),
]


def label_topic(top_terms):
    term_set = set(t for t in top_terms)
    best = None
    for name, seeds in CANDIDATE_TOPIC_LABELS:
        hits = len(term_set & set(seeds))
        if best is None or hits > best[0]:
            best = (hits, name)
    if best and best[0] >= 2:
        name = best[1]
    else:
        name = "Topic: " + ", ".join(top_terms[:3])
    return name


def run(force: bool = False):
    ensure_dirs()
    df = load_clean()

    # Load DTM produced by Phase 2 Step 1
    try:
        vect, matrix = joblib.load(DTM_PKL)
    except Exception as e:
        logger.error("Could not load DTM; run features first: %s", e)
        raise

    # --- 1. LSA / SVD ---
    k = _choose_svd(matrix.shape[1], 20)
    svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=7,
                      random_state=42)
    X_svd = svd.fit_transform(matrix)
    explained = svd.explained_variance_ratio_
    cumulative = np.cumsum(explained).tolist()
    feat_names = np.asarray(vect.get_feature_names_out())
    # top terms per component
    comp_terms = {}
    for i in range(min(k, 10)):
        comp = svd.components_[i]
        top_idx = np.argsort(np.abs(comp))[::-1][:12]
        comp_terms[f"component_{i}"] = [
            {"term": str(feat_names[j]), "weight": round(float(comp[j]), 5)}
            for j in top_idx
        ]
    joblib.dump(
        {"model": svd, "matrix": X_svd, "explained_variance_ratio": explained.tolist(),
         "cumulative": cumulative, "component_terms": comp_terms},
        LSA_PKL,
    )
    logger.info("LSA: explained var %.2f%% in %d comps",
                cumulative[-1] * 100, k)

    # --- 2. LDA topics ---
    # BoW counts for LDA
    bow_vect = CountVectorizer(min_df=5, max_df=0.95, ngram_range=(1, 2),
                             max_features=5000)
    bow = bow_vect.fit_transform(df["content_processed"])
    bow = bow_vect.fit_transform(df["content_processed"])

    # subsample for grid search (perplexity), fit final on all data
    if len(df) > 20000:
        grid_idx = np.random.default_rng(42).choice(len(df), size=20000, replace=False)
        bow_grid = bow[grid_idx]
    else:
        bow_grid = bow

    perp_results = {}
    best = None
    n_grid = [6, 8, 10]
    for n in n_grid:
        try:
            lda = LatentDirichletAllocation(
                n_components=n, learning_method="batch", random_state=42,
                max_iter=10,
            )
            lda.fit(bow_grid)
            perp = float(lda.perplexity(bow_grid))
            perp_results[n] = perp
            logger.info("LDA n=%d perplexity=%.2f", n, perp)
            if best is None or perp < best[1]:
                best = (n, perp)
        except Exception as e:
            logger.warning("LDA n=%d failed: %s", n, e)
    n_best = best[0] if best else 8

    lda_final = LatentDirichletAllocation(
        n_components=n_best, learning_method="batch", random_state=42, max_iter=15
    )
    lda_final.fit(bow)
    joblib.dump({"model": lda_final, "vectorizer": bow_vect}, LDA_MODEL_PKL)

    vocab = np.asarray(bow_vect.get_feature_names_out())
    topic_terms = {}
    topic_labels = []
    # top 15 terms per topic
    for topic_idx, comp in enumerate(lda_final.components_):
        top_idx = np.argsort(comp)[::-1][:15]
        terms = [str(vocab[j]) for j in top_idx]
        label = label_topic(terms[:8])
        topic_terms[str(topic_idx)] = {
            "label": label,
            "terms": terms,
            "weights": [round(float(comp[j]), 4) for j in top_idx],
        }
        topic_labels.append(label)
    save_json(topic_terms, TOPIC_TERMS_JSON)

    # topic mix per doc
    topic_mix = lda_final.transform(bow)
    top_topic = np.argmax(topic_mix, axis=1)
    mix_df = pd.DataFrame({
        "dominant_topic": top_topic,
    })
    mix_df.to_csv(TOPIC_MIX_CSV, index=False)
    # topic distribution
    dist = pd.Series(top_topic).value_counts().sort_index().tolist()
    perp_json = {
        "perplexities": {str(k): round(v, 2) for k, v in perp_results.items()},
        "selected_n_components": int(n_best),
        "best_perplexity": round(best[1], 2) if best else None,
        "topic_distribution": dist,
    }
    save_json(perp_json, LDA_PERPLEXITY_JSON)

    # --- 3. t-SNE ---
    n_samples = min(len(df), 15000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=n_samples, replace=False)
    X_in = X_svd[idx, :]
    if X_in.shape[1] > 50:
        X_in = X_in[:, :50]
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=42, max_iter=500, learning_rate="auto")
    coords = tsne.fit_transform(X_in)
    tsne_df = pd.DataFrame({
        "review_idx": idx,
        "tsne_1": coords[:, 0],
        "tsne_2": coords[:, 1],
        "recommended": df["recommended"].values[idx],
        "recommended_binary": df["recommended_binary"].values[idx],
        "overall_rating": df["overall_rating"].values[idx],
        "cabin_flown": df["cabin_flown"].astype(str).values[idx],
        "airline_name": df["airline_name"].values[idx],
        "dominant_topic": top_topic[idx],
    })
    tsne_df.to_csv(TSNE_CSV, index=False)
    logger.info("t-SNE done: %d points", n_samples)

    # --- 4. PCA 2D (whole dataset via SVD scores) ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_svd)
    joblib.dump({"pca": pca, "coords": X_pca}, FEATURES_DIR / "pca2.pkl")

    logger.info("Phase 2 dimensionality reduction complete.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Step 2: dim reduction + topics.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv if argv is not None else [])
    return run(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
