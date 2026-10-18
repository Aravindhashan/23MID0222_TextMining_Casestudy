"""
pipeline.insights
-----------------
Phase 4 - advanced insights:
  - time trends (yearly sentiment / recommend-rate + event annotations)
  - pre/post-COVID topic shifts
  - airline clustering (KMeans + Ward-ish hierarchical) on aspect profiles
  - Passenger Satisfaction Index (PSI) ranking
  - NPS-style promoters/detractors per airline
  - aspect co-occurrence network
  - cabin vs country breakdowns
  - semantic word embeddings (co-occurrence SVD) with "most similar" support

Outputs:
   data/trends.csv
   data/airline_clusters.csv
   data/psi_ranking.csv
   data/nps.csv
   data/aspect_network.json
   data/cabin_country.csv
   data/country_stats.json
   models/word2vec.pkl (SVD embeddings; kept name for dashboard compat)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from .common import (
    ASPECT_NETWORK_JSON,
    AIRLINE_CLUSTERS_CSV,
    CABIN_COUNTRY_CSV,
    FEATURES_DIR,
    MODELS_DIR,
    PSI_RANKING_CSV,
    TRENDS_CSV,
    NPS_CSV,
    WORD2VEC_PKL,
    logger,
    save_json,
    ensure_dirs,
    CLEAN_CSV,
    DATA_DIR,
)
from .features import load_clean

ASPECTS = {
    "seat_comfort": {"seeds": ["seat", "legroom", "comfort", "recline", "space"],
                     "rating_col": "seat_comfort_rating", "label": "Seat comfort"},
    "cabin_staff": {"seeds": ["crew", "staff", "attendant", "service", "hostess", "steward"],
                    "rating_col": "cabin_staff_rating", "label": "Cabin staff"},
    "food_beverages": {"seeds": ["food", "meal", "snack", "drink", "beverage", "catering", "dinner"],
                       "rating_col": "food_beverages_rating", "label": "Food & beverages"},
    "inflight_entertainment": {"seeds": ["entertainment", "movie", "screen", "tv", "wifi", "monitor"],
                               "rating_col": "inflight_entertainment_rating", "label": "Inflight entertainment"},
    "ground_service": {"seeds": ["ground", "boarding", "gate", "counter", "airport", "lounge"],
                       "rating_col": "ground_service_rating", "label": "Ground service"},
    "wifi_connectivity": {"seeds": ["wifi", "internet", "connectivity", "connection", "signal"],
                          "rating_col": "wifi_connectivity_rating", "label": "WiFi connectivity"},
    "value_money": {"seeds": ["value", "money", "price", "cheap", "expensive", "fare", "worth"],
                    "rating_col": "value_money_rating", "label": "Value for money"},
}
ASPECT_COLS = [v["rating_col"] for v in ASPECTS.values()]
ASPECT_NAMES = list(ASPECTS.keys())


def _vader_series(df):
    sia = SentimentIntensityAnalyzer()
    return df["content"].fillna("").astype(str).map(lambda t: sia.polarity_scores(t)["compound"])


def run(force=False):
    ensure_dirs()
    df = load_clean()
    if "vader_compound" not in df.columns:
        logger.info("Computing VADER sentiment for all reviews ...")
        df["vader_compound"] = _vader_series(df)
    # t-SNE coords reference maybe; for trends need LDA dominant per review.
    # Load LDA bundle (built in dimreduce) for topic distribution.
    lda_bundle = None
    try:
        import joblib as _jb
        lda_bundle = _jb.load(__import__("pipeline.common", fromlist=["LDA_MODEL_PKL"]).LDA_MODEL_PKL)
    except Exception as e:
        logger.warning("LDA bundle unavailable: %s", e)
        lda_bundle = None

    # ---------------- 1. Trends + topic shifts ----------------
    if lda_bundle is not None:
        bow_vect = lda_bundle["vectorizer"]
        lda = lda_bundle["model"]
        bow = bow_vect.transform(df["content_processed"].astype(str))
        probs = lda.transform(bow)
        df["dominant_topic"] = np.argmax(probs, axis=1)
        topic_cols = [f"topic_{i}_share" for i in range(probs.shape[1])]
        for i in range(probs.shape[1]):
            df[f"topic_{i}_share_tmp"] = probs[:, i]
    else:
        df["dominant_topic"] = 0

    agg = {}
    g = df.groupby("year", dropna=True)
    yearly = pd.DataFrame({
        "reviews": g.size(),
        "recommend_rate": g["recommended_binary"].mean(),
        "overall_rating": g["overall_rating"].mean(),
        "vader_mean": g["vader_compound"].mean(),
    }).reset_index()
    for i in range(probs.shape[1]) if lda_bundle is not None else []:
        yearly[f"topic_{i}_share"] = df.groupby("year")[f"topic_{i}_share_tmp"].mean().values
    # clean tmp
    for c in [c for c in df.columns if c.endswith("_share_tmp")]:
        del df[c]
    yearly.to_csv(TRENDS_CSV, index=False)
    logger.info("Saved trends -> %s", TRENDS_CSV)

    # pre/post COVID topic mix
    covid_rows = []
    for era_name, mask in [
        ("pre_covid", df["year"] <= 2019),
        ("covid_era", (df["year"] >= 2020) & (df["year"] <= 2021)),
        ("post_covid", df["year"] >= 2022),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        topic_means = []
        if lda_bundle is not None:
            p = lda.transform(bow_vect.transform(sub["content_processed"].astype(str)))
            for i in range(p.shape[1]):
                topic_means.append(float(p[:, i].mean()))
        row = {"era": era_name, "reviews": int(len(sub)),
               "recommend_rate": float(sub["recommended_binary"].mean()),
               "mean_overall_rating": float(sub["overall_rating"].mean()),
               "mean_vader": float(sub["vader_compound"].mean())}
        for i, v in enumerate(topic_means):
            row[f"topic_{i}"] = round(v, 4)
        covid_rows.append(row)
    covid_df = pd.DataFrame(covid_rows)
    covid_df.to_json(DATA_DIR / "covid_topics.json", orient="records", indent=2)
    logger.info("Saved covid era summary -> %s", DATA_DIR / "covid_topics.json")

    # ---------------- 2. Cabin & country breakdowns ----------------
    cabin = df.groupby("cabin_flown", dropna=True).agg(
        reviews=("recommended_binary", "size"),
        recommend_rate=("recommended_binary", "mean"),
        overall_rating=("overall_rating", "mean"),
        vader_mean=("vader_compound", "mean"),
    ).reset_index()
    for a in ASPECT_NAMES:
        col = ASPECTS[a]["rating_col"]
        if col in df.columns:
            cabin[a + "_stars"] = df.groupby("cabin_flown", dropna=True)[col].mean().values
    cabin.to_csv(CABIN_COUNTRY_CSV, index=False)
    # country stats (top 12)
    if "author_country" in df.columns:
        top_countries = df["author_country"].value_counts().head(12).index
        country_rows = []
        for c in top_countries:
            sub = df[df["author_country"] == c]
            country_rows.append({
                "country": c, "reviews": int(len(sub)),
                "recommend_rate": float(sub["recommended_binary"].mean()),
                "overall_rating": float(sub["overall_rating"].mean()),
                "vader_mean": float(sub["vader_compound"].mean()),
            })
        import json as _j
        with open(DATA_DIR / "country_stats.json", "w") as fh:
            _j.dump(country_rows, fh, indent=2, default=str)

    # ---------------- 3. Airline clustering ----------------
    profile_rows = []
    airlines_map = {}
    for airline, g in df.groupby("airline_name"):
        if len(g) < 15:
            continue
        rec = {
            "airline_name": airline,
            "reviews": int(len(g)),
        }
        for a in ASPECT_NAMES:
            col = ASPECTS[a]["rating_col"]
            rec[a] = float(g[col].dropna().mean()) if (col in g and g[col].notna().any()) else np.nan
        rec["overall_rating"] = float(g["overall_rating"].mean())
        rec["vader_mean"] = float(g["vader_compound"].mean())
        rec["recommend_rate"] = float(g["recommended_binary"].mean())
        profile_rows.append(rec)
    prof = pd.DataFrame(profile_rows)
    # impute missing aspect means with column mean
    for a in ASPECT_NAMES:
        if a in prof.columns:
            prof[a] = prof[a].fillna(prof[a].mean())
    prof["overall_rating"] = prof["overall_rating"].fillna(prof["overall_rating"].mean())
    prof["vader_mean"] = prof["vader_mean"].fillna(prof["vader_mean"].mean())
    prof["recommend_rate"] = prof["recommend_rate"].fillna(prof["recommend_rate"].mean())

    feat_cols = ASPECT_NAMES + ["overall_rating", "vader_mean", "recommend_rate"]
    Xp = prof[feat_cols].values
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(Xp)
    # best k via silhouette
    best_k = 3
    best_sil = -999
    sils = {}
    maxk = min(6, len(prof) - 1)
    for k in range(2, maxk + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(Xs)
            sil = silhouette_score(Xs, labels)
            sils[k] = float(sil)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception as e:
            logger.warning("KMeans k=%d failed: %s", k, e)
    logger.info("Best k=%d (sil=%.3f)", best_k, best_sil)
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km_final.fit_predict(Xs)
    # agglomerative clusters
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    try:
        agg_labels = agg.fit_predict(Xs)
    except Exception:
        agg_labels = km_labels
    prof["cluster"] = km_labels
    # name clusters by quality tier & top scoring aspect
    cluster_names = {}
    for cl in range(best_k):
        sub = prof[prof["cluster"] == cl]
        if sub.empty:
            continue
        if float(sub["overall_rating"].mean()) >= 7:
            tier = "High satisfaction"
        elif float(sub["overall_rating"].mean()) >= 5:
            tier = "Mid-tier"
        else:
            tier = "Low satisfaction"
        # top aspect vs overall baseline
        best_aspect = max(ASPECT_NAMES, key=lambda a: float(sub[a].mean()))
        cluster_names[cl] = f"{tier} ({ASPECTS[best_aspect]['label']})"
    prof["cluster_label"] = prof["cluster"].map(cluster_names)
    prof.to_csv(AIRLINE_CLUSTERS_CSV, index=False)
    save_json({"best_k": int(best_k), "silhouettes": sils, "cluster_names": {str(k): v for k, v in cluster_names.items()}},
              DATA_DIR / "cluster_meta.json")
    logger.info("Saved clusters -> %s", AIRLINE_CLUSTERS_CSV)

    # ---------------- 4. NPS per airline ----------------
    nps_rows = []
    for airline, g in df.groupby("airline_name"):
        if len(g) < 15:
            continue
        promoters = int(((g["recommended_binary"] == 1) & (g["overall_rating"] >= 7)).sum())
        detractors = int(((g["recommended_binary"] == 0) & (g["overall_rating"] <= 4)).sum())
        total = int(len(g))
        nps = (promoters - detractors) / total * 100
        nps_rows.append({
            "airline_name": airline, "reviews": total,
            "promoters": promoters, "detractors": detractors,
            "passives": total - promoters - detractors,
            "nps": round(float(nps), 2),
            "vader_mean": float(g["vader_compound"].mean()),
        })
    nps_df = pd.DataFrame(nps_rows).sort_values("nps", ascending=False).reset_index(drop=True)
    nps_df.to_csv(NPS_CSV, index=False)
    # corr NPS vs vader
    corr_nps = float(nps_df["nps"].corr(nps_df["vader_mean"])) if len(nps_df) > 2 else None
    logger.info("Saved NPS -> %s (corr w/ vader %.3f)", NPS_CSV, corr_nps if corr_nps is not None else float("nan"))

    # ---------------- 5. Aspect co-occurrence network ----------------
    nodes = []
    mention_counts = {a: 0 for a in ASPECT_NAMES}
    sent_acc = {a: [] for a in ASPECT_NAMES}
    for i, row in df.iterrows():
        low = str(row["content"]).lower()
        present = []
        for a in ASPECT_NAMES:
            if any(seed in low for seed in ASPECTS[a]["seeds"]):
                present.append(a)
                mention_counts[a] += 1
                sent_acc[a].append(float(row["vader_compound"]))
        # edges counted later in pairwise
    for a in ASPECT_NAMES:
        nodes.append({
            "id": a, "label": ASPECTS[a]["label"],
            "mentions": int(mention_counts[a]),
            "sentiment": round(float(np.mean(sent_acc[a])) if sent_acc[a] else 0.0, 3),
        })
    edges = []
    edge_counter = {}
    for _, row in df.iterrows():
        low = str(row["content"]).lower()
        present = [a for a in ASPECT_NAMES
                   if any(seed in low for seed in ASPECTS[a]["seeds"])]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                key = tuple(sorted((present[i], present[j])))
                edge_counter[key] = edge_counter.get(key, 0) + 1
    for (a, b), cnt in edge_counter.items():
        edges.append({"source": a, "target": b, "count": int(cnt)})
    edges.sort(key=lambda e: -e["count"])
    net = {"nodes": nodes, "edges": edges, "top_edges": edges[:25]}
    save_json(net, ASPECT_NETWORK_JSON)

    # ---------------- 6. PSI ranking ----------------
    # refine per-airline PSI combining recommend_rate, overall_rating, aspect mean, vader
    psi_rows = []
    for airline, g in df.groupby("airline_name"):
        if len(g) < 15:
            continue
        rec_rate = float(g["recommended_binary"].mean())
        arr = float(g["overall_rating"].mean())
        vader = float(g["vader_compound"].mean())
        aspect_mean = float(np.nanmean([g[ASPECTS[a]["rating_col"]].mean() for a in ASPECT_NAMES
                                         if ASPECTS[a]["rating_col"] in g.columns]))
        # normalize
        n_rec = rec_rate * 100
        n_rating = (arr - 1) / 9.0 * 100
        n_vader = (vader + 1) / 2.0 * 100
        n_aspect = (aspect_mean - 1) / 4.0 * 100 if not np.isnan(aspect_mean) else 50.0
        psi = 0.30 * n_rec + 0.30 * n_rating + 0.20 * n_vader + 0.20 * n_aspect
        psi_rows.append({
            "airline_name": airline, "reviews": int(len(g)),
            "psi_score": round(float(psi), 2),
            "recommend_rate_pct": round(n_rec, 2),
            "avg_overall_rating": round(arr, 3),
            "vader_mean": round(vader, 3),
            "avg_aspect_stars": round(float(aspect_mean), 3) if not np.isnan(aspect_mean) else None,
        })
    psi_df = pd.DataFrame(psi_rows).sort_values("psi_score", ascending=False).reset_index(drop=True)
    psi_df.to_csv(PSI_RANKING_CSV, index=False)
    logger.info("Saved PSI ranking -> %s (%d airlines)", PSI_RANKING_CSV, len(psi_df))

    # ---------------- 7. Word embeddings (Word2Vec skip-gram if available, else SVD co-occurrence) ----------------
    tokens_lists = [str(t).split() for t in df["content_processed"].tolist()]
    gensim_ok = False
    try:
        from gensim.models import Word2Vec as _W2V
        gensim_ok = True
    except Exception:
        gensim_ok = False

    if gensim_ok:
        w2v = _W2V(sentences=tokens_lists, vector_size=100, window=5,
                      min_count=20, workers=2, sg=1, epochs=6, seed=42)
        vocab = list(w2v.wv.index_to_key)
        w2i = {w: i for i, w in enumerate(vocab)}
        vecs = np.array([w2v.wv[w] for w in vocab], dtype=np.float32)
        method = "word2vec-skipgram"
    else:
        # fallback co-occurrence SVD
        flat = [w for toks in tokens_lists for w in toks]
        word_counts = pd.Series(flat).value_counts()
        vocab = [w for w in word_counts.index if word_counts[w] >= 30][:4000]
        w2i = {w: i for i, w in enumerate(vocab)}
        window = 4
        cooc = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
        for toks in tokens_lists:
            for i, w in enumerate(toks):
                if w not in w2i:
                    continue
                for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                    if j == i:
                        continue
                    w2 = toks[j]
                    if w2 in w2i:
                        cooc[w2i[w], w2i[w2]] += 1.0
        total = cooc.sum()
        row = cooc.sum(axis=1, keepdims=True)
        col = cooc.sum(axis=0, keepdims=True)
        expected = (row @ col) / max(total, 1)
        pmi = np.log((cooc + 1e-6) / (expected + 1e-6))
        pmi = np.clip(pmi, 0, None)
        svd = TruncatedSVD(n_components=100, random_state=42, n_iter=7)
        vecs = svd.fit_transform(pmi).astype(np.float32)
        method = "cooccurrence-svd"

    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = vecs / norms
    # 2D coords
    pca = PCA(n_components=2, random_state=42)
    coords2 = pca.fit_transform(vecs).astype(np.float32)
    embed = {
        "method": method,
        "vocab": vocab,
        "index": w2i,
        "vectors": vecs.astype(np.float32),
        "coords_2d": coords2.astype(np.float32),
    }
    joblib.dump(embed, WORD2VEC_PKL)
    logger.info("Saved word embeddings -> %s (%d words, %s)", WORD2VEC_PKL, len(vocab), method)

    logger.info("Phase 4 insights complete.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 4 Step 1: advanced insights.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv if argv is not None else [])
    return run(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
