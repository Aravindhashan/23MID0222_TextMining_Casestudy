"""
Turbulence in Words - Airline Review Text-Mining Dashboard

Single Streamlit app. Run with:  streamlit run app.py

Reads artifacts from pipeline.build_all. Never trains on load.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from pipeline.common import (
    CLEANING_STATS_JSON, CLEAN_CSV, load_json,
    FEATURES_META_JSON, ENGINEERED_CSV, TSNE_CSV, TOPIC_TERMS_JSON,
    LDA_PERPLEXITY_JSON, LSA_PKL, DTM_PKL, LDA_MODEL_PKL,
    BEST_CLASSIFIER_PKL, METRICS_JSON, LIME_SAMPLES_JSON, ERRORS_CSV, ROC_CURVES_CSV,
    TAGGER_LEXICONS_JSON, ASPECT_TAGS_CSV, TAGGER_VALIDATION_JSON, CONFLICTS_CSV, SCORECARD_CSV,
    TRENDS_CSV, AIRLINE_CLUSTERS_CSV, PSI_RANKING_CSV, NPS_CSV, ASPECT_NETWORK_JSON,
    CABIN_COUNTRY_CSV, WORD2VEC_PKL,
)

sns.set_theme(style="white")

st.set_page_config(
    page_title="Turbulence in Words",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- theme ----------
def _theme() -> str:
    return st.session_state.get("theme", "light")


def _toggle_theme():
    st.session_state["theme"] = "dark" if st.session_state.get("theme", "light") == "light" else "light"


LIGHT_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700;9..144,900&family=Manrope:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --paper: #fafaf7;
  --ink: #0f172a;
  --ink-soft: #334155;
  --muted: #64748b;
  --line: #e5e5e0;
  --line-strong: #cbd5e1;
  --accent: #7f1d1d;
  --accent-soft: #fce4e4;
  --good: #166534;
  --bg: #fafaf7;
  --card: #ffffff;
  --shadow: none;
  --radius: 2px;
}

.stApp { background: var(--paper); }
html, body, [class*="css"] {
  font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--ink);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}
[data-testid="stMetricValue"], table, .stDataFrame, code, .mono {
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Fraunces', Georgia, serif;
  color: var(--ink);
  letter-spacing: -0.02em;
  font-weight: 600;
  line-height: 1.05;
}
h1 { font-size: clamp(2.4rem, 4.2vw, 3.6rem); font-weight: 700; }
h2 { font-size: clamp(1.5rem, 2.4vw, 2rem); margin-top: 2.4rem; border-top: 1px solid var(--line); padding-top: 1.2rem; }
h3 { font-size: 1.15rem; margin-top: 1.4rem; font-weight: 600; }
p, li { line-height: 1.65; font-size: 1rem; }
.dek { font-size: 1.15rem; color: var(--ink-soft); max-width: 62ch; line-height: 1.55; }
.byline { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 1.6rem; }

.stApp { background: var(--paper); }

::selection { background: var(--accent-soft); color: var(--ink); }
input, textarea, [contenteditable] { caret-color: var(--accent); }
:focus-visible { outline: 1px solid var(--accent); outline-offset: 3px; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line-strong); }

[data-testid="stSidebar"] {
  background: var(--paper);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: var(--ink); }
[data-testid="stSidebar"] .stRadio label { color: var(--ink); font-size: 0.9rem; }
[data-testid="stSidebar"] hr { border-color: var(--line); }

.pullquote {
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(3rem, 6vw, 5rem);
  font-weight: 700;
  line-height: 0.95;
  letter-spacing: -0.03em;
  color: var(--accent);
  margin: 0.4rem 0 0.2rem 0;
}
.pullquote-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted);
  margin-bottom: 0.2rem;
}
.fact-row { display: flex; gap: 2.4rem; flex-wrap: wrap; margin: 1.2rem 0 2rem 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 1rem 0; }
.fact { }
.fact .v { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600; color: var(--ink); }
.fact .k { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }

[data-testid="stMetric"] { background: transparent; border: none; padding: 0; box-shadow: none; }
[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 500; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 600; }

.stButton > button, .stDownloadButton > button {
  border-radius: 0;
  background: var(--ink);
  color: var(--paper);
  border: 1px solid var(--ink);
  font-weight: 500;
  padding: 0.4rem 1rem;
  letter-spacing: 0.02em;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--accent); border-color: var(--accent); }

.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius: 0; padding: 8px 14px; font-weight: 500; color: var(--muted); background: transparent; }
.stTabs [aria-selected="true"] { color: var(--ink); border-bottom: 2px solid var(--accent); }

.stDataFrame, .stTable { border: 1px solid var(--line); border-radius: 0; }

hr { border-color: var(--line); }
div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 0; background: var(--card); }

.hero {
  padding: 1.2rem 0 2rem 0;
  border-bottom: 1px solid var(--line);
  margin-bottom: 2rem;
  animation: hero-in 0.6s ease-out both;
}
@keyframes hero-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

footer { visibility: hidden; }
</style>
"""

DARK_CSS = r"""
<style>
:root {
  --paper: #0b0f14;
  --ink: #e6e6e1;
  --ink-soft: #cbd5e1;
  --muted: #94a3b8;
  --line: #1f2937;
  --line-strong: #334155;
  --accent: #f87171;
  --accent-soft: #3f1d1d;
  --good: #4ade80;
  --bg: #0b0f14;
  --card: #111827;
  --shadow: none;
  --radius: 2px;
}
.stApp { background: var(--paper); }
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }
.stDataFrame, .stTable { background: var(--card); }
div[data-testid="stExpander"] { background: var(--card); }
</style>
"""

st.markdown(LIGHT_CSS, unsafe_allow_html=True)
if _theme() == "dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def apply_chart_style(fig, ax=None):
    """Signature chart style: thin lines, no chartjunk, serif titles."""
    fig.patch.set_facecolor("none")
    if ax is not None:
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.8)
        ax.tick_params(colors="#64748b", labelsize=9)
        ax.xaxis.label.set_color("#334155")
        ax.yaxis.label.set_color("#334155")
        ax.title.set_fontfamily("Fraunces")
        ax.title.set_fontsize(12)
        ax.grid(False)
    return fig


# ---------- loaders ----------
@st.cache_data(show_spinner="Loading cleaned reviews ...")
def load_reviews():
    if not CLEAN_CSV.exists():
        return None
    return pd.read_csv(CLEAN_CSV, parse_dates=["date"], low_memory=False)


@st.cache_data(show_spinner=False)
def load_stats():
    return load_json(CLEANING_STATS_JSON) if CLEANING_STATS_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_features_meta():
    return load_json(FEATURES_META_JSON) if FEATURES_META_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_engineered():
    return pd.read_csv(ENGINEERED_CSV) if ENGINEERED_CSV.exists() else None


@st.cache_resource(show_spinner="Loading LSA ...")
def load_lsa():
    return joblib.load(LSA_PKL) if LSA_PKL.exists() else None


@st.cache_data(show_spinner=False)
def load_lda_perp():
    return load_json(LDA_PERPLEXITY_JSON) if LDA_PERPLEXITY_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_topic_terms():
    return load_json(TOPIC_TERMS_JSON) if TOPIC_TERMS_JSON.exists() else None


@st.cache_resource(show_spinner=False)
def load_classifier_bundle():
    return joblib.load(BEST_CLASSIFIER_PKL) if BEST_CLASSIFIER_PKL.exists() else None


@st.cache_data(show_spinner=False)
def load_metrics():
    return load_json(METRICS_JSON) if METRICS_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_lime_samples():
    return load_json(LIME_SAMPLES_JSON) if LIME_SAMPLES_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_errors():
    return pd.read_csv(ERRORS_CSV) if ERRORS_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_roc_curves():
    return pd.read_csv(ROC_CURVES_CSV) if ROC_CURVES_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_aspect_tags():
    return pd.read_csv(ASPECT_TAGS_CSV) if ASPECT_TAGS_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_tagger_validation():
    return load_json(TAGGER_VALIDATION_JSON) if TAGGER_VALIDATION_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_conflicts():
    return pd.read_csv(CONFLICTS_CSV) if CONFLICTS_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_scorecard():
    return pd.read_csv(SCORECARD_CSV) if SCORECARD_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_trends():
    return pd.read_csv(TRENDS_CSV) if TRENDS_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_airline_clusters():
    return pd.read_csv(AIRLINE_CLUSTERS_CSV) if AIRLINE_CLUSTERS_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_psi_ranking():
    return pd.read_csv(PSI_RANKING_CSV) if PSI_RANKING_CSV.exists() else None


@st.cache_data(show_spinner=False)
def load_nps():
    return pd.read_csv(NPS_CSV) if NPS_CSV.exists() else None


@st.cache_resource(show_spinner=False)
def load_network():
    return load_json(ASPECT_NETWORK_JSON) if ASPECT_NETWORK_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_cabin_country():
    return pd.read_csv(CABIN_COUNTRY_CSV) if CABIN_COUNTRY_CSV.exists() else None


@st.cache_resource(show_spinner=False)
def load_word_embeddings():
    return joblib.load(WORD2VEC_PKL) if WORD2VEC_PKL.exists() else None


@st.cache_data(show_spinner=False)
def load_tsne():
    return pd.read_csv(TSNE_CSV) if TSNE_CSV.exists() else None


def missing_artifacts_message():
    st.error("No pipeline artifacts found.\n\nRun the pipeline first:\n\n```bash\npython -m pipeline.build_all\n```")


def bar_fig(series, title, xlabel="", ylabel="", color="#0f172a", horizontal=False, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        sns.barplot(x=series.values, y=series.index.astype(str), ax=ax, color=color, width=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
    else:
        sns.barplot(x=series.index.astype(str), y=series.values, ax=ax, color=color, width=0.6)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.xticks(rotation=35, ha="right")
    ax.set_title(title, fontsize=12, weight="bold", fontfamily="Georgia")
    ax.grid(False)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#64748b", labelsize=9)
    fig.tight_layout()
    return fig


def sidebar_filters(df):
    st.sidebar.title("Turbulence in Words")
    st.sidebar.caption("Airline passenger review mining - Text Mining case study")

    if st.sidebar.button("Toggle theme", use_container_width=True):
        _toggle_theme()
        st.rerun()

    filters = {}
    if df is not None and not df.empty:
        st.sidebar.header("Filters")
        airlines = sorted(df["airline_name"].dropna().unique())
        filters["airlines"] = st.sidebar.multiselect("Airlines", airlines, placeholder="All airlines")
        if "year" in df.columns and df["year"].notna().any():
            ymin, ymax = int(df["year"].min()), int(df["year"].max())
            filters["year_range"] = st.sidebar.slider("Year range", ymin, ymax, (ymin, ymax))
        if "cabin_flown" in df.columns:
            cabins = sorted(df["cabin_flown"].dropna().unique())
            filters["cabins"] = st.sidebar.multiselect("Cabin class", cabins, placeholder="All cabins")

    st.sidebar.divider()
    st.sidebar.caption(
        "Pipeline status\n\n"
        "Data and cleaning: complete\n"
        "Features: complete\n"
        "Dimensionality reduction: complete\n"
        "Classification: complete\n"
        "Aspect tagging: complete\n"
        "Insights: complete\n"
    )
    return filters


def apply_filters(df, filters):
    out = df
    if filters.get("airlines"):
        out = out[out["airline_name"].isin(filters["airlines"])]
    if filters.get("year_range") and "year" in out.columns:
        lo, hi = filters["year_range"]
        out = out[out["year"].between(lo, hi) | out["year"].isna()]
    if filters.get("cabins"):
        out = out[out["cabin_flown"].isin(filters["cabins"])]
    return out


# ---------- pages ----------
def page_overview(df, stats):
    rec_rate = df["recommended_binary"].mean() * 100 if "recommended_binary" in df else float("nan")
    n_airlines = df["airline_name"].nunique() if "airline_name" in df else 0

    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("# Turbulence in Words")
    st.markdown('<p class="dek">Mining the passenger\'s voice - an end-to-end text-mining study of Skytrax airline reviews, ending in an opinion scorecard whose aspect tags are validated against real star ratings.</p>', unsafe_allow_html=True)
    st.markdown('<p class="byline">Text Mining case study - 41,388 reviews - 362 airlines - seed 42</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="pullquote-label">Recommend rate</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="pullquote">{rec_rate:.1f}%</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fact-row">'
        f'<div class="fact"><div class="v">{stats["rows_in"]:,}</div><div class="k">Reviews (raw)</div></div>'
        f'<div class="fact"><div class="v">{stats["rows_out"]:,}</div><div class="k">Reviews (cleaned)</div></div>'
        f'<div class="fact"><div class="v">{n_airlines:,}</div><div class="k">Airlines</div></div>'
        f'<div class="fact"><div class="v">{df["overall_rating"].mean():.2f}</div><div class="k">Mean overall rating</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Most-reviewed airlines")
        if "airline_name" in df.columns:
            top = df["airline_name"].value_counts().head(12)
            st.pyplot(bar_fig(top, "", horizontal=True, figsize=(7, 5), color="#0f172a"))
            plt.close("all")
    with right:
        st.subheader("Overall rating")
        if "overall_rating" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.histplot(df["overall_rating"].dropna(), bins=10, ax=ax, color="#7f1d1d", edgecolor="#0f172a", linewidth=0.6)
            ax.set_xlabel("rating (1-10)")
            ax.set_ylabel("reviews")
            ax.set_title("")
            apply_chart_style(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if "year" in df.columns and df["year"].notna().any():
        st.subheader("Reviews per year")
        yearly = df.groupby("year").size()
        fig, ax = plt.subplots(figsize=(12, 2.6))
        ax.plot(yearly.index, yearly.values, color="#0f172a", linewidth=1.4, marker="o", markersize=3)
        ax.set_xlabel("")
        ax.set_ylabel("reviews")
        apply_chart_style(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Project roadmap and data dictionary"):
        st.markdown(
            "**Roadmap** - Phase 1 data foundation (complete), Phase 2 features and reduction (complete), "
            "Phase 3 classification and tagging (complete), Phase 4 insights and report (complete).\n\n"
            "**Data dictionary** - content / content_clean / content_processed; recommended / recommended_binary; "
            "overall_rating (1-10) and seven aspect ratings (1-5); airline_name, date, year, month, cabin_flown, "
            "type_traveller, author_country, route, aircraft."
        )


def page_cleaning(df, stats):
    st.title("Data and Cleaning")
    st.markdown('<p class="dek">From raw scrape to analysis-ready reviews - what was removed and why.</p>', unsafe_allow_html=True)

    steps = stats.get("steps", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in", f"{stats['rows_in']:,}")
    c2.metric("Rows out", f"{stats['rows_out']:,}")
    c3.metric("Duplicates dropped", f"{steps.get('dropped_duplicates', 0):,}")
    c4.metric("Placeholders to NaN", f"{steps.get('rating_placeholders_to_nan', 0):,}")

    left, right = st.columns([3, 2])
    with left:
        drop_steps = {
            "empty content": steps.get("dropped_empty_content", 0),
            "duplicates": steps.get("dropped_duplicates", 0),
            "no tokens left": steps.get("dropped_no_tokens_after_processing", 0),
        }
        drop_series = pd.Series({k: v for k, v in drop_steps.items() if v})
        if not drop_series.empty:
            st.pyplot(bar_fig(drop_series, "Rows removed per cleaning step", color="#7f1d1d"))
            plt.close("all")
    with right:
        st.markdown("**Placeholder to NaN conversions**")
        st.metric("Rating placeholders normalised", f"{steps.get('rating_placeholders_to_nan', 0):,}")
        st.metric("Date parse failures", f"{steps.get('date_parse_failures', 0):,}")
        st.caption("Skytrax stores unrated aspects as n / N/A; these become NaN so they never pollute numeric features.")

    st.divider()
    st.subheader("Missing data after cleaning")
    rating_cols = [c for c in df.columns if c.endswith("_rating")]
    if rating_cols:
        miss = df[rating_cols].isna().mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 3.6))
        sns.barplot(x=miss.index, y=miss.values, ax=ax, color="#0f172a", width=0.6)
        ax.set_ylabel("Percent missing")
        ax.set_title("Share of unrated aspect values", fontsize=12, weight="bold", fontfamily="Georgia")
        plt.xticks(rotation=30, ha="right")
        apply_chart_style(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Rating distributions")
    picks = st.multiselect("Choose rating columns", rating_cols,
                           default=[c for c in ("overall_rating", "cabin_staff_rating", "value_money_rating") if c in rating_cols])
    if picks:
        n = len(picks)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
        for ax, col in zip(axes[0], picks):
            sns.histplot(df[col].dropna(), bins=10 if col == "overall_rating" else 5, ax=ax, color="#0f172a")
            ax.set_title(col, fontsize=11, fontfamily="Georgia")
        apply_chart_style(fig, axes[0][0])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close("all")

    st.divider()
    st.subheader("Before and after cleaning (text viewer)")
    if "content_processed" in df.columns:
        idx = st.number_input("Review index", 0, len(df) - 1, 0, step=1)
        row = df.iloc[int(idx)]
        t1, t2, t3 = st.tabs(["Original", "Clean", "Processed (NLP)"])
        t1.write(str(row.get("content", ""))[:3000])
        t2.write(str(row.get("content_clean", ""))[:3000])
        t3.write(str(row.get("content_processed", ""))[:3000])
        st.caption(f"Review number {int(idx)} - {row.get('airline_name', '')} - {row.get('date', '')}")


def page_features(df, stats):
    st.title("Feature Extraction")
    st.markdown('<p class="dek">TF-IDF (unigrams + bigrams), mined collocations, and engineered per-review features.</p>', unsafe_allow_html=True)

    fm = load_features_meta()
    if fm is None:
        st.warning("No feature artifacts. Run: python -m pipeline.build_all --only features")
        return
    eng = load_engineered()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{fm['n_reviews']:,}")
    c2.metric("TF-IDF features", f"{fm['n_features']:,}")
    c3.metric("Vocab", f"{fm['vocab_size']:,}")
    c4.metric("Matrix", f"{fm['tfidf_shape'][0]:,} x {fm['tfidf_shape'][1]:,}")

    left, right = st.columns([2, 1])
    with left:
        polarity_mode = st.radio("Top terms for:", ["positive", "negative"], horizontal=True)
        terms = fm["top_terms"][polarity_mode]
        ts = pd.Series({t["term"]: t["weight"] for t in terms}).sort_values()
        st.pyplot(bar_fig(ts, f"Top terms ({polarity_mode})", horizontal=True, figsize=(8, 6), color="#0f172a"))
        plt.close("all")
    with right:
        st.markdown("**TF-IDF parameters**")
        st.json(fm["tfidf_params"])
        corr = fm.get("engineered_correlations_with_overall_rating", {})
        if corr:
            st.markdown("**Engineered feature correlations with overall rating**")
            st.dataframe(pd.DataFrame({"feature": list(corr.keys()), "corr": list(corr.values())}), hide_index=True)

    st.divider()
    st.subheader("Mined collocations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Most frequent bigrams")
        freq = fm["top_collocations"]["frequent"][:15]
        st.dataframe(pd.DataFrame(freq), hide_index=True)
    with col2:
        st.markdown("Highest PMI bigrams")
        pmi = fm["top_collocations"]["pmi"][:15]
        st.dataframe(pd.DataFrame(pmi), hide_index=True)

    if eng is not None and not eng.empty:
        st.divider()
        st.subheader("Engineered features - distributions")
        num_cols = [c for c in eng.columns if c not in ("review_idx",)]
        pick = st.selectbox("Feature", num_cols, index=0 if num_cols else 0)
        if pick:
            fig, ax = plt.subplots(figsize=(8, 3.4))
            sns.histplot(eng[pick].dropna(), ax=ax, color="#0f172a", kde=True, edgecolor="#7f1d1d", linewidth=0.6)
            ax.set_title(pick, fontsize=12, weight="bold", fontfamily="Georgia")
            apply_chart_style(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def page_topics(df, stats):
    st.title("Topics and Dimensionality Reduction")
    st.markdown('<p class="dek">LSA, PCA, t-SNE and LDA topic modelling.</p>', unsafe_allow_html=True)

    lda_perp = load_lda_perp()
    tt = load_topic_terms()
    tsne = load_tsne()
    lsa = load_lsa()

    if lda_perp is not None:
        st.subheader("LDA perplexity (model selection)")
        perp = lda_perp.get("perplexities", {})
        st.dataframe(pd.DataFrame({"n_components": list(perp.keys()), "perplexity": list(perp.values())}), hide_index=True)
        st.caption(f"Selected: {lda_perp.get('selected_n_components')} components (perplexity {lda_perp.get('best_perplexity')})")

    if tt is not None:
        st.subheader("LDA topics")
        selected = st.selectbox("Topic", list(tt.keys()), format_func=lambda t: f"Topic {t} - {tt[t].get('label', '')}")
        t = tt[selected]
        terms = t.get("terms", [])
        weights = t.get("weights", [])
        if terms:
            ts = pd.Series(dict(zip(terms, weights))).sort_values()
            st.pyplot(bar_fig(ts, f"Topic {selected}: {t.get('label', '')}", horizontal=True, figsize=(8, 5), color="#0f172a"))
            plt.close("all")
        if lda_perp and "topic_distribution" in lda_perp:
            st.markdown("Reviews per dominant topic")
            dist = pd.Series(lda_perp["topic_distribution"], index=[f"T{i}" for i in range(len(lda_perp["topic_distribution"]))])
            st.pyplot(bar_fig(dist, "", figsize=(9, 3.2), color="#7f1d1d"))
            plt.close("all")

    if tsne is not None and not tsne.empty:
        st.divider()
        st.subheader("t-SNE map")
        color_col = st.selectbox("Colour by", ["recommended_binary", "overall_rating", "cabin_flown", "dominant_topic"], index=0)
        if color_col in tsne.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = "RdYlGn" if color_col == "recommended_binary" else "viridis"
            sc = ax.scatter(tsne["tsne_1"], tsne["tsne_2"], c=tsne[color_col], cmap=cmap, s=4, alpha=0.55, edgecolors="none")
            plt.colorbar(sc, ax=ax, shrink=0.7)
            ax.set_title(f"t-SNE coloured by {color_col}", fontsize=12, weight="bold", fontfamily="Georgia")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            apply_chart_style(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if lsa is not None:
        st.divider()
        st.subheader("LSA (TruncatedSVD)")
        scree = lsa.get("scree") if isinstance(lsa, dict) else None
        if scree is not None:
            fig, ax = plt.subplots(figsize=(8, 3.6))
            ax.plot(range(1, len(scree) + 1), scree, color="#0f172a", linewidth=1.4, marker="o", markersize=3)
            ax.set_title("Explained variance by SVD component", fontsize=12, weight="bold", fontfamily="Georgia")
            ax.set_xlabel("component")
            ax.set_ylabel("explained variance")
            apply_chart_style(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def page_classify(df, stats):
    st.title("Opinion Classification")
    st.markdown('<p class="dek">Predicting whether a passenger recommends the airline (recommended yes / no).</p>', unsafe_allow_html=True)

    metrics = load_metrics()
    bundle = load_classifier_bundle()
    roc = load_roc_curves()
    errors = load_errors()
    lime = load_lime_samples()

    if metrics is None:
        st.warning("No classification artifacts. Run: python -m pipeline.build_all --only classify")
        return

    results = metrics.get("results", {})
    if results:
        st.subheader("Model comparison")
        rows = []
        for name, r in results.items():
            rows.append({
                "model": name,
                "CV F1 (mean)": round(r.get("cv_f1_mean", 0), 3),
                "CV F1 (std)": round(r.get("cv_f1_std", 0), 3),
                "test F1": round(r.get("test_f1", 0), 3),
                "test accuracy": round(r.get("test_accuracy", 0), 3),
                "test ROC-AUC": round(r.get("test_roc_auc", 0), 3),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"Best model: {metrics.get('best_model_name')} (selected by CV F1).")

    if roc is not None and not roc.empty:
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, g in roc.groupby("model"):
            ax.plot(g["fpr"], g["tpr"], label=name, linewidth=1.4)
        ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves", fontsize=12, weight="bold", fontfamily="Georgia")
        ax.legend(frameon=False, fontsize=9)
        apply_chart_style(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    best = metrics.get("best_model_name")
    cm = results.get(best, {}).get("confusion_matrix") if best else None
    if cm is not None:
        st.subheader("Confusion matrix (best model)")
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", ax=ax, cbar=False, linewidths=0.5, linecolor="#e5e5e0")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    importances = metrics.get("importances", {})
    if importances.get("top_positive"):
        st.subheader("Top words pushing recommended")
        st.dataframe(pd.DataFrame(importances["top_positive"]), hide_index=True)
    if importances.get("top_negative"):
        st.subheader("Top words pushing not recommended")
        st.dataframe(pd.DataFrame(importances["top_negative"]), hide_index=True)

    if bundle is not None:
        st.divider()
        st.subheader("Live predictor")
        st.caption("Paste a review and get a recommended / not recommended prediction.")
        txt = st.text_area("Review text", height=120)
        if st.button("Predict") and txt.strip():
            vect = bundle["vectorizer"]
            model = bundle["model"]
            X = vect.transform([txt])
            pred = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else float("nan")
            label = "Recommended" if pred == 1 else "Not recommended"
            st.markdown(f"**Prediction:** {label}  (P(recommended) = {proba:.4f})")
            feature_names = np.asarray(vect.get_feature_names_out())
            coefs = getattr(model, "coef_", None)
            if coefs is not None:
                coefs = coefs.ravel()
                doc = X[0]
                idxs = doc.nonzero()[1]
                contrib = sorted([(feature_names[j], float(doc[0, j] * coefs[j])) for j in idxs], key=lambda x: x[1], reverse=True)
                top_pos = contrib[:8]
                top_neg = sorted(contrib, key=lambda x: x[1])[:8]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("Pushing recommended")
                    st.dataframe(pd.DataFrame(top_pos, columns=["word", "contribution"]), hide_index=True)
                with c2:
                    st.markdown("Pushing not recommended")
                    st.dataframe(pd.DataFrame(top_neg, columns=["word", "contribution"]), hide_index=True)

    if lime is not None and lime.get("available"):
        st.divider()
        st.subheader("LIME explanations (sample reviews)")
        for smp in lime.get("samples", [])[:5]:
            with st.expander(f"Review {smp.get('review_index')} - true={smp.get('true_label')} pred={smp.get('predicted_label')}"):
                st.write(smp.get("text_excerpt", ""))
                w = smp.get("words", [])
                if w:
                    st.dataframe(pd.DataFrame(w), hide_index=True)

    if errors is not None and not errors.empty:
        st.divider()
        st.subheader("Misclassified examples")
        st.dataframe(errors.head(20), hide_index=True)


def page_tagging(df, stats):
    st.title("Aspect Tagging")
    st.markdown('<p class="dek">Aspect-level GOOD / BAD tags predicted from text, validated against Skytrax star ratings.</p>', unsafe_allow_html=True)

    tv = load_tagger_validation()
    conflicts = load_conflicts()
    tags = load_aspect_tags()

    if tv is not None:
        st.subheader("Tagger agreement vs star ratings")
        rows = []
        for k, v in tv.items():
            rows.append({
                "aspect": v.get("aspect_label", k),
                "tagger A (%)": round(v["tagger_A"]["agreement_pct"], 2),
                "tagger B (%)": round(v["tagger_B"]["agreement_pct"], 2),
                "compared reviews": v["tagger_A"]["compared_reviews"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 4.2))
        aspects = [r["aspect"] for r in rows]
        a = [r["tagger A (%)"] for r in rows]
        b = [r["tagger B (%)"] for r in rows]
        x = np.arange(len(aspects))
        w = 0.35
        ax.bar(x - w / 2, a, w, label="Tagger A (lexicon)", color="#0f172a")
        ax.bar(x + w / 2, b, w, label="Tagger B (spaCy)", color="#7f1d1d")
        ax.set_xticks(x)
        ax.set_xticklabels(aspects, rotation=25, ha="right")
        ax.set_ylabel("agreement %")
        ax.set_title("Aspect tagger agreement vs Skytrax ratings", fontsize=12, weight="bold", fontfamily="Georgia")
        ax.legend(frameon=False, fontsize=9)
        apply_chart_style(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if tags is not None and not tags.empty:
        st.divider()
        st.subheader("Tag frequency (Tagger A)")
        tag_cols = [c for c in tags.columns if c.startswith("tag_")]
        if tag_cols:
            freq = tags[tag_cols].apply(lambda s: s.value_counts()).fillna(0)
            st.dataframe(freq, hide_index=True)

    if conflicts is not None and not conflicts.empty:
        st.divider()
        st.subheader("Sentiment vs rating conflicts")
        st.caption(f"{len(conflicts)} reviews where text sentiment opposes the overall rating.")
        st.dataframe(conflicts.head(20), hide_index=True)


def page_scorecard(df, stats):
    st.title("Airline Scorecard")
    st.markdown('<p class="dek">Per-airline aspect breakdown and composite Passenger Satisfaction Index.</p>', unsafe_allow_html=True)

    sc = load_scorecard()
    psi = load_psi_ranking()
    nps = load_nps()
    clusters = load_airline_clusters()

    if sc is None or sc.empty:
        st.warning("No scorecard artifact. Run: python -m pipeline.build_all --only aspect_tag")
        return

    airline = st.selectbox("Airline", sorted(sc["airline_name"].unique()))
    row = sc[sc["airline_name"] == airline].iloc[0]

    st.markdown(f"### {airline}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", f"{int(row['reviews']):,}")
    c2.metric("Recommend rate", f"{row['recommended_rate_pct']:.1f}%")
    c3.metric("Avg overall rating", f"{row['avg_overall_rating']:.2f}")

    if psi is not None and not psi.empty and airline in set(psi["airline_name"]):
        psi_row = psi[psi["airline_name"] == airline].iloc[0]
        st.metric("PSI score", f"{psi_row['psi_score']:.2f}")

    st.subheader("Aspect stars")
    aspects = ["seat_comfort", "cabin_staff", "food_beverages", "inflight_entertainment", "ground_service", "wifi_connectivity", "value_money"]
    avail = [a for a in aspects if f"{a}_stars" in sc.columns]
    if avail:
        sdf = pd.DataFrame({"aspect": [a.replace("_", " ").title() for a in avail],
                            "stars": [row[f"{a}_stars"] for a in avail]})
        st.dataframe(sdf, hide_index=True)

    st.divider()
    st.subheader("Top 15 airlines by PSI")
    if psi is not None and not psi.empty:
        pr = psi.head(15).set_index("airline_name")["psi_score"]
        st.pyplot(bar_fig(pr, "", horizontal=True, figsize=(8, 5), color="#0f172a"))
        plt.close("all")

    if nps is not None and not nps.empty:
        st.divider()
        st.subheader("NPS (top 10)")
        st.dataframe(nps.head(10), hide_index=True)


def page_trends(df, stats):
    st.title("Trends and Networks")
    st.markdown('<p class="dek">Sentiment over time, event annotations, aspect co-occurrence graph, and word embeddings.</p>', unsafe_allow_html=True)

    trends = load_trends()
    if trends is None:
        st.warning("No trends artifact. Run: python -m pipeline.build_all --only insights")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", f"{int(trends['year'].min())}-{int(trends['year'].max())}")
    c2.metric("Avg recommend rate", f"{trends['recommend_rate'].mean() * 100:.1f}%")
    c3.metric("Best year (VADER)", f"{int(trends.loc[trends['vader_mean'].idxmax(), 'year'])}")
    c4.metric("Worst year (VADER)", f"{int(trends.loc[trends['vader_mean'].idxmin(), 'year'])}")

    st.subheader("Sentiment and satisfaction over time")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(trends["year"], trends["vader_mean"], color="#0f172a", linewidth=1.4, marker="o", markersize=3, label="VADER mean")
    ax.plot(trends["year"], trends["recommend_rate"], color="#7f1d1d", linewidth=1.4, marker="s", markersize=3, label="Recommend rate")
    secondary = ax.twinx()
    secondary.plot(trends["year"], trends["overall_rating"], color="#64748b", marker="^", linestyle="--", linewidth=1.2, label="Overall rating")
    secondary.set_ylabel("overall rating (1-10)", color="#64748b")
    events = {2008: "2008 crisis", 2010: "Volcanic ash", 2013: "Dreamliner issues", 2014: "Ebola / MH370", 2015: "Dataset end"}
    ymin, ymax = ax.get_ylim()
    for yr, lab in events.items():
        if yr >= trends["year"].min() and yr <= trends["year"].max():
            ax.axvline(yr, color="#cbd5e1", linestyle=":", alpha=0.8)
            ax.text(yr + 0.2, ymax, lab, fontsize=8, rotation=90, verticalalignment="top", color="#64748b")
    ax.set_title("Yearly passenger sentiment", fontsize=12, weight="bold", fontfamily="Georgia")
    ax.set_xlabel("year")
    ax.set_ylabel("sentiment / rate")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    apply_chart_style(fig, ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    cc = load_cabin_country()
    if cc is not None and not cc.empty:
        st.divider()
        st.subheader("Cabin class performance")
        st.dataframe(cc.round(3), hide_index=True)
        if "cabin_flown" in cc and "overall_rating" in cc:
            fig, ax = plt.subplots(figsize=(8, 3.4))
            sns.barplot(data=cc, x="cabin_flown", y="overall_rating", ax=ax, color="#0f172a", width=0.6)
            ax.set_title("Avg overall rating by cabin", fontsize=11, weight="bold", fontfamily="Georgia")
            plt.xticks(rotation=30, ha="right")
            apply_chart_style(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    net = load_network()
    if net is not None:
        st.divider()
        st.subheader("Aspect co-occurrence network")
        nodes = pd.DataFrame(net["nodes"])
        edges = pd.DataFrame(net["edges"])
        import networkx as nx
        G = nx.Graph()
        for n in net["nodes"]:
            G.add_node(n["id"], label=n["label"], mentions=n["mentions"], sentiment=n["sentiment"])
        for e in net["edges"]:
            G.add_edge(e["source"], e["target"], weight=e["count"])
        layout = nx.spring_layout(G, seed=42, k=2)
        fig, ax = plt.subplots(figsize=(9, 6))
        node_sizes = [nodes.set_index("id").loc[n, "mentions"] * 2 for n in G.nodes()]
        node_colors = [nodes.set_index("id").loc[n, "sentiment"] for n in G.nodes()]
        nx.draw_networkx_nodes(G, layout, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.RdYlGn, ax=ax, linewidths=0.5, edgecolors="#0f172a")
        widths = [G[u][v]["weight"] / max(1, max(e["count"] for e in net["edges"])) * 4 for u, v in G.edges()]
        nx.draw_networkx_edges(G, layout, width=widths, alpha=0.3, ax=ax, edge_color="#94a3b8")
        nx.draw_networkx_labels(G, layout, labels={n: n for n in G.nodes()}, ax=ax, font_size=8, font_family="Manrope")
        ax.set_title("Aspect co-occurrence (node size = mentions, color = sentiment)", fontsize=11, weight="bold", fontfamily="Georgia")
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        picked = st.selectbox("Highlight aspect node", [n["id"] for n in net["nodes"]])
        sub = edges[(edges["source"] == picked) | (edges["target"] == picked)].sort_values("count", ascending=False).head(15)
        st.dataframe(sub, hide_index=True)

    clusters = load_airline_clusters()
    if clusters is not None and not clusters.empty:
        st.divider()
        st.subheader("Airline clusters")
        keep = [c for c in ["airline_name", "reviews", "cluster", "cluster_label", "overall_rating"] if c in clusters.columns]
        st.dataframe(clusters[keep].sort_values("cluster") if "cluster" in clusters.columns else clusters[keep], hide_index=True, height=300)

    psi = load_psi_ranking()
    if psi is not None and not psi.empty:
        st.divider()
        st.subheader("PSI ranking (top 20)")
        pr = psi.head(20).set_index("airline_name")["psi_score"]
        st.pyplot(bar_fig(pr, "", horizontal=True, figsize=(8, 6), color="#0f172a"))
        plt.close("all")

    nps = load_nps()
    if nps is not None and not nps.empty:
        st.divider()
        st.subheader("NPS (top 15)")
        st.dataframe(nps.head(15), hide_index=True)
        if "nps" in nps and "vader_mean" in nps:
            cc = nps["nps"].corr(nps["vader_mean"])
            st.caption(f"Correlation NPS vs review sentiment: {cc:.3f}")

    emb = load_word_embeddings()
    if emb is not None:
        st.divider()
        st.subheader("Word embeddings")
        vocab = emb["vocab"]
        index = emb["index"]
        vecs = emb["vectors"]
        coords = emb["coords_2d"]
        st.caption(f"Method: {emb.get('method', '?')} - vocab size {len(vocab)}")
        word = st.text_input("Word", value="delay", key="we_word")
        if word.strip().lower() in index:
            wi = index[word.strip().lower()]
            target = vecs[wi]
            sims = vecs @ target
            order = np.argsort(sims)[::-1][1:16]
            nbrs = [(vocab[i], float(sims[i])) for i in order]
            nbdf = pd.DataFrame(nbrs, columns=["neighbor", "cosine"])
            st.dataframe(nbdf, hide_index=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            nbr_words = [n[0] for n in nbrs] + [word.strip().lower()]
            idxs = [index[w] for w in nbr_words if w in index]
            pts = coords[idxs]
            ax.scatter(pts[:, 0], pts[:, 1], color="#0f172a", s=24)
            for w, p_ in zip(nbr_words, pts):
                ax.text(p_[0], p_[1], w, fontsize=9, fontfamily="Manrope")
            ax.set_title("2D semantic neighbourhood", fontsize=11, weight="bold", fontfamily="Georgia")
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("Word not in vocabulary. Try: delay, crew, food, seat, value, refund, cancel, comfortable.")


def page_about(df, stats):
    st.title("About and Reproducibility")
    st.markdown(
        "### Project: Turbulence in Words - Mining the Passenger's Voice\n"
        "End-to-end text-mining case study on Skytrax airline passenger reviews.\n\n"
        "### Pipeline stages\n"
        "1. Data collection and cleaning\n"
        "2. Feature extraction - TF-IDF (uni/bigrams), BoW, mined collocations, engineered features\n"
        "3. Dimensionality reduction - TruncatedSVD (LSA), PCA, t-SNE, LDA topic modeling\n"
        "4. Opinion classification - NB / Logistic Regression / SVM / RandomForest (best: LR)\n"
        "5. Aspect tagging - two independent taggers (lexicon+VADER vs spaCy dependency)\n"
        "6. Visualizations everywhere (word clouds, heatmaps, network graph, radar, time series, t-SNE)\n\n"
        "### Reproducibility\n"
        "All randomness pinned to seed 42. Pipeline is idempotent.\n\n"
        "```bash\npip install -r requirements.txt\npython -m pipeline.build_all\nstreamlit run app.py\n```\n"
    )
    st.markdown("### Limitations")
    st.markdown(
        "- Reviews are from 2002-2015 (no recent data, English only).\n"
        "- Lexicon-based aspect tagging accuracy drops for implicit sentiment.\n"
        "- Classifiers are bag-of-words, not transformers.\n"
        "- LSA captures only about 4.5 percent of variance.\n"
    )


PAGES = {
    "Overview": page_overview,
    "Data and Cleaning": page_cleaning,
    "Features": page_features,
    "Topics and Reduction": page_topics,
    "Opinion Classification": page_classify,
    "Aspect Tagging": page_tagging,
    "Airline Scorecard": page_scorecard,
    "Trends and Networks": page_trends,
    "About and Reproducibility": page_about,
}


def _ensure_nltk():
    import nltk
    for pkg in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab",
                "averaged_perceptron_tagger", "vader_lexicon"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def _ensure_artifacts():
    from pathlib import Path
    if not CLEAN_CSV.exists():
        with st.spinner("Building pipeline artifacts (first run, may take a few minutes)..."):
            import subprocess, sys
            try:
                subprocess.run([sys.executable, "-m", "pipeline.build_all"], check=True)
            except Exception as e:
                st.error(f"Pipeline build failed: {e}")
                st.stop()
        st.rerun()


def main():
    _ensure_nltk()
    _ensure_artifacts()
    df = load_reviews()
    stats = load_stats()

    filters = sidebar_filters(df)
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    if df is None or stats is None:
        missing_artifacts_message()
        return

    df_f = apply_filters(df, filters)
    if df_f.empty:
        st.warning("No reviews match the current filters.")
        return
    if filters.get("airlines") or filters.get("cabins") or filters.get("year_range"):
        st.caption(f"Filters active - showing {len(df_f):,} of {len(df):,} reviews.")

    PAGES[choice](df_f, stats)

    st.divider()
    st.caption("Turbulence in Words - Text Mining case study - Data: Skytrax reviews - All randomness pinned to seed 42.")


if __name__ == "__main__":
    main()