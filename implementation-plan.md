# Implementation Plan — "Turbulence in Words: Mining the Passenger's Voice"

**Subject:** Text Mining — End-to-End Case Study
**Stack:** Python 3.11 (pandas, numpy, scikit-learn, matplotlib, scipy, bs4, requests already installed)
**Deliverable:** ONE complete **Streamlit dashboard** (`app.py`) backed by a modular Python pipeline + written case-study report
**Working dir:** `C:\Users\aravi\Downloads\TextMining_CaseStudy`
**Structure:** 4 implementation phases (below), each ending with a verifiable checkpoint.

> No separate `.ipynb` notebooks. The case study is a single Streamlit app with sidebar navigation over cached artifacts produced by plain `.py` pipeline modules.

---

## 1. Project Idea (Unique Angle)

End-to-end opinion mining of **41,396 airline passenger reviews** (Skytrax), presented as an interactive dashboard ending in an **Airline Opinion Scorecard** with 👍/👎 aspect tags derived from text alone and **validated against real per-aspect star ratings**.

Standing-out points:
1. Aspect tags *predicted from text* and *evaluated* vs ground-truth star ratings (agreement %).
2. Business-facing artifact: composite **Passenger Satisfaction Index (PSI)** ranking + per-airline scorecard.
3. Extras rarely seen in coursework: sentiment–rating conflict detection, event-annotated time trends, airline clustering with radar charts, LIME explainability, collocation mining, aspect co-occurrence network, Word2Vec semantic map.

## 2. Dataset (Static — No API Keys)

- **URL:** `https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv`
- **Size:** ~10.4 MB, 41,396 reviews (verified HTTP 200 from this machine)
- **Cite:** github.com/quankiquanki/skytrax-reviews-dataset (scraped from airlinequality.com)

| Columns | Use |
|---|---|
| `content`, `title` | Text → cleaning → features |
| `recommended` (yes/no) | Label for opinion classification |
| `overall_rating` (1–10) | Regression target, sentiment calibration, conflict detection |
| `seat_comfort`, `cabin_staff`, `food_beverages`, `ground_service`, `inflight_entertainment`, `value_money`, `wifi_connectivity` (1–5) | Ground truth for validating the aspect tagger |
| `airline_name`, `date`, `type_traveller`, `cabin_flown`, `route`, `author_country` | Group-by dims: scorecard, time trends, clustering, country/cabin analysis |

## 3. Architecture: Pipeline + Dashboard

```
[1] pipeline/*.py  — deterministic, seed=42, run per-phase
        ↓ writes artifacts to data/ and models/
[2] app.py         — Streamlit dashboard, reads artifacts only, loads in seconds
```

- Pipeline modules are importable functions; each `main()` saves its artifacts → the dashboard never trains models on load.
- `python -m pipeline.build_all` refreshes every phase's artifacts end-to-end.

## 4. Repository Layout (grows phase by phase)

```
TextMining_CaseStudy/
├── app.py                      # the ONE Streamlit dashboard
├── pipeline/
│   ├── __init__.py  common.py  build_all.py
│   ├── fetch_data.py  clean.py            # Phase 1
│   ├── features.py  dimreduce.py          # Phase 2
│   ├── classify.py  aspect_tag.py         # Phase 3
│   └── insights.py                        # Phase 4
├── data/                       # raw, cleaned, features/, all *.csv artifacts
├── models/                     # *.pkl, tagger_lexicons.json, metrics.json
├── report/Report.md  figures/
├── requirements.txt
└── README.md                   # quickstart: install → build_all → streamlit run
```

---

# PHASE 1 — Data Foundation & Dashboard Skeleton
**Covers requirement (1) — collection + cleaning. Ends with a runnable dashboard.**

### Tasks
1. **Env setup:** `pip install streamlit nltk seaborn wordcloud spacy lime networkx`; `python -m spacy download en_core_web_sm`; NLTK downloads (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`); write `requirements.txt`.
2. **Scaffolding:** create repo layout, `pipeline/common.py` (paths, `RANDOM_STATE=42`, load/save helpers), `pipeline/__init__.py`, initial `pipeline/build_all.py`.
3. **`pipeline/fetch_data.py`:** retry-safe download of the CSV → `data/raw_airline_reviews.csv`; profile printout (shape, dtypes, date span, top airlines).
4. **`pipeline/clean.py`:**
   - Map `"n"` / `"N/A"` / empty-string placeholders in the 7 rating columns → `NaN`
   - Dedup on (author, date, content); drop empty-content rows
   - Text: HTML-entity decode, contractions expanded, lowercase, tokenize, stopword removal, lemmatize, keep negation markers (`not_*`), strip non-alpha
   - Parse `date` → datetime + `year`/`month`; compute cleaning-stats (rows in/out, per-step counts)
   - Outputs: `data/clean_reviews.csv`, `data/cleaning_stats.json`
5. **Dashboard skeleton `app.py`:** sidebar shell + page router + `@st.cache_data` loaders; **pages: 🏠 Overview, 🧹 Data & Cleaning** (raw-vs-clean counts, cleaning-stats bar, missing-data heatmap of rating columns, rating distributions, before/after text viewer).

### ✅ Checkpoint 1
- `python -m pipeline.build_all` downloads + cleans without errors and is idempotent
- `streamlit run app.py` shows 2 working pages with real numbers; clean CSV has sensible row count (~41k → deduped) and no placeholder values in rating columns

---

# PHASE 2 — Features & Dimensionality Reduction
**Covers requirements (2) and (3). Adds 2 dashboard pages.**

### Tasks
1. **`pipeline/features.py`:**
   - BoW counts + TF-IDF (unigrams+bigrams; tune `min_df`, `max_df`, `max_features`)
   - **Collocation/phrase mining:** PMI- or frequency-filtered bigrams ("customer service", "value for money") → auto-extend vocabulary/phrases
   - Engineered features: review length, exclamation count, caps ratio, TextBlob polarity (lexicon baseline)
   - Outputs: `data/features/dtm.pkl`, `data/features/engineered.csv`, `features_meta.json` (vocab, top terms, collocations)
2. **`pipeline/dimreduce.py`:**
   - TruncatedSVD (LSA) on TF-IDF + explained-variance/scree data
   - LDA topics (grid 6–10, pick by coherence) → hand-name topics (crew, delays/baggage, food, value, punctuality, entertainment, wifi); save top terms per topic
   - t-SNE (subsample ≤20k) + PCA → 2D coords with `recommended`, `overall_rating`, `cabin_flown` labels
   - Outputs: `data/features/lsa.pkl`, `models/lda_model.pkl`, `data/tsne_coords.csv`, `data/topic_terms.json`, `data/topic_mix.csv`
3. **Dashboard pages:**
   - **🔤 Features:** top-20 term bar (pos/neg toggle), TF-IDF stats, top-collocations table, length/exclamation distributions + correlation with rating, word clouds (overall)
   - **📉 Topics & Reduction:** LDA topic–term bar charts (topic picker), topic-mix donut, **t-SNE scatter** colored by chosen column, scree plot, economy-vs-business grouped bars, country×sentiment heatmap

### ✅ Checkpoint 2
- `build_all` regenerates features + topics artifacts
- Dashboard shows interactive t-SNE and at least 5 interpretable, hand-named topics

---

# PHASE 3 — Opinion Classification & Aspect Tagging (core novelty)
**Covers requirements (4) and (5). Adds 3 dashboard pages including the centerpiece scorecard.**

### Tasks
1. **`pipeline/classify.py`:**
   - Label: `recommended` binary; secondary 3-bin `overall_rating` task
   - Models: Multinomial NB, Logistic Regression, Linear SVM, Random Forest (TF-IDF + engineered features); stratified k-fold CV; accuracy/precision/recall/F1/ROC-AUC
   - Save best model; LR top-feature importances (words driving "not recommended")
   - **LIME explanations** for 5 representative samples → `models/lime_samples.json`
   - **Misclassification analysis** → `models/errors.csv` with error categories (negation, mixed opinion, sarcasm, other)
   - **Length-vs-rating correlation** stats (angry passengers write longer?)
   - Outputs: `models/best_classifier.pkl`, `models/metrics.json`, `models/roc_curves.csv`
2. **`pipeline/aspect_tag.py`:**
   - **Tagger A (lexicon+negation):** 7 aspect lexicons + sentiment lexicon → tags `GOOD_CREW`, `BAD_FOOD`, etc.
   - **Tagger B (spaCy dependency ABSA):** link aspect nouns → sentiment adjectives via dependency paths, sentence-level
   - **Validation:** implied aspect sentiment vs the 7 star-rating columns (1–2=👎, 3=neutral, 4–5=👍) → per-aspect agreement % / accuracy for each tagger
   - **Conflict detection:** flag reviews where text sentiment contradicts `overall_rating`; categorize ~10 manual samples (sarcasm, aspect mismatch, …)
   - **Scorecard:** per-airline table — % positive tags per aspect, recommended-rate, avg rating
   - Outputs: `models/tagger_lexicons.json`, `data/aspect_tags.csv`, `data/tagger_validation.json`, `data/conflicts.csv`, `data/scorecard.csv`
3. **Dashboard pages:**
   - **🤖 Opinion Classification:** model-comparison table (CV ± std), overlaid ROC curves, confusion matrix, threshold slider, top "not recommended" words, **"Try it yourself" live review predictor**, LIME display, misclassified-examples browser
   - **🏷️ Aspect Tagging:** Tagger A-vs-B agreement bars per aspect, **live tagger demo** (paste review → tags), tag frequency chart, conflict explorer dropdown
   - **🏆 Airline Scorecard:** airline picker → radar chart vs cluster average, per-aspect 👍/👎 bars, recommended-rate, sample glowing/scathing reviews, sortable PSI-placeholder league table (PSI finalized in Phase 4)

### ✅ Checkpoint 3
- Best classifier F1 ≥ ~0.90 expected (recommendation is strongly lexical); at least one tagger ≥ ~75% agreement with star ratings per aspect is reported honestly
- All 3 pages interactive; live prediction and live tagger demos work

---

# PHASE 4 — Advanced Insights, Dashboard Completion & Report
**Covers requirement (6) visualization + all add-ons. Finishes the project.**

### Tasks
1. **`pipeline/insights.py`:**
   - **Time trends:** yearly sentiment & recommend-rate; **event annotations** (COVID-2020, airline incidents); pre/post-COVID LDA topic shifts
   - **Airline clustering:** k-means + hierarchical on airlines' 7-dim aspect profiles; silhouette-based k; cluster labels
   - **PSI (Passenger Satisfaction Index):** weighted blend of (text sentiment, aspect ratings, recommend-rate) → ranked league table
   - **NPS:** Promoters/Passives/Detractors per airline + correlation with text sentiment
   - **Aspect co-occurrence network:** networkx JSON (nodes=aspect, edge=co-mention, sized by freq, colored by sentiment)
   - **Cabin/country breakdowns:** business-vs-economy per aspect; top-10 countries × sentiment
   - **Word2Vec:** skip-gram on reviews → most-similar-to any term + 2D semantic coords of aspect terms
   - Outputs: `data/trends.csv`, `data/airline_clusters.csv`, `data/psi_ranking.csv`, `data/nps.csv`, `data/aspect_network.json`, `data/cabin_country.csv`, `models/word2vec.pkl`
2. **Dashboard completion (`app.py`):**
   - **📅 Trends & Networks:** sentiment-over-time line with event annotations, pre/post-COVID topic comparison, **co-occurrence network graph** (node picker), Word2Vec "similar to ___" input + semantic map
   - **ℹ️ About & Reproducibility:** methodology, dataset citation, limitations, future work, package versions, **"Re-run full pipeline" button** (subprocess `build_all`)
   - **Global sidebar filters** (airline multi-select, cabin class, year range) wired across all pages; consistent styling; export-to-PNG/CSV buttons on key tables
3. **`report/Report.md`:** business problem → data → cleaning → features → topics → classification → tagging + validation + conflicts → scorecard/PSI → trends/clusters → error analysis → limitations → future work; embed figure PNGs (`figures/`); dataset citation; reproducibility steps.
4. **`README.md` + `requirements.txt` finalization:** quickstart (`pip install -r requirements.txt` → `python -m pipeline.build_all` → `streamlit run app.py`).

### ✅ Checkpoint 4 (final)
- Fresh clone → 3 commands → full dashboard in < ~15 min (incl. pipeline run)
- All 9 pages populated; global filters work; PSI league table + scorecard complete
- Report complete with embedded figures and honest limitations section

---

# Global Rules & Scope Guardrails

- Single deliverable = one Streamlit app; **no `.ipynb` files** in the final repo
- `random_state=42` everywhere; pipeline idempotent; dashboard reads artifacts only (no training on load); `streamlit run app.py` stays fast
- No scraping, no Kaggle token, no GPU — offline after one CSV download
- spaCy ABSA is rule-based (dependency paths), course-appropriate
- **If time runs short:** demote to "future work" in this order — (a) "Re-run pipeline" button, (b) Word2Vec semantic map, (c) LIME page section. The 9 core pages and all 4 phase checkpoints stay.
