# ✈️ Turbulence in Words — Mining the Passenger's Voice

An end-to-end **text-mining case study** on 41,000+ Skytrax airline passenger
reviews, delivered as a single **Streamlit dashboard** backed by a modular
Python pipeline.

```
Raw reviews ──► Clean ──► Features ──► Dim. reduction ──► Classification ──► Aspect tagging ──► Scorecard + Insights
   (web)       (NLTK)     (TF-IDF)     (LSA/LDA/t-SNE)    (ML models)       (ABSA)            (dashboard)
```

## Quickstart

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download(['stopwords','wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger'])"

# Build all artifacts (downloads the dataset on first run)
python -m pipeline.build_all

# Launch the dashboard
streamlit run app.py
```

> Behind a proxy? NLTK downloads may need `NLTK_ALLOW_PROXIED_URLOPEN=1`.

## Layout

```
app.py                  # the ONE Streamlit dashboard
pipeline/               # modular, re-runnable pipeline stages
  common.py             # paths, constants (RANDOM_STATE=42), IO helpers
  fetch_data.py         # Phase 1: download raw dataset
  clean.py              # Phase 1: cleaning + stats
  build_all.py          # orchestrator
data/                   # artifacts (raw, cleaned, features)
models/                 # trained models & metrics
report/                 # written case study
figures/                # exported plots
```

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Data foundation: download, cleaning, dashboard skeleton | ✅ complete |
| 2 | TF-IDF features + LSA/LDA/t-SNE | ✅ complete |
| 3 | Opinion classification + aspect tagging + scorecard | ✅ complete |
| 4 | Trends, clustering, PSI, networks, report | ✅ complete |

## Dataset

Skytrax user reviews (scraped from [airlinequality.com](https://www.airlinequality.com)),
mirrored at
[quankiquanki/skytrax-reviews-dataset](https://github.com/quankiquanki/skytrax-reviews-dataset).
Static CSV, no API key required.

## Reproducibility

All randomness is pinned (`RANDOM_STATE = 42`). Every pipeline stage is
idempotent: re-running reuses cached raw data unless `--force` is passed.

```bash
python -m pipeline.build_all --force            # full rebuild
python -m pipeline.build_all --only clean       # single stage
```
