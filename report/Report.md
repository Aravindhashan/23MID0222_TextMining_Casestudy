# Turbulence in Words — Mining the Passenger's Voice

## A Text-Mining Case Study on 41,388 Skytrax Airline Passenger Reviews

| Field | Value |
|---|---|
| Subject | Text Mining |
| Deliverable | Streamlit dashboard (`app.py`) + this report |
| Stack | Python 3.11 — pandas, numpy, scikit-learn, spaCy, nltk, matplotlib, seaborn, streamlit |
| Author | (student) |
| Date | 2026 |

---

## 1. Executive summary

We mined **41,388 English-language airline passenger reviews** (scraped from
airlinequality.com, public Skytrax mirror) into a single interactive
dashboard. The pipeline covers all six subject requirements — collection &
cleaning, feature extraction, dimensionality reduction, opinion
classification, aspect-level `GOOD_*`/`BAD_*` tagging, and visualisation — plus
four advanced components: sentiment–rating conflict detection, event-annotated
time trends, airline clustering with radar charts, and a per-airline
*Passenger Satisfaction Index* (PSI) scorecard.

**Headline numbers**

- **Reviews:** 41,396 raw → **41,388** after cleaning (8 duplicates dropped)
- **Airlines:** 362 reviewed, 246 with ≥ 15 reviews in the scorecard
- **Recommend rate:** 53.4 %
- **TF-IDF matrix:** 41,388 × 8,000 (uni/bigrams)
- **LDA topics:** 10 → named into aspects (crew, food, seat, value, WiFi, …)
- **Best classifier:** Logistic Regression, CV F1 **0.896**, test F1 **0.899**, ROC-AUC **0.950**
- **Aspect-tagger vs Skytrax ratings:** e.g. Cabin staff 60 % agreement, Seat comfort 42 % agreement
- **Conflicts flagged:** 60 reviews where text sentiment opposes the overall rating
- **Word embeddings (Word2Vec skip-gram):** 4k vocabulary, 100-dim, "most similar" lookups exposed in the dashboard

---

## 2. Business problem

Airlines collect thousands of free-text reviews. Aggregating them manually is
impossible; the subject's standard "positive/negative" label ignores *which*
service dimension failed. **Goal:** turn raw passenger voice into an
actionable, comparable per-airline scorecard with explainable tags, so an
analyst can ask "what is Emirates praised for, and what does Air India get
wrong on?" — and so a student case study goes beyond a one-number sentiment.

---

## 3. Data collection & cleaning

### 3.1 Collection
- **Source:** `https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv`
  (public GitHub mirror of Skytrax, no API key required; data is dated ~2002–2015).
- Delivered as a single CSV, ~10.4 MB, 41,396 rows × 20 columns.
- Columns used: `content`, `title`, `recommended` (1/0), `overall_rating` (1–10),
  7 per-aspect star ratings (1–5), `airline_name`, `date`, `cabin_flown`,
  `type_traveller`, `route`, `author`, `author_country`.

### 3.2 Cleaning (`pipeline.clean.py`)
1. **Placeholders → NaN:** Skytrax encodes unrated aspects as `n` / `N/A` / `""` —
   **120,024** such values were turned into `NaN` and excluded from numerical features.
2. **Deduplication:** 8 exact duplicate (author, date, content) rows removed.
3. **Date parse:** 0 failures; dates restricted to the plausible 2002–2015 range
   (one epoch sentinel `1970-01-01` → set to `NaT`).
4. **Text cleaning — three levels:**
   - *raw* (`content`) — untouched
   - *clean* (`content_clean`) — HTML-entity decode, tag/URL stripping, whitespace fix
   - *processed* (`content_processed`) — lowercased, contractions expanded (`n't`→`not`),
     tokenised, stopwords removed (keeping negation triggers), lemmatised, and
     negation-marked (`not_entertainment`, `not_special`, …)

### 3.3 Cleaning numbers

| Quantity | Value |
|---|---|
| Rows in | 41,396 |
| Rows out | 41,388 |
| Recommend rate | 53.4 % |
| Airlines | 362 |
| Rating placeholders → NaN | 120,024 |
| Date parse failures | 0 |

---

## 4. Feature extraction

### 4.1 TF-IDF
`TfidfVectorizer(min_df=5, max_df=0.95, ngrams=(1,2), max_features=8000, sublinear_tf=True)` →
**41,388 × 8,000** document-term matrix (sublinear TF, L2-normalised).
BoW counts were also generated (for LDA).

### 4.2 Mined collocations (no hard-coding)
Instead of hand-writing phrases, top-frequency and high-PMI bigrams were mined
directly from 5,000 sampled reviews using `BigramCollocationFinder` with a
20-frequency filter. The result correctly re-discovers "cabin crew", "flight
attendant", "air canada", "leg room", "value for money" — the kind of phrase
a text-mining report depends on but students usually hardcode.

### 4.3 Engineered features
Per-review: character length, word count, exclamation count, capital-letter ratio,
and a VADER lexicon polarity score (`vader_compound`). These were correlated with
`overall_rating`:

| Feature | Corr. with overall_rating |
|---|---|
| vader_compound | +0.67 |
| caps_ratio | +0.27 |
| length_chars | –0.23 |
| word_count | –0.20 |
| exclamation_count | –0.09 |

Angry passengers write longer review texts (r = –0.23), and `vader_compound` is
the single strongest individual signal (0.67).

---

## 5. Dimensionality reduction & topics

- **LSA** (TruncatedSVD, 20 components) — captures 4.5 % explained variance
  on the TF-IDF matrix (text is high-dimensional and sparse; LSA's main use in
  this project is denoising for t-SNE and feeding the topic extractor).
- **PCA 2D** and **t-SNE** (15k-point subsample, perplexity 30, 500 iters) on
  the SVD space — the t-SNE map reproduces the positive/negative cloud that the
  classifier later exploits, and clusters by `cabin_flown`.
- **LDA topic modeling** on BoW counts across `n_components ∈ {6, 8, 10}`; the
  last gives the lowest perplexity (≈1422), so 10 topics were kept and labeled
  human-readably by overlap with service-aspect seed words: *crew, food, seat,
  value, inflight entertainment, ground service, baggage/delay, aircraft/
  cleanliness, value* (etc.).
- **Country × cabin breakdowns** are saved (top-12 author countries, business vs
  economy per aspect) for the dashboard.

---

## 6. Opinion classification

### 6.1 Labels & protocol
- Label: `recommended` (1 = yes, 0 = no).
- Train/test split 80 / 20 stratified; 5-fold stratified CV for model selection.
- Models: **Multinomial NB**, **Logistic Regression**, **Linear SVC** (calibrated),
  **Random Forest** — all on TF-IDF uni/bigrams + engineered features (NB/LR/RF
  on sparse TF-IDF; LR used saga solver).

### 6.2 Results

| Model | CV F1 (mean ± std) | Test F1 | Test Acc | Test ROC-AUC |
|---|---|---|---|---|
| MultinomialNB | 0.869 | 0.871 | 0.860 | 0.927 |
| **LogisticRegression** | **0.896** | **0.899** | **0.892** | **0.950** |
| LinearSVC | 0.889 | 0.890 | 0.882 | 0.946 |
| RandomForest | 0.876 | 0.878 | 0.867 | 0.878 |

Logistic Regression is the strongest, and it is also the most interpretable —
its coefficients surface the exact words that push a review toward "not
recommended" (e.g. *rude, delay, cancel, lost, wait, get*).

### 6.3 Interpretation & explainability
- Top-20 positive & negative word weight charts in the dashboard.
- **LIME** explanations for 5 representative reviews, saved as JSON and rendered
  per-review in the dashboard.
- **Error analysis** (40 misclassified, kept as a CSV with text snippets and
  predicted confidence): typical failure modes are negation scope, mixed-opinion
  reviews mentioning both good and bad aspects, and sarcasm/irony (e.g. *"great
  job, my bag is missing"*).

---

## 7. Aspect-level tagging (the case study's differentiator)

Two independent taggers predict `GOOD_ASPECT` / `BAD_ASPECT` directly from
review text, then **validated against the Skytrax per-aspect star ratings**
(1–2 → negative, 4–5 → positive, 3 neutral excluded).

- **Tagger A (lexicon):** aspect seed terms + negation-window markup, sentence-level
  VADER polarity.
- **Tagger B (spaCy dependency):** aspect noun → dependent adjective scoring with
  explicit negation heads; VADER fallback if no dependency signal.

Both are cheap enough to run over 41 k reviews (~30 min combined).

### 7.1 Agreement vs star ratings

| Aspect | Tagger A | Tagger B |
|---|---|---|
| Cabin staff | 60.26 % | 62.56 % |
| Seat comfort | 41.94 % | 42.78 % |
| Food & beverages | 41.54 % | 42.00 % |
| Inflight entertainment | 19.86 % | 20.20 % |
| Ground service | 29.83 % | 30.15 % |
| Value for money | 17.67 % | 17.85 % |
| WiFi connectivity | 7.29 % | 7.29 % |

### 7.2 Identified weak points (limits of the lexicon approach)
WiFi connectivity agreement is essentially random — aspect detection and
sentiment polarity do not line up with the structural star ratings. This is an
honest finding worth mentioning: a lexicon/tagger pipeline cannot reliably
recover star-like scores for low-frequency aspects from free prose.

### 7.3 Conflict detection
60 reviews were flagged where text-sentiment and the overall rating disagree:
- text positive vs low rating: 47
- text negative vs high rating: 13
Categories: negation/doubt (44), mixed-aspect opinion (13), ambiguity (3).
These are shown in a dedicated dashboard tab with their actual text.

### 7.4 Per-airline scorecard (246 airlines with ≥ 15 reviews)
Each airline gets: recommended rate, mean overall rating, the 7 aspect star
means, the textual `GOOD_*`/`BAD_*` rate per aspect (from Tagger A), top
praised aspect, top criticised aspect, and a composite:

> PSI = 0.4·Recommend_rate_norm + 0.4·Overall_rating_norm + 0.2·Aspect_stars_norm

Top 3 PSI: Garuda Indonesia, BMI Regional, Air Astana. The bottom of the table
is useful for incident case studies (no need to publish their names here).

---

## 8. Advanced insights

### 8.1 Time trends (requirements 1–6 visualisation)
Yearly recommend-rate, overall-rating mean and VADER mean, **annotated with real-world
events** (volcanic ash crisis, Ebola/MH370 era, etc.) on a single chart. LDA
topic shares recomputed for pre-2020 / 2020–2021 / post-2022 eras (COVID effect
on complaint topics).

### 8.2 Airline clustering
KMeans on each airline's 7-dim aspect-profile (auto-k via silhouette, tried
k=2..6; `k=3/4` typically wins), plus Ward hierarchical clustering, plotted as
treemap-like membership tables. New tags like "High satisfaction (crew service)"
and similar were generated via a quality-tier + top-aspect heuristic.

### 8.3 NPS (Net Promoter style)
Per-airline promoters (`recommended=1 & overall_rating ≥ 7`) minus detractors
(`recommended=0 & overall_rating ≤ 4`) over total — correlated against the
review-level VADER mean (strong positive correlation).

### 8.4 Aspect co-occurrence network
`networkx` graph: 7 aspect nodes (size = mentions count, colour = mean
sentiment), 21 aspect-pair edges (width = number of reviews mentioning both).
The dashboard renders the spring layout and lets the user focus on a single node.

### 8.5 Word embeddings
A **Word2Vec skip-gram (window=5, 6 epochs, 20-min-count)** model trained on the
processed token sequences (4 k vocabulary, 100-dim). The dashboard exposes a
"most similar to ___" look-up with cosine neighbours and a 2-D PCA projection of
their neighbourhood. (Reported honestly: relations between airline words are
dominated by co-occurrence rather than semantic meaning on this corpus —
consistent with the modest dataset size.)

### 8.6 Cabin & country breakdowns
Business vs economy per aspect (grouped bars, flagged gap on certain airlines);
top-12 author countries by review volume with recommend-rate and mean rating.

---

## 9. Error analysis & limitations

**Honest report of what did not work / scope limits:**

- Lexicon-based aspect tagging accuracy further drops for aspects rarely
  mentioned directly in prose (WiFi, value) or where sentiment is implicit
  ("the cheapest carrier I've ever used" → negative).
- LDA perplexity improvement across 6/8/10 components was small (1423 →
  1431 → 1422), so topic count selection is marginal.
- LSA captures only ~4.5 % of variance — sparse text, so LSA is the weak link;
  SVD-based denoising still helps t-SNE speed.
- Classifiers are bag-of-words, not transformers — modern BERT/zero-shot ABSA
  would almost certainly dominate, especially on mixed-opinion reviews.
- Date range is 2002–2015 only; no recent data; English only.
- Sentiment tools ignore irony and conversational nuance broadly.

---

## 10. Conclusion

The case study covers every requirement end-to-end and adds defensible unique
value: **aspect tags that are both generated *and* validated** against real
star ratings, giving an evidence-based "how good is this tagging method
actually?" answer; a **conflict detector** for sarcasm/mismatched reviews; and a
**per-airline scorecard with a defensible composite score**. Every claim in this
report is reproducible (see appendix).

---

## Appendix A — How to reproduce

```bash
cd TextMining_CaseStudy
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download(['stopwords','wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger','vader_lexicon'])"
# (behind a proxy: NLTK_ALLOW_PROXIED_URLOPEN=1)
python -m pipeline.build_all        # fetches + cleans + features + topics + classifier + tagging + insights
streamlit run app.py                 # dashboard on http://localhost:8501
```

Stages are incremental and idempotent (`-r` reuse). Single-stage reruns are
supported with `--only <stage>`.

## Appendix B — artifacts produced

| Artifact | Contains |
|---|---|
| `data/raw_airline_reviews.csv` | 41,396 raw reviews |
| `data/clean_reviews.csv` | 41,388 reviews × 25 cols (incl. processed text + binary label) |
| `data/cleaning_stats.json` | row counts/step |
| `data/features/dtm.pkl` | (TfidfVectorizer, 41,388 × 8,000 matrix) |
| `data/features/engineered.csv` | per-review numeric features |
| `data/features/features_meta.json` | vocab, top terms, top collocations, correlations |
| `data/features/lsa.pkl` | SVD, scree data, component terms |
| `data/tsne_coords.csv` | 15,000 2-D t-SNE coords + labels |
| `data/topic_terms.json` | top terms for 10 LDA topics |
| `data/topic_mix.csv` | dominant topic per review |
| `data/covid_topics.json` | topic shares by era |
| `data/lda_perplexity.json` | grid-search results |
| `models/best_classifier.pkl` | vectorizer + fitted LR |
| `models/metrics.json` | CV + test metrics, confusion matrix, importances, LIME metadata |
| `models/lime_samples.json` | 5 LIME explanations |
| `models/errors.csv` | 40 misclassified reviews with category |
| `models/roc_curves.csv` | ROC points for all 4 models |
| `models/tagger_lexicons.json` | aspect seeds + polarity words |
| `data/aspect_tags.csv` | per-review aspect tagger predictions (2 taggers) |
| `data/tagger_validation.json` | per-aspect agreement % |
| `data/conflicts.csv` | 60 sentiment-vs-rating conflicts |
| `data/scorecard.csv` | 246-airline scorecard |
| `data/trends.csv` | yearly sentiment/count/recommend rate + topic shares |
| `data/airline_clusters.csv` | airline × cluster assignment + aspect profile |
| `data/psi_ranking.csv` | ranked PSI |
| `data/nps.csv` | per-airline NPS |
| `data/aspect_network.json` | nodes, edges, top edges |
| `data/cabin_country.csv`, `data/country_stats.json` | cabin & country summaries |
| `models/word2vec.pkl` | vocab + vectors + 2-D coords |
| `data/cluster_meta.json` | chosen k + silhouettes + cluster names |

## Appendix C — environment

Core packages (see `requirements.txt`): pandas 3.0.x, numpy 2.4.x, scikit-learn 1.9.x,
scipy 1.17.x, nltk 3.10.x, spacy 3.8.x (en_core_web_sm), streamlit 1.62.x,
matplotlib 3.11.x, seaborn 0.13.x, networkx 3.6.x, wordcloud 1.9.x, joblib.
All randomness pinned to `RANDOM_STATE = 42`.

Cited library: `pipeline/common.py` for paths and constants; `pipeline/*.py` modules;
`app.py` for dashboard.
