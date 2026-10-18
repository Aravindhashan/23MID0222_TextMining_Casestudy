"""
pipeline.aspect_tag
-------------------
Phase 3 Step 2 — Aspect-based tagging (good/bad per aspect) + validation + scorecard.

Tagger A: lexicon + negation-window based, sentence-level VADER sentiment.
Tagger B: spaCy dependency-parsing based (modifier scoring) + sentence VADER fallback.
Both tags are compared against the per-aspect Skytrax star ratings to report agreement%.

Outputs:
   models/tagger_lexicons.json
   data/aspect_tags.csv        (per-review aspect sentiment columns + tag string)
   data/tagger_validation.json (agreement per aspect for both taggers)
   data/conflicts.csv          (reviews where text sentiment conflicts with rating)
   data/scorecard.csv          (per-airline scorecard)
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
except Exception:
    pass

from .common import (
    ASPECT_TAGS_CSV,
    CONFLICTS_CSV,
    SCORECARD_CSV,
    TAGGER_LEXICONS_JSON,
    TAGGER_VALIDATION_JSON,
    logger,
    load_json,
    save_json,
    ensure_dirs,
)

ASPECTS = {
    "seat_comfort": {
        "seeds": ["seat", "legroom", "leg room", "comfort", "recline", "space", "pitch"],
        "rating_col": "seat_comfort_rating",
        "label": "Seat comfort",
    },
    "cabin_staff": {
        "seeds": ["crew", "staff", "attendant", "service", "flight attendant", "hostess", "steward"],
        "rating_col": "cabin_staff_rating",
        "label": "Cabin staff",
    },
    "food_beverages": {
        "seeds": ["food", "meal", "snack", "drink", "beverage", "catering", "dinner", "lunch", "breakfast"],
        "rating_col": "food_beverages_rating",
        "label": "Food & beverages",
    },
    "inflight_entertainment": {
        "seeds": ["entertainment", "movie", "screen", "tv", "wifi", "inflight", "monitor", "magazine"],
        "rating_col": "inflight_entertainment_rating",
        "label": "Inflight entertainment",
    },
    "ground_service": {
        "seeds": ["ground", "check", "boarding", "gate", "counter", "airport", "lounge"],
        "rating_col": "ground_service_rating",
        "label": "Ground service",
    },
    "wifi_connectivity": {
        "seeds": ["wifi", "internet", "connectivity", "connection", "signal"],
        "rating_col": "wifi_connectivity_rating",
        "label": "WiFi connectivity",
    },
    "value_money": {
        "seeds": ["value", "money", "price", "cheap", "expensive", "fare", "worth", "cost"],
        "rating_col": "value_money_rating",
        "label": "Value for money",
    },
}

POL_POS = {"great", "excellent", "good", "friendly", "comfortable", "nice", "amazing",
           "helpful", "clean", "fantastic", "wonderful", "polite", "attentive", "prompt",
           "fine", "superb", "smooth", "caring"}
POL_NEG = {"bad", "terrible", "rude", "delayed", "cancelled", "worst", "poor", "dirty",
           "horrible", "awful", "lost", "late", "rude", "old", "unclean", "waste",
           "uncomfortable", "noisy"}


def split_sentences(text):
    return [s for s in text.replace("!", ".").split(".") if s.strip()]


def vader_sentence_score(s, sia):
    try:
        return sia.polarity_scores(s)["compound"]
    except Exception:
        return 0.0


class TaggerA:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def predict(self, text):
        text = str(text)
        low = text.lower()
        sents = split_sentences(text)
        result = {}
        for aspect, info in ASPECTS.items():
            seeds = info["seeds"]
            if not any(seed in low for seed in seeds):
                result[aspect] = "none"
                continue
            scores = []
            for sent in sents:
                sl = sent.lower()
                if any(seed in sl for seed in seeds):
                    scores.append(vader_sentence_score(sent, self.sia))
            agg = (sum(scores) / len(scores)) if scores else 0.0
            if agg >= 0.12:
                result[aspect] = "good"
            elif agg <= -0.12:
                result[aspect] = "bad"
            else:
                result[aspect] = "neutral"
        return result


class TaggerB:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except Exception as e:
            logger.warning("spaCy model unavailable for tagger B: %s", e)
            self.nlp = None

    def _dep_modifiers(self, doc, seeds):
        score = 0
        for tok in doc:
            if tok.lemma_.lower() in seeds or tok.lower_ in seeds:
                # modifier candidates (adjectives / adverbs / heads)
                cand = [c for c in tok.children if c.dep_ in ("amod", "advmod", "acomp", "nsubj")]
                if tok.head is not None and tok.head != tok:
                    cand.append(tok.head)
                for m in cand:
                    # negation
                    neg = any(child.lower_ in {"not", "no", "never", "n't"} or
                              child.dep_ == "neg" for child in m.children) or \
                          (m.head is not None and m.head.dep_ == "neg")
                    if m.lower_ in POL_POS:
                        score += -1 if neg else 1
                    elif m.lower_ in POL_NEG:
                        score += 1 if neg else -1
        return score

    def predict(self, text):
        text = str(text)
        low = text.lower()
        result = {}
        doc = self.nlp(text.lower()) if self.nlp is not None else None
        sents = split_sentences(text)
        for aspect, info in ASPECTS.items():
            seeds = info["seeds"]
            seeds_low = [s.lower() for s in seeds]
            if not any(seed in low for seed in seeds_low):
                result[aspect] = "none"
                continue
            dep_score = 0
            if doc is not None:
                # dependency mods from whole doc
                seed_set = set()
                for sd in seeds_low:
                    seed_set |= set(sd.split())
                dep_score = self._dep_modifiers(doc, seed_set)
            # fallback sentence-level VADER when no decisive dependency signal
            scores = []
            for sent in sents:
                sl = sent.lower()
                if any(seed in sl for seed in seeds_low):
                    scores.append(vader_sentence_score(sent, self.sia))
            sent_score = (sum(scores) / len(scores)) if scores else 0.0
            if abs(dep_score) >= 1:
                agg = dep_score
                if agg >= 1:
                    result[aspect] = "good"
                elif agg <= -1:
                    result[aspect] = "bad"
                else:
                    result[aspect] = "neutral"
            else:
                if sent_score >= 0.12:
                    result[aspect] = "good"
                elif sent_score <= -0.12:
                    result[aspect] = "bad"
                else:
                    result[aspect] = "neutral"
        return result


def agree(pred_val, stars):
    if star_val is None or (isinstance(star_val, float) and np.isnan(star_val)):
        return False
    if star_val >= 4:
        truth = "good"
    elif star_val <= 2:
        truth = "bad"
    else:
        return None  # 3-star neutral excluded
    if pred_val == "good" and truth == "good":
        return True
    if pred_val == "bad" and truth == "bad":
        return True
    return False


def run(force: bool = False):
    ensure_dirs()
    from .common import CLEAN_CSV
    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    df = df.reset_index(drop=True)

    tagger_a = TaggerA()
    tagger_b = TaggerB()

    tag_rows = []
    logger.info("Tagging %d reviews ...", len(df))

    a_preds = {a: [] for a in ASPECTS}
    b_preds = {a: [] for a in ASPECTS}
    a_msg = "good"
    for i, row in df.iterrows():
        txt = row["content"] if pd.notna(row["content"]) else ""
        ra = tagger_a.predict(txt)
        rb = tagger_b.predict(txt)
        for a in ASPECTS:
            a_preds[a].append(ra.get(a, "none"))
            b_preds[a].append(rb.get(a, "none"))
        good_tags = [f"GOOD_{a.upper()}" for a, v in ra.items() if v == "good"]
        bad_tags = [f"BAD_{a.upper()}" for a, v in ra.items() if v == "bad"]
        plugin = good_tags + bad_tags
        tag_rows.append({
            "review_idx": i,
            "tags": "; ".join(plugin) if plugin else "none",
        })
        if (i + 1) % 5000 == 0:
            logger.info("  tagged %d / %d", i + 1, len(df))

    # add per-aspect columns
    tags_df = pd.DataFrame(tag_rows)
    for a in ASPECTS:
        tags_df[f"ta__{a}"] = a_preds[a]
        tags_df[f"tb__{a}"] = b_preds[a]
    tags_df.to_csv(ASPECT_TAGS_CSV, index=False)
    logger.info("Saved aspect_tags -> %s", ASPECT_TAGS_CSV)

    # validation agreement
    val = {}
    for a, info in ASPECTS.items():
        col = info["rating_col"]
        stars = df[col] if col in df.columns else pd.Series([None] * len(df))
        a_tot, a_hit = 0, 0
        b_tot, b_hit = 0, 0
        for i in range(len(df)):
            s = stars.iloc[i]
            pa = a_preds[a][i]
            pb = b_preds[a][i]
            resa = agree(pa, s)
            if resa is not None:
                a_tot += 1
                if resa:
                    a_hit += 1
            resb = agree(pb, s)
            if resb is not None:
                b_tot += 1
                if resb:
                    b_hit += 1
        val[a] = {
            "aspect_label": info["label"],
            "rating_col": col,
            "tagger_A": {
                "agreement_pct": round(100.0 * a_hit / a_tot, 2) if a_tot else None,
                "compared_reviews": int(a_tot),
                "agreed": int(a_hit),
            },
            "tagger_B": {
                "agreement_pct": round(100.0 * b_hit / b_tot, 2) if b_tot else None,
                "compared_reviews": int(b_tot),
                "agreed": int(b_hit),
            },
        }
    save_json(val, TAGGER_VALIDATION_JSON)

    # conflict detection
    sia = SentimentIntensityAnalyzer()
    rows = []
    for i, row in df.iterrows():
        txt = str(row["content"]) if pd.notna(row["content"]) else ""
        compound = sia.polarity_scores(str(txt))["compound"] if txt.strip() else 0.0
        try:
            rating = float(row["overall_rating"])
        except (TypeError, ValueError):
            rating = None
        if rating is None or (isinstance(rating, float) and np.isnan(rating)):
            continue
        conflict = None
        if compound >= 0.2 and rating <= 3:
            conflict = "positive text vs low rating"
        elif compound <= -0.2 and rating >= 7:
            conflict = "negative text vs high rating"
        if conflict is None:
            continue
        low_txt = txt.lower()
        if any(w in low_txt for w in ["sarcasm", "great job", "thanks", "oh great", "yeah right"]):
            cat = "possible sarcasm"
        elif "not" in low_txt:
            cat = "negation / doubt mismatch"
        else:
            aspects_mentioned = [a for a in ASPECTS if any(s in low_txt for s in ASPECTS[a]["seeds"])]
            if len(aspects_mentioned) >= 3:
                cat = "mixed-aspect opinion"
            else:
                cat = "ambiguity/nuance"
        rows.append({
            "review_idx": int(i),
            "airline_name": str(row["airline_name"]) if "airline_name" in row else "",
            "overall_rating": rating,
            "vader_compound": round(float(compound), 3),
            "conflict_type": conflict,
            "category": cat,
            "text_excerpt": txt[:260],
        })
    conflicts = pd.DataFrame(rows)
    if not conflicts.empty:
        conflicts = conflicts.head(60).reset_index(drop=True)
    conflicts.to_csv(CONFLICTS_CSV, index=False)
    logger.info("Saved conflicts -> %s (%d rows)", CONFLICTS_CSV, len(conflicts))

    # scorecard
    score_rows = []
    # aspect tag positive rates
    def aspect_tag_rate(a, polarity):
        col = f"ta__{a}"
        if col not in tags_df.columns:
            return None
        vals = tags_df[col]
        mentioned = (vals != "none").sum()
        if mentioned == 0:
            return None
        return float((vals == polarity).sum() / mentioned)

    for airline, g in df.groupby("airline_name"):
        if len(g) < 15:
            continue
        idx = g.index
        # aspect star means
        aspect_vals = {
            a: (float(g[info["rating_col"]].dropna().mean())
                if info["rating_col"] in g.columns else None)
            for a, info in ASPECTS.items()
        }
        # recommended rate
        rec_rate = float(g["recommended_binary"].mean() * 100) if "recommended_binary" in g.columns else 0.0
        arr = float(g["overall_rating"].mean()) if "overall_rating" in g else 0.0
        # text aspect pos rates
        text_pos = {a: aspect_tag_rate(a, "good") for a in ASPECTS}
        text_neg = {a: aspect_tag_rate(a, "bad") for a in ASPECTS}
        top_pos_aspect = max(text_pos, key=lambda a: (text_pos[a] if text_pos[a] is not None else -1))
        top_neg_aspect = max(text_neg, key=lambda a: (text_neg[a] if text_neg[a] is not None else -1))
        # PSI (preliminary; Phase 4 refines with trends)
        norm_rating = (arr - 1) / 9.0 * 100
        aspect_mean = np.nanmean([v for v in aspect_vals.values() if v is not None])
        norm_aspect = (aspect_mean - 1) / 4.0 * 100 if not np.isnan(aspect_mean) else 50.0
        psi = 0.4 * rec_rate + 0.4 * norm_rating + 0.2 * norm_aspect
        row = {
            "airline_name": airline,
            "reviews": int(len(g)),
            "recommended_rate_pct": round(rec_rate, 2),
            "avg_overall_rating": round(arr, 3),
            "top_praised_aspect": top_pos_aspect if (text_pos[top_pos_aspect] is not None) else "",
            "top_criticism_aspect": top_neg_aspect if (text_neg[top_neg_aspect] is not None) else "",
            "psi_score": round(float(psi), 2),
        }
        for a, info in ASPECTS.items():
            row[f"{a}_stars"] = round(aspect_vals[a], 3) if aspect_vals[a] is not None else None
            row[f"{a}_text_pos_rate"] = round(text_pos[a] * 100, 2) if text_pos[a] is not None else None
        score_rows.append(row)
    scorecard = pd.DataFrame(score_rows).sort_values("psi_score", ascending=False).reset_index(drop=True)
    scorecard.to_csv(SCORECARD_CSV, index=False)
    logger.info("Saved scorecard -> %s (%d airlines)", SCORECARD_CSV, len(scorecard))

    # lexicons
    save_json(
        {"aspects": {k: {"seeds": v["seeds"]} for k, v in ASPECTS.items()},
         "pol_pos": sorted(POL_POS), "pol_neg": sorted(POL_NEG),
         "agreement_note": "predicted sentiment vs Skytrax per-aspect star ratings"},
        TAGGER_LEXICONS_JSON,
    )
    return 0


def agree(pred_val, star_val):
    if star_val is None or (isinstance(star_val, float) and (star_val != star_val)):
        return None
    if star_val >= 4:
        truth = "good"
    elif star_val <= 2:
        truth = "bad"
    else:
        return None
    if pred_val == "good" and truth == "good":
        return True
    if pred_val == "bad" and truth == "bad":
        return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 Step 2: aspect tagging + scorecard.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv if argv is not None else [])
    return run(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
