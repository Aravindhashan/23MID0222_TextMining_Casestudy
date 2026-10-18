"""
pipeline.classify
----------------
Phase 3 Step 1 — Opinion classification on `recommended` (yes/no).

- Train/test split (80/20, stratified) + 5-fold CV (accuracy, F1, ROC-AUC)
- Models: MultinomialNB, LogisticRegression, LinearSVC, RandomForest
- Best model (by CV F1): feature importances, test predictions/confidence saved
- LIME explanations for 5 reviews
- Misclassification error analysis (categorised)
- Also rebuilds a sklearn Pipeline for LIME's text -> predict interface and
  fits the final (best) model on all available data for dashboard use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
except Exception:
    pass

from .common import (
    BEST_CLASSIFIER_PKL,
    CLASSIFIER_TRAINTEST_CSV,
    DTM_PKL,
    ERRORS_CSV,
    FEATURES_META_JSON,
    LIME_SAMPLES_JSON,
    METRICS_JSON,
    ROC_CURVES_CSV,
    logger,
    load_json,
    save_json,
    ensure_dirs,
    MODELS_DIR,
)


def _jsonable(x):
    import numpy as np
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def load():
    df = pd.read_csv(
        __import__("pipeline.common", fromlist=["CLEAN_CSV"]).CLEAN_CSV,
        low_memory=False)
    df = df.dropna(subset=["content_processed", "recommended_binary"]).reset_index(drop=True)
    store = joblib.load(DTM_PKL)
    vect, X = store
    y = df["recommended_binary"].astype(int).values
    return df, vect, X, y


def evaluate(model, X, y, cv):
    scoring = ["accuracy", "f1", "roc_auc"]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring,
                         n_jobs=1 if "SVM" in type(model).__name__ else -1)
    out = {}
    for m in scoring:
        vals = res[f"test_{m}"]
        out[f"cv_{m}_mean"] = float(np.mean(vals))
        out[f"cv_{m}_std"] = float(np.std(vals))
    return out


def roc_curve_data(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    # downsample to 200 points
    stride = max(1, len(fpr) // 200)
    idx = list(range(0, len(fpr), stride))
    if idx[-1] != len(fpr) - 1:
        idx.append(len(fpr) - 1)
    return [(float(fpr[i]), float(tpr[i])) for i in idx]


def categorise_error(text, proc):
    if "not_" in proc:
        return "negation-driven mismatch"
    if any(w in proc for w in ("thanks", "great job", "wonderful", "terrible",
                               "worst", "sarcasm")):
        return "possible sarcasm"
    words = set(proc.split())
    pos = words & {"good", "great", "excellent", "friendly", "nice", "comfortable"}
    neg = words & {"bad", "worst", "rude", "delay", "cancelled", "lost"}
    if pos and neg:
        return "mixed opinion"
    return "nuance/ambiguity"


def run(force=False):
    ensure_dirs()
    df, vect, X, y = load()
    text = df["content_processed"].astype(str).tolist()

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

    tractext = [text[i] for i in idx_te]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_defs = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="saga", n_jobs=-1, C=1.0),
        "LinearSVC": CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
    }

    results = {}
    roc_rows = []
    trained_test = {}
    for name, m in model_defs.items():
        logger.info("Training %s ...", name)
        cvres = evaluate(m, X, y, cv)
        # fit on train
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        # score
        if hasattr(m, "predict_proba"):
            scr = m.predict_proba(X_te)[:, 1]
        else:
            scr = m.decision_function(X_te)
        test_metrics = {
            "test_accuracy": float(accuracy_score(y_te, preds)),
            "test_precision": float(precision_score(y_te, preds, zero_division=0)),
            "test_recall": float(recall_score(y_te, preds, zero_division=0)),
            "test_f1": float(f1_score(y_te, preds, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_te, scr)),
        }
        cm = confusion_matrix(y_te, preds).tolist()
        res = {"model": name, **cvres, **test_metrics, "confusion_matrix": cm}
        # confusion matrix TN/FP/FN/TP
        try:
            tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
            res["tn"], res["fp"], res["fn"], res["tp"] = int(tn), int(fp), int(fn), int(tp)
        except Exception:
            pass
        results[name] = res
        trained_test[name] = (m, scr)
        # ROC
        for fpr, tpr in roc_curve_data(y_te, scr):
            roc_rows.append({"model": name, "fpr": fpr, "tpr": tpr})

    # pick best by cv f1
    best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])
    logger.info("Best model by CV F1: %s", best_name)

    best_model = model_defs[best_name]
    best_model.fit(X, y)  # final on all data
    # feature importances for LR (coef) regardless
    feature_names = np.asarray(vect.get_feature_names_out())
    importances = {}
    try:
        if best_name == "LogisticRegression":
            coefs = best_model.coef_.ravel()
            top_pos = np.argsort(coefs)[::-1][:20]
            top_neg = np.argsort(coefs)[:20]
            importances = {
                "top_positive": [{"term": str(feature_names[i]), "coef": round(float(coefs[i]), 5)} for i in top_pos],
                "top_negative": [{"term": str(feature_names[i]), "coef": round(float(coefs[i]), 5)} for i in top_neg],
            }
        elif best_name in ("MultinomialNB",):
            coefs = best_model.feature_log_prob_[1] - best_model.feature_log_prob_[0]
            top_pos = np.argsort(coefs)[::-1][:20]
            top_neg = np.argsort(coefs)[:20]
            importances = {
                "top_positive": [{"term": str(feature_names[i]), "coef": round(float(coefs[i]), 5)} for i in top_pos],
                "top_negative": [{"term": str(feature_names[i]), "coef": round(float(coefs[i]), 5)} for i in top_neg],
            }
        elif best_name == "RandomForest":
            coefs = best_model.feature_importances_
            order = np.argsort(coefs)[::-1][:30]
            importances = {
                "top_features": [{"term": str(feature_names[i]), "importance": round(float(coefs[i]), 5)} for i in order],
            }
        elif best_name == "LinearSVC":
            inner = best_model  # calibrated, no coef
            importances = {}
    except Exception as e:
        logger.warning("Importance extraction failed for %s: %s", best_name, e)

    pipe = Pipeline([("tfidf", None)])
    # Save TF-IDF vectorizer + best model together (for LIME/live prediction)
    bundle = {"vectorizer": vect, "model": best_model, "best_model_name": best_name,
              "importances": importances}
    joblib.dump(bundle, BEST_CLASSIFIER_PKL)
    logger.info("Saved best classifier -> %s (%s)", BEST_CLASSIFIER_PKL, best_name)

    # conf matrix / test predictions (best test predictions already exist)
    # Re-fit best model on train (single model) for LIME samples from test data.
    mtest, scrtest = trained_test[best_name]
    os.makedirs(MODELS_DIR, exist_ok=True)
    test_pred_df = pd.DataFrame({
        "review_idx": idx_te,
        "true": y_te,
        "pred": mtest.predict(X_te),
        "score": scrtest,
        "text_sample": [str(t)[:200] for t in tractext],
    })
    test_pred_df.to_csv(CLASSIFIER_TRAINTEST_CSV, index=False)

    rc_df = pd.DataFrame(roc_rows)
    rc_df.to_csv(ROC_CURVES_CSV, index=False)

    metrics = {"results": results, "best_model_name": best_name,
               "importances": importances,
               "train_size": int(len(idx_tr)), "test_size": int(len(idx_te))}
    save_json(metrics, METRICS_JSON)

    # LIME
    lime_samples = try_lime(vect, mtest, text, df, idx_te, y_te)
    save_json(lime_samples, LIME_SAMPLES_JSON)

    # Misclassification analysis
    err = build_errors(vect, mtest, tractext, y_te, idx_te, df)
    err.to_csv(ERRORS_CSV, index=False)
    logger.info("Errors: %s", len(err))

    # length-vs-rating correlation note
    corr = float(pd.Series(df["content"].astype(str).str.len().values).corr(
        pd.Series(df["overall_rating"].astype(float))))
    save_json({"review_length_corr_with_overall_rating": round(corr, 4)},
              MODELS_DIR / "length_corr.json")

    return 0


def try_lime(vect, model, text, df, idx_te, y_te):
    out = {"available": False, "samples": []}
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception as e:
        out["error"] = f"lime import failed: {e}"
        return out
    try:
        explainer = LimeTextExplainer(class_names=["not recommended", "recommended"])
        predict_fn = lambda arr: model.predict_proba(vect.transform(list(arr)))
        # choose 5 representative: 2 predicted pos, 2 neg, 1 confident wrong
        preds = model.predict(__import__("joblib").load(DTM_PKL)[0].transform(text) if False else "")
    except Exception:
        pass

    # simpler: pick 5 indices from test
    samples = []
    try:
        preds_test = model.predict(__import__("joblib").load(DTM_PKL)[0].transform([text[i] for i in idx_te]))
    except Exception as e:
        out["error"] = f"predict failed: {e}"
        return out

    # choose 2 correct positives, 2 correct negatives, 1 misclassified
    correct_pos = [i for i in range(len(idx_te)) if preds_test[i] == 1 and y_te[i] == 1]
    correct_neg = [i for i in range(len(idx_te)) if preds_test[i] == 0 and y_te[i] == 0]
    wrong = [i for i in range(len(idx_te)) if preds_test[i] != y_te[i]]
    pick = correct_pos[:2] + correct_neg[:2] + wrong[:1]
    # if needed fill
    pool = list(range(len(idx_te)))
    for p in pool:
        if len(pick) >= 5:
            break
        if p not in pick:
            pick.append(p)

    for p in pick[:5]:
        global_idx = int(idx_te[p])
        raw = text[global_idx]
        try:
            exp = explainer.explain_instance(raw, predict_fn, num_features=12,
                                             num_samples=200, top_labels=1)
            label = exp.top_labels[0]
            weights = exp.as_list(label=label)
            samples.append({
                "review_index": global_idx,
                "true_label": int(y_te[p]),
                "predicted_label": int(preds_test[p]),
                "label_explained": int(label),
                "words": [{"word": w, "weight": round(float(wt), 4)} for w, wt in weights],
                "text_excerpt": str(df["content"].iloc[global_idx])[:400],
            })
        except Exception as e:
            samples.append({"review_index": global_idx, "error": str(e)})
    out = {"available": True, "samples": samples}
    return out


def build_errors(vect, model, tractext, y_te, idx_te, df):
    try:
        preds = model.predict(vect.transform(tractext))
        prob = model.predict_proba(vect.transform(tractext))[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_te))
    except Exception:
        preds = model.predict(__import__("joblib").load(DTM_PKL)[0].transform(tractext))
        prob = np.zeros(len(y_te))
    rows = []
    for i in range(len(y_te)):
        if preds[i] != y_te[i]:
            proc = df["content_processed"].iloc[idx_te[i]]
            rows.append({
                "review_idx": int(idx_te[i]),
                "airline_name": str(df["airline_name"].iloc[idx_te[i]]) if "airline_name" in df else "",
                "true": int(y_te[i]),
                "pred": int(preds[i]),
                "confidence": round(float(prob[i]), 4),
                "error_category": categorise_error(str(df["content"].iloc[idx_te[i]]), str(proc)),
                "text_excerpt": str(df["content"].iloc[idx_te[i]])[:300],
            })
    # cap to 40 sorted by confidence desc
    edf = pd.DataFrame(rows)
    if not edf.empty:
        edf = edf.sort_values("confidence", ascending=False).head(40).reset_index(drop=True)
    return edf


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 Step 1: opinion classification.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv if argv is not None else [])
    return run(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
