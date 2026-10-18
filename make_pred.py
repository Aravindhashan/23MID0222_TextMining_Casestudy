import joblib, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r"C:\Users\aravi\Downloads\TextMining_CaseStudy"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

bundle = joblib.load(os.path.join(BASE, "models", "best_classifier.pkl"))
vect = bundle["vectorizer"]
model = bundle["model"]
print("model:", bundle["best_model_name"])

df = pd.read_csv(os.path.join(BASE, "data", "clean_reviews.csv"), low_memory=False)
df = df.dropna(subset=["content_processed", "recommended_binary"]).reset_index(drop=True)

# pick 6 representative reviews: 2 positive, 2 negative, 2 mixed-ish by length
# Auto-pick: 2 confident correct positives, 2 confident correct negatives, 1 misclassified
Xall = vect.transform(df["content_processed"].astype(str).tolist())
proba_all = model.predict_proba(Xall)[:,1]
pred_all = model.predict(Xall)
y = df["recommended_binary"].astype(int).values
correct_pos = [i for i in range(len(df)) if y[i]==1 and pred_all[i]==1]
correct_neg = [i for i in range(len(df)) if y[i]==0 and pred_all[i]==0]
wrong = [i for i in range(len(df)) if y[i]!=pred_all[i]]
# sort by confidence
correct_pos_sorted = sorted(correct_pos, key=lambda i: proba_all[i], reverse=True)
correct_neg_sorted = sorted(correct_neg, key=lambda i: proba_all[i])
wrong_sorted = sorted(wrong, key=lambda i: abs(proba_all[i]-0.5), reverse=True)
picks = correct_pos_sorted[:2] + correct_neg_sorted[:2] + wrong_sorted[:1]

rows = []
X = vect.transform(df.loc[picks, "content_processed"].astype(str).tolist())
proba = model.predict_proba(X)[:,1]
preds = model.predict(X)
for i, idx in enumerate(picks):
    excerpt = str(df.loc[idx, "content"])[:160].replace("\n"," ")
    rows.append({
        "review_idx": int(idx),
        "true_label": int(df.loc[idx, "recommended_binary"]),
        "predicted": int(preds[i]),
        "prob_recommended": round(float(proba[i]), 4),
        "excerpt": excerpt,
    })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, "models", "prediction_examples.csv"), index=False)
print(out[["review_idx","true_label","predicted","prob_recommended"]].to_string(index=False))

# Top contributing words for the first positive example
# coef * tfidf for that doc
feature_names = np.asarray(vect.get_feature_names_out())
coefs = model.coef_.ravel()
doc = X[0]
# get nonzero entries
indices = doc.nonzero()[1]
contrib = [(feature_names[j], float(doc[0,j]*coefs[j])) for j in indices]
contrib_sorted = sorted(contrib, key=lambda x: x[1], reverse=True)
top_pos = contrib_sorted[:10]
top_neg = sorted(contrib, key=lambda x: x[1])[:10]

fig, axes = plt.subplots(1,2, figsize=(11,5))
axes[0].barh([w for w,_ in top_pos][::-1], [v for _,v in top_pos][::-1], color='#2e7d32')
axes[0].set_title('Top words pushing "recommended" (sample review)')
axes[1].barh([w for w,_ in top_neg][::-1], [v for _,v in top_neg][::-1], color='#c62828')
axes[1].set_title('Top words pushing "not recommended" (same review)')
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_prediction_contrib.png"), dpi=120, bbox_inches='tight')
plt.close()
print("saved fig_prediction_contrib.png")

# Save a summary json for report
summary = {
    "model": bundle["best_model_name"],
    "examples": rows,
    "top_positive_contrib": [{"word":w,"contrib":round(v,4)} for w,v in top_pos],
    "top_negative_contrib": [{"word":w,"contrib":round(v,4)} for w,v in top_neg],
}
json.dump(summary, open(os.path.join(BASE, "models", "prediction_example.json"), "w"), indent=2)
print("done")