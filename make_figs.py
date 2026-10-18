import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json, os

BASE = r"C:\Users\aravi\Downloads\TextMining_CaseStudy"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=120, bbox_inches='tight')
    plt.close()
    print("saved", name)

# 1. cleaning
cs = json.load(open(os.path.join(BASE,"data","cleaning_stats.json")))
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].bar(['Raw rows','Clean rows'], [cs['rows_in'], cs['rows_out']], color=['#9bbcd6','#2e7d32'])
axes[0].set_title('Rows before/after cleaning')
axes[0].set_ylabel('reviews')
for i,v in enumerate([cs['rows_in'], cs['rows_out']]):
    axes[0].text(i, v+200, f"{v:,}", ha='center')
steps = cs['steps']
labels = ['Placeholders->NaN','Dropped duplicates','Implausible dates->NaT','Date parse failures']
vals = [steps['rating_placeholders_to_nan'], steps['dropped_duplicates'], steps['implausible_dates_to_nat'], steps['date_parse_failures']]
axes[1].barh(labels, vals, color='#c62828')
axes[1].set_title('Cleaning step counts')
axes[1].set_xscale('log')
save('fig_cleaning.png')

# 2. top terms
fm = json.load(open(os.path.join(BASE,"data","features","features_meta.json")))
pos = fm['top_terms']['positive'][:15]
neg = fm['top_terms']['negative'][:15]
fig, axes = plt.subplots(1,2, figsize=(12,6))
axes[0].barh([t['term'] for t in pos][::-1], [t['weight'] for t in pos][::-1], color='#2e7d32')
axes[0].set_title('Top TF-IDF terms (positive reviews)')
axes[1].barh([t['term'] for t in neg][::-1], [t['weight'] for t in neg][::-1], color='#c62828')
axes[1].set_title('Top TF-IDF terms (negative reviews)')
save('fig_top_terms.png')

# 3. LDA topic distribution
lp = json.load(open(os.path.join(BASE,"data","lda_perplexity.json")))
tt = json.load(open(os.path.join(BASE,"data","topic_terms.json")))
topics = [tt[str(i)]['label'] for i in range(10)]
dist = lp['topic_distribution']
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(range(10), dist, color='#1565c0')
ax.set_xticks(range(10))
ax.set_xticklabels([f"T{i}\n{topics[i][:14]}" for i in range(10)], rotation=45, ha='right', fontsize=8)
ax.set_title('LDA: reviews per dominant topic (10 topics)')
ax.set_ylabel('reviews')
save('fig_lda_topics.png')

# 4. t-SNE
ts = pd.read_csv(os.path.join(BASE,"data","tsne_coords.csv"))
fig, ax = plt.subplots(figsize=(8,7))
sc = ax.scatter(ts['tsne_1'], ts['tsne_2'], c=ts['recommended_binary'], cmap='RdYlGn', s=4, alpha=0.5)
ax.set_title('t-SNE of reviews (colored by recommended)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.colorbar(sc, ax=ax, label='recommended (0/1)')
save('fig_tsne.png')

# 5. model comparison
m = json.load(open(os.path.join(BASE,"models","metrics.json")))
names = list(m['results'].keys())
f1 = [m['results'][n]['test_f1'] for n in names]
auc = [m['results'][n]['test_roc_auc'] for n in names]
fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(len(names))
w=0.35
ax.bar(x-w/2, f1, w, label='Test F1', color='#1565c0')
ax.bar(x+w/2, auc, w, label='Test ROC-AUC', color='#2e7d32')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
ax.set_ylim(0.8,1.0); ax.legend(); ax.set_title('Classifier comparison')
save('fig_model_compare.png')

# 6. confusion matrix best
cm = m['results']['LogisticRegression']['confusion_matrix']
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i][j], ha='center', va='center', color='black')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred 0','Pred 1']); ax.set_yticklabels(['True 0','True 1'])
ax.set_title('Logistic Regression confusion matrix')
save('fig_confusion.png')

# 7. aspect tagger agreement
tv = json.load(open(os.path.join(BASE,"data","tagger_validation.json")))
aspects = [tv[k]['aspect_label'] for k in tv]
a = [tv[k]['tagger_A']['agreement_pct'] for k in tv]
b = [tv[k]['tagger_B']['agreement_pct'] for k in tv]
fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(aspects))
w=0.35
ax.bar(x-w/2, a, w, label='Tagger A (lexicon)', color='#1565c0')
ax.bar(x+w/2, b, w, label='Tagger B (spaCy)', color='#2e7d32')
ax.set_xticks(x); ax.set_xticklabels(aspects, rotation=25, ha='right')
ax.set_ylabel('agreement %'); ax.set_title('Aspect tagger agreement vs Skytrax ratings')
ax.legend()
save('fig_tagger_agreement.png')

# 8. PSI top 10
psi = pd.read_csv(os.path.join(BASE,"data","psi_ranking.csv"))
top = psi.sort_values('psi_score', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(top['airline_name'][::-1], top['psi_score'][::-1], color='#6a1b9a')
ax.set_title('Top 10 airlines by PSI')
ax.set_xlabel('PSI score')
save('fig_psi_top10.png')

# 9. trends
tr = pd.read_csv(os.path.join(BASE,"data","trends.csv"))
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(tr['year'], tr['recommend_rate']*100, marker='o', label='Recommend rate %', color='#1565c0')
ax.plot(tr['year'], tr['overall_rating'], marker='s', label='Mean overall rating', color='#2e7d32')
ax.set_title('Yearly recommend rate & overall rating')
ax.set_xlabel('year'); ax.legend()
save('fig_trends.png')

# 10. cabin comparison
cc = pd.read_csv(os.path.join(BASE,"data","cabin_country.csv"))
aspects_cols = ['seat_comfort_stars','cabin_staff_stars','food_beverages_stars','inflight_entertainment_stars','value_money_stars']
labels = ['Seat','Crew','Food','IFE','Value']
fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(len(labels))
w=0.25
for i,cabin in enumerate(['Economy','Business Class','First Class']):
    row = cc[cc['cabin_flown']==cabin].iloc[0]
    ax.bar(x + (i-1)*w, [row[c] for c in aspects_cols], w, label=cabin)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('mean stars (1-5)'); ax.set_title('Cabin class comparison by aspect')
ax.legend()
save('fig_cabin.png')

# 11. aspect network node mentions
an = json.load(open(os.path.join(BASE,"data","aspect_network.json")))
nodes = an['nodes']
fig, ax = plt.subplots(figsize=(8,5))
ax.barh([n['label'] for n in nodes][::-1], [n['mentions'] for n in nodes][::-1], color='#00838f')
ax.set_title('Aspect mention counts (co-occurrence network nodes)')
save('fig_aspect_mentions.png')

# 12. nps vs vader
nps = pd.read_csv(os.path.join(BASE,"data","nps.csv"))
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(nps['vader_mean'], nps['nps'], s=12, alpha=0.5, color='#6a1b9a')
ax.set_xlabel('Mean VADER sentiment'); ax.set_ylabel('NPS')
ax.set_title('NPS vs text sentiment (per airline)')
save('fig_nps.png')

print("ALL DONE")