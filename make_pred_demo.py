import json, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = r"C:\Users\aravi\Downloads\TextMining_CaseStudy"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

summary = json.load(open(os.path.join(BASE, "models", "prediction_example.json")))
preds = summary["examples"]
# pick the first confident positive example for the demo card
ex = preds[0]
df = pd.read_csv(os.path.join(BASE, "data", "clean_reviews.csv"), low_memory=False)
excerpt = str(df.loc[ex["review_idx"], "content"])[:320].replace("\n"," ")

fig = plt.figure(figsize=(8.5, 6))
ax = fig.add_axes([0.05,0.05,0.9,0.9])
ax.axis('off')

# card background
card = FancyBboxPatch((0.02,0.02),0.96,0.96, boxstyle="round,pad=0.02,rounding_size=0.02",
                      facecolor='#f5f7fa', edgecolor='#1565c0', linewidth=2)
ax.add_patch(card)

ax.text(0.06,0.92, "Dashboard: Opinion Classification → 'Try it yourself'", fontsize=13, fontweight='bold', color='#1565c0')
ax.text(0.06,0.86, "Input review:", fontsize=11, fontweight='bold')
# wrap excerpt
import textwrap
wrapped = textwrap.fill(excerpt, width=95)
ax.text(0.06,0.82, wrapped, fontsize=9, va='top', family='monospace')

ax.text(0.06,0.58, "Prediction:", fontsize=11, fontweight='bold')
label = "RECOMMENDED (yes)" if ex["predicted"]==1 else "NOT RECOMMENDED (no)"
color = '#2e7d32' if ex["predicted"]==1 else '#c62828'
ax.text(0.22,0.58, label, fontsize=12, fontweight='bold', color=color)
ax.text(0.06,0.53, f"Probability (recommended): {ex['prob_recommended']:.4f}", fontsize=10)

# probability bar
bar_x, bar_y, bar_w, bar_h = 0.06, 0.47, 0.88, 0.04
ax.add_patch(FancyBboxPatch((bar_x,bar_y),bar_w,bar_h, boxstyle="round,pad=0.005", facecolor='#ddd', edgecolor='none'))
ax.add_patch(FancyBboxPatch((bar_x,bar_y),bar_w*ex['prob_recommended'],bar_h, boxstyle="round,pad=0.005", facecolor=color, edgecolor='none'))
ax.text(bar_x+bar_w*ex['prob_recommended']+0.01, bar_y+0.005, f"{ex['prob_recommended']*100:.1f}%", fontsize=9, color=color)

ax.text(0.06,0.40, "Top contributing words:", fontsize=11, fontweight='bold')
tp = summary["top_positive_contrib"][:6]
tn = summary["top_negative_contrib"][:6]
y0 = 0.36
for i,d in enumerate(tp):
    ax.text(0.06, y0 - i*0.035, f"+  {d['word']:<18}  {d['contrib']:+.4f}", fontsize=9, color='#2e7d32', family='monospace')
for i,d in enumerate(tn):
    ax.text(0.52, y0 - i*0.035, f"-  {d['word']:<18}  {d['contrib']:+.4f}", fontsize=9, color='#c62828', family='monospace')

ax.text(0.06,0.06, "Model: Logistic Regression (CV F1 0.896) on TF-IDF uni/bigrams + engineered features.", fontsize=8, fontstyle='italic', color='#555')

plt.savefig(os.path.join(FIG, "fig_prediction_demo.png"), dpi=130, bbox_inches='tight')
plt.close()
print("saved fig_prediction_demo.png")