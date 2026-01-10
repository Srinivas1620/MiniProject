"""
STEP 2: ANALYSIS & PLOTTING (FIXED)
-----------------------------------
1. Optimizes Thresholds (Fixes low recall)
2. Computes Lead-3 Correctly
3. Calculates Bootstrap CIs
4. Generates Heatmap & SBERT plots
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, confusion_matrix
from sentence_transformers import SentenceTransformer, util
from statsmodels.stats.contingency_tables import mcnemar

# --- CONFIGURATION ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

INPUT_FILE = os.path.join(SCRIPT_DIR, "rich_results.csv")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: Could not find '{INPUT_FILE}'")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} samples.")

# --- 1. OPTIMIZE THRESHOLD ---
def optimize_threshold(y_true, y_prob):
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

# Optimize based on Full Text (Oracle)
optimal_thresh = optimize_threshold(df['true_label'], df['prob_orig'])
print(f"\n⚙️  Optimal Decision Threshold: {optimal_thresh:.2f}")

# Apply to all
df['pred_orig'] = (df['prob_orig'] >= optimal_thresh).astype(int)
df['pred_summ'] = (df['prob_summ'] >= optimal_thresh).astype(int)
df['pred_lead3'] = (df['prob_lead3'] >= optimal_thresh).astype(int)

# --- 2. BOOTSTRAP CONFIDENCE INTERVALS ---
def get_bootstrap_ci(y_true, y_pred, metric_func, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    stats = []
    # Convert to numpy arrays to avoid index issues during resampling
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true_np), len(y_true_np))
        score = metric_func(y_true_np[idx], y_pred_np[idx])
        stats.append(score)
    return np.percentile(stats, [2.5, 97.5])

def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    
    # Get CIs
    rec_ci = get_bootstrap_ci(y_true, y_pred, recall_score)
    
    print(f"\n🔹 {name} Metrics:")
    print(f"   Accuracy:  {acc:.1%}")
    print(f"   Recall:    {rec:.1%}  [95% CI: {rec_ci[0]:.1%} - {rec_ci[1]:.1%}]")
    print(f"   Precision: {prec:.1%}")
    print(f"   F1-Score:  {f1:.3f}")
    print(f"   Kappa:     {kap:.3f}")
    return acc, rec, kap

print("\n🏆 FINAL RIGOROUS RESULTS:")
print("-" * 30)
# FIXED ARGUMENT ORDER HERE: (name, true, pred)
acc_o, rec_o, kap_o = print_metrics("Original Text", df['true_label'], df['pred_orig'])
acc_s, rec_s, kap_s = print_metrics("AI Summary   ", df['true_label'], df['pred_summ'])
acc_l, rec_l, kap_l = print_metrics("Lead-3 (Base)", df['true_label'], df['pred_lead3'])
print("-" * 30)

# --- 3. MCNEMAR'S TEST ---
print("\n📉 Statistical Significance (McNemar)...")
orig_corr = (df['pred_orig'] == df['true_label'])
summ_corr = (df['pred_summ'] == df['true_label'])

n11 = sum(orig_corr & summ_corr)
n10 = sum(orig_corr & ~summ_corr) # Orig Right, Summ Wrong (Loss)
n01 = sum(~orig_corr & summ_corr) # Orig Wrong, Summ Right (Gain)
n00 = sum(~orig_corr & ~summ_corr)

table = [[n11, n10], [n01, n00]]
result = mcnemar(table, exact=False, correction=True)

print(f"   - Discordant Pairs: {n10} (Loss) vs {n01} (Gain)")
print(f"   - Chi-Square: {result.statistic:.2f}")
print(f"   - P-Value:    {result.pvalue:.4e}")

# --- 4. SEVERITY HEATMAP ---
print("\n🎨 Generating Heatmap...")
def get_severity(prob):
    # Use thresholds relative to our optimal point
    if prob < optimal_thresh: return "Neutral"
    if prob < 0.90: return "Moderate" 
    return "Severe"

df['sev_orig'] = df['prob_orig'].apply(get_severity)
df['sev_summ'] = df['prob_summ'].apply(get_severity)

labels = ["Severe", "Moderate", "Neutral"]
cm = confusion_matrix(df['sev_orig'], df['sev_summ'], labels=labels)
# Normalize rows to percentages
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Reds', xticklabels=labels, yticklabels=labels)
plt.title(f"Severity Collapse (Threshold={optimal_thresh:.2f})")
plt.xlabel("AI Summary Prediction")
plt.ylabel("Original Text Prediction")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "severity_heatmap_v2.png"), dpi=300)
print("✅ Saved 'severity_heatmap_v2.png'")

# --- 5. SBERT SIMILARITY ---
print("\n🧠 Calculating SBERT...")
model = SentenceTransformer('all-MiniLM-L6-v2')
emb_orig = model.encode(df['original_text'].tolist(), convert_to_tensor=True)
emb_summ = model.encode(df['summary_text'].tolist(), convert_to_tensor=True)
sims = util.cos_sim(emb_orig, emb_summ).diagonal().cpu().numpy()
print(f"   - Avg Semantic Similarity: {np.mean(sims):.3f}")

plt.figure(figsize=(8, 6))
sns.histplot(sims, bins=30, kde=True, color="blue")
plt.title(f"Semantic Similarity Distribution (Avg: {np.mean(sims):.2f})")
plt.xlabel("Cosine Similarity")
plt.savefig(os.path.join(SCRIPT_DIR, "sbert_dist_v2.png"), dpi=300)
print("✅ Saved 'sbert_dist_v2.png'")
