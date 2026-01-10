"""
STEP 1: EXPERIMENT PIPELINE (IDLE COMPATIBLE)
---------------------------------------------
1. Uses ABSOLUTE PATH to find the file in C:/Users/varsh/
2. Loads Data
3. Runs Split -> Summarize -> Classify
4. Saves output to C:/Users/varsh/rich_results.csv
"""

import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import pipeline
from tqdm import tqdm

# --- CONFIGURATION (HARDCODED FOR YOUR PC) ---
# We use 'r' before the string to handle backslashes correctly in Windows
FILE_PATH = r"C:\Users\varsh\depression_dataset_reddit_cleaned.csv"
OUTPUT_PATH = r"C:\Users\varsh\rich_results.csv"

SUMM_MODEL = "google/pegasus-xsum"
CLF_MODEL = "paulagarciaserrano/roberta-depression-detection"
BATCH_SIZE = 16
SEED = 42

# --- SETUP ---
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Running on {'GPU' if device==0 else 'CPU'}...")
print(f"📂 Looking for file at: {FILE_PATH}")

# 1. LOAD DATA
if os.path.exists(FILE_PATH):
    print(f"✅ Found file!")
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"   Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        exit()
else:
    print(f"❌ FILE NOT FOUND AT: {FILE_PATH}")
    print("👉 Please make sure 'depression_dataset_reddit_cleaned.csv' is inside 'C:\\Users\\varsh\\'")
    exit()

# 2. PREPARE DATA
print("🧹 Cleaning Data...")
if 'clean_text' in df.columns: df.rename(columns={'clean_text': 'text'}, inplace=True)
if 'is_depression' in df.columns: df.rename(columns={'is_depression': 'label'}, inplace=True)

df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)

# 3. SPLIT
try:
    _, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['label'])
except:
    _, test_df = train_test_split(df, test_size=0.20, random_state=SEED)

print(f"📊 Test Set Size: {len(test_df)} samples")

# 4. LOAD MODELS
print("⏳ Loading AI Models (This takes a minute)...")
dtype_setting = torch.float16 if device==0 else torch.float32
try:
    summ_pipe = pipeline("summarization", model=SUMM_MODEL, device=device, torch_dtype=dtype_setting)
    clf_pipe = pipeline("text-classification", model=CLF_MODEL, device=device, top_k=None, truncation=True)
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit()

# 5. HELPER FUNCTIONS
def get_prob(pred):
    scores = {p['label']: p['score'] for p in pred}
    return scores.get('severe', 0) + scores.get('moderate', 0) + scores.get('depression', 0) + scores.get('1', 0)

def lead_3(text):
    return '. '.join(str(text).split('.')[:3])

# 6. RUN EXPERIMENT
results = []
texts = test_df['text'].tolist()
labels = test_df['label'].tolist()

print("🔬 Running Safety Audit...")
# Using a simple loop instead of tqdm if tqdm behaves weirdly in IDLE
total_batches = len(texts) // BATCH_SIZE + 1

for i in range(0, len(texts), BATCH_SIZE):
    print(f"   Processing Batch {i // BATCH_SIZE + 1}/{total_batches}...", end="\r")
    batch_txt = texts[i:i+BATCH_SIZE]
    batch_lbl = labels[i:i+BATCH_SIZE]
    
    # A. Summarize
    try:
        summ_out = summ_pipe(batch_txt, max_length=128, min_length=30, truncation=True)
        batch_summ = [x['summary_text'] for x in summ_out]
    except:
        batch_summ = batch_txt 
        
    # B. Baseline
    batch_lead3 = [lead_3(t) for t in batch_txt]
    
    # C. Classify
    try:
        pred_orig = clf_pipe(batch_txt)
        pred_summ = clf_pipe(batch_summ)
        pred_lead3 = clf_pipe(batch_lead3)
        
        # D. Save Results
        for j in range(len(batch_txt)):
            results.append({
                "original_text": batch_txt[j],
                "summary_text": batch_summ[j],
                "lead3_text": batch_lead3[j],
                "true_label": batch_lbl[j],
                "prob_orig": get_prob(pred_orig[j]),
                "prob_summ": get_prob(pred_summ[j]),
                "prob_lead3": get_prob(pred_lead3[j])
            })
    except Exception as e:
        print(f"\nSkipping batch {i} due to error: {e}")
        continue

# 7. EXPORT
print("\n💾 Saving Results...")
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ DONE! Saved results to: {OUTPUT_PATH}")
