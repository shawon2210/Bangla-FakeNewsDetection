"""
Master script: Run all enhanced experiments sequentially and generate
the final comparison table.

Experiments:
  1. BanglaBERT Enhanced Text-Only Baseline
  2. EfficientNet-B3 Image-Only Ablation
  3. Enhanced Multimodal v3 (proposed)

Then generates comparison_table.md with all results.
"""
import subprocess, os, sys, json

BASE_DIR = "/mnt/d/all files/ThesisP/BanglaFakeNewsProject/BanglaFakeNewsProject"
SAVE_DIR = os.path.join(BASE_DIR, "outputs_v3")
os.makedirs(SAVE_DIR, exist_ok=True)

experiments = [
    ("exp1_enhanced.py", "results_exp1_enhanced.json"),
    ("exp3_enhanced.py", "results_exp3_enhanced.json"),
    ("train_v3_enhanced.py", "results_enhanced_v3.json"),
]

for script, result_file in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script)],
        cwd=BASE_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
    else:
        print(f"\n[DONE] {script}")

# Collect results
print(f"\n{'='*60}")
print("ALL EXPERIMENTS COMPLETE — GENERATING COMPARISON TABLE")
print(f"{'='*60}\n")

all_results = []
for _, result_file in experiments:
    path = os.path.join(SAVE_DIR, result_file)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        all_results.append(data)
        print(f"\n{data['experiment']}:")
        print(f"  Accuracy:  {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")
        print(f"  Precision: {data['precision_weighted']:.4f}")
        print(f"  Recall:    {data['recall_weighted']:.4f}")
        print(f"  F1:        {data['f1_weighted']:.4f}")
    else:
        print(f"\n[MISSING] {result_file}")

# Save combined summary
with open(os.path.join(SAVE_DIR, "all_enhanced_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# Generate markdown comparison table
table = """# Bangla Fake News Detection — Results Comparison

## Previous State-of-the-Art (from literature)

| Method | Modality | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Remarks |
|--------|----------|-------------|---------------|------------|--------------|---------|
| FastText [1] | Text | 89.00 | 88.73 | 89.00 | 88.85 | Embedding-based method |
| CNN [4] | Text | 83.40 | 82.91 | 83.40 | 83.12 | Traditional ML (TF-IDF) |
| CNN-LSTM [3] | Text | 75.00 | 74.23 | 75.00 | 74.58 | Multichannel CNN-LSTM |

## Our Experiments — Previous (v1/v2)

| Method | Modality | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Remarks |
|--------|----------|-------------|---------------|------------|--------------|---------|
| BanglaBERT Baseline (Ours, v1) | Text | 74.27 | 72.89 | 74.27 | 72.14 | Transformer baseline |
| XLM-RoBERTa Baseline (Ours, v1) | Text | 71.53 | 70.84 | 71.53 | 70.12 | Multilingual transformer |
| ResNet50 Ablation (Ours, v1) | Image | 58.33 | 55.12 | 58.33 | 56.43 | Image-only ablation |
| Proposed Multimodal (Ours, v2) | Text + Image | 64.48 | 65.42 | 64.48 | 63.91 | BanglaBERT + ResNet50 |

## Our Experiments — Enhanced (v3)

| Method | Modality | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Remarks |
|--------|----------|-------------|---------------|------------|--------------|---------|
"""

for r in all_results:
    name = r["experiment"]
    mod = "Text" if "Text" in name or "Baseline" in name else ("Image" if "Image" in name else "Text + Image")
    table += f"| {name} | {mod} | {r['accuracy_pct']:.2f} | {r['precision_weighted']:.4f} | {r['recall_weighted']:.4f} | {r['f1_weighted']:.4f} | Enhanced v3 |\n"

table += """
## Key Improvements in v3

1. **Text encoder**: Uses both headline + description (concatenated, 256 tokens)
2. **Text encoder**: Unfreezes more layers (4 instead of 6), uses weighted layer pooling
3. **Image encoder**: EfficientNet-B3 replaces ResNet50, uses avg+max pooling
4. **Image augmentation**: RandomResizedCrop, ColorJitter, RandomErasing
5. **Fusion**: Bilinear fusion + gated multimodal fusion (learns modality trust)
6. **Regularization**: Label smoothing (0.1), multi-sample dropout, gradient accumulation
7. **Training**: Cosine annealing with warm restarts, 30 epochs, mixed precision
"""

with open(os.path.join(SAVE_DIR, "comparison_table.md"), "w") as f:
    f.write(table)

print(f"\nComparison table saved to {SAVE_DIR}/comparison_table.md")
print(f"All results saved to {SAVE_DIR}/all_enhanced_results.json")
