"""
Master script: Runs all 4 experiments sequentially.
"""
import subprocess, os, sys, json

BASE_DIR = r"d:\all files\ThesisP\BanglaFakeNewsProject\BanglaFakeNewsProject"
experiments = [
    ("exp1_banglabert_baseline.py", "results_banglabert_baseline.json"),
    ("exp2_xlmroberta_baseline.py", "results_xlmroberta_baseline.json"),
    ("exp3_resnet50_ablation.py", "results_resnet50_ablation.json"),
    ("exp4_banglabert_ablation.py", "results_banglabert_ablation.json"),
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

# Collect all results
print(f"\n{'='*60}")
print("ALL EXPERIMENTS COMPLETE — SUMMARY")
print(f"{'='*60}")

summary = []
for _, result_file in experiments:
    path = os.path.join(BASE_DIR, result_file)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        summary.append(data)
        print(f"\n{data['experiment']}:")
        print(f"  Accuracy:  {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")
        print(f"  Precision: {data['precision']:.4f}")
        print(f"  Recall:    {data['recall']:.4f}")
        print(f"  F1:        {data['f1']:.4f}")
    else:
        print(f"\n[MISSING] {result_file}")

# Save combined summary
with open(os.path.join(BASE_DIR, "all_experiments_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nCombined summary saved to all_experiments_summary.json")
