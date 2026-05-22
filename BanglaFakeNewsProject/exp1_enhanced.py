"""
Experiment 1 (Enhanced): BanglaBERT Text-Only Baseline v3.
Uses headline + description, enhanced text encoder, 30 epochs, cosine LR.
"""
import os, sys, time, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm

from model_v3_enhanced import TextOnlyEnhanced
from data_v3_enhanced import TextOnlyDataset, tokenizer, MAX_LENGTH
import pandas as pd

BASE_DIR = "/mnt/d/all files/ThesisP/BanglaFakeNewsProject/BanglaFakeNewsProject"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.join(BASE_DIR, "outputs_v3")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "csebuetnlp/banglabert"
BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
PATIENCE = 10
USE_AMP = torch.cuda.is_available()

print("=" * 60)
print("EXPERIMENT 1 (Enhanced): BanglaBERT Text-Only Baseline v3")
print("=" * 60)
print(f"Device: {DEVICE}")

# Data
train_df = pd.read_csv(os.path.join(BASE_DIR, "Train.csv"), low_memory=False)
val_df = pd.read_csv(os.path.join(BASE_DIR, "Validation.csv"), low_memory=False)
test_df = pd.read_csv(os.path.join(BASE_DIR, "Test.csv"), low_memory=False)
print(f"Data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

train_ds = TextOnlyDataset(train_df, tokenizer, MAX_LENGTH)
val_ds = TextOnlyDataset(val_df, tokenizer, MAX_LENGTH)
test_ds = TextOnlyDataset(test_df, tokenizer, MAX_LENGTH)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Model
model = TextOnlyEnhanced(MODEL_NAME, num_classes=2, freeze_text_layers=4).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,} total, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

# Optimizer
decay = [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n and "norm" not in n]
no_decay = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n)]
optimizer = optim.AdamW([
    {"params": decay, "lr": LR, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay, "lr": LR, "weight_decay": 0.0},
])

total_steps = len(train_dl) * EPOCHS
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = GradScaler() if USE_AMP else None

best_val, best_state, patience_ctr = 0, None, 0
start = time.time()

for ep in range(EPOCHS):
    # Train
    model.train()
    tl, tp, tlb, tprobs = 0, [], [], []
    for b in tqdm(train_dl, desc=f"  E{ep+1}/{EPOCHS}", leave=False):
        optimizer.zero_grad()
        if USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
                loss = criterion(logits, b["label"].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            loss = criterion(logits, b["label"].to(DEVICE))
            loss.backward()
            optimizer.step()
        scheduler.step()
        tl += loss.item()
        tp.extend(torch.argmax(logits, 1).cpu().numpy())
        tlb.extend(b["label"].numpy())
        tprobs.extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
    ta = accuracy_score(tlb, tp)

    # Val
    model.eval()
    vl, vp, vlb, vprobs = 0, [], [], []
    with torch.no_grad():
        for b in val_dl:
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            vl += criterion(logits, b["label"].to(DEVICE)).item()
            vp.extend(torch.argmax(logits, 1).cpu().numpy())
            vlb.extend(b["label"].numpy())
            vprobs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    va = accuracy_score(vlb, vp)
    _, _, vf1, _ = precision_recall_fscore_support(vlb, vp, average="weighted")

    print(f"  E{ep+1}: TrainAcc={ta:.4f} ValAcc={va:.4f} ValF1={vf1:.4f}")

    if va > best_val:
        best_val = va
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
        print(f"    -> New best (ValAcc={va:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"    -> Early stopping at epoch {ep+1}")
            break

print(f"Done in {(time.time()-start)/60:.1f}min. BestVal={best_val:.4f}")

# Test
model.load_state_dict(best_state)
model.eval()
tep, tel, teprobs = [], [], []
with torch.no_grad():
    for b in test_dl:
        logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        tep.extend(torch.argmax(logits, 1).cpu().numpy())
        tel.extend(b["label"].numpy())
        teprobs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())

acc = accuracy_score(tel, tep)
p_per, r_per, f1_per, _ = precision_recall_fscore_support(tel, tep, average=None, labels=[0, 1])
p_w, r_w, f1_w, _ = precision_recall_fscore_support(tel, tep, average="weighted")
cm = confusion_matrix(tel, tep)
roc = roc_auc_score(tel, teprobs) if len(np.unique(tel)) > 1 else 0

print(f"\n{'='*60}\nBanglaBERT Enhanced Baseline — TEST RESULTS")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (weighted): {p_w:.4f}, Real={p_per[0]:.4f}, Fake={p_per[1]:.4f}")
print(f"Recall (weighted):    {r_w:.4f}, Real={r_per[0]:.4f}, Fake={r_per[1]:.4f}")
print(f"F1 (weighted):        {f1_w:.4f}, Real={f1_per[0]:.4f}, Fake={f1_per[1]:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(f"Confusion Matrix: {cm.tolist()}")
print(f"{'='*60}")

res = {
    "experiment": "BanglaBERT Enhanced Baseline v3 (Ours)",
    "accuracy": acc, "accuracy_pct": acc * 100,
    "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
    "precision_real": p_per[0], "precision_fake": p_per[1],
    "recall_real": r_per[0], "recall_fake": r_per[1],
    "f1_real": f1_per[0], "f1_fake": f1_per[1],
    "roc_auc": roc,
    "confusion_matrix": cm.tolist(),
    "best_val_acc": best_val,
    "epochs_actual": ep + 1,
}
with open(os.path.join(SAVE_DIR, "results_exp1_enhanced.json"), "w") as f:
    json.dump(res, f, indent=2)
print("Saved results_exp1_enhanced.json")
