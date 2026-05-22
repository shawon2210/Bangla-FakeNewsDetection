"""
Experiment 1: BanglaBERT Text-Only Baseline (FIXED)
- Full dataset (no .head() truncation)
- 5 epochs with early stopping
- Per-class precision/recall/f1 output
"""
import os, sys, time, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm

MODEL_NAME = "csebuetnlp/banglabert"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
PATIENCE = 2
DEVICE = torch.device("cpu")
BASE_DIR = "/mnt/d/all files/ThesisP/BanglaFakeNewsProject/BanglaFakeNewsProject"

print("=" * 60)
print("EXPERIMENT 1: BanglaBERT Text-Only Baseline (FULL DATASET)")
print("=" * 60)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data.iloc[i]
        text = str(r.get("headline","")) + " " + str(r.get("description",""))
        enc = self.tok(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(int(r["label"]), dtype=torch.long)}

class BanglaBERTClassifier(nn.Module):
    def __init__(self, model_name, freeze_layers=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters(): p.requires_grad = False
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        return self.fc(self.drop(x))

# FULL dataset - no truncation
train_df = pd.read_csv(os.path.join(BASE_DIR,"Train.csv"), low_memory=False)
val_df = pd.read_csv(os.path.join(BASE_DIR,"Validation.csv"), low_memory=False)
test_df = pd.read_csv(os.path.join(BASE_DIR,"Test.csv"), low_memory=False)
print(f"Data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dl = DataLoader(TextDataset(train_df, tok, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(TextDataset(val_df, tok, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl = DataLoader(TextDataset(test_df, tok, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = BanglaBERTClassifier(MODEL_NAME).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,} total, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

crit = nn.CrossEntropyLoss()
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1*len(train_dl)*EPOCHS), num_training_steps=len(train_dl)*EPOCHS)

best_val, best_state, patience_ctr = 0, None, 0
start = time.time()
for ep in range(EPOCHS):
    model.train(); tl, tp, tlb = 0, [], []
    for b in tqdm(train_dl, desc=f"E{ep+1}/{EPOCHS}"):
        opt.zero_grad()
        logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        loss = crit(logits, b["label"].to(DEVICE))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tl += loss.item(); tp.extend(torch.argmax(logits,1).cpu().numpy()); tlb.extend(b["label"].numpy())
    ta = accuracy_score(tlb, tp)
    model.eval(); vl, vp, vlb = 0, [], []
    with torch.no_grad():
        for b in val_dl:
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            vl += crit(logits, b["label"].to(DEVICE)).item()
            vp.extend(torch.argmax(logits,1).cpu().numpy()); vlb.extend(b["label"].numpy())
    va = accuracy_score(vlb, vp); _,_,vf1,_ = precision_recall_fscore_support(vlb, vp, average="weighted")
    print(f"  E{ep+1}: TrainAcc={ta:.4f} ValAcc={va:.4f} ValF1={vf1:.4f}")
    if va > best_val:
        best_val = va; best_state = {k:v.clone() for k,v in model.state_dict().items()}; patience_ctr = 0
        print(f"  -> New best model saved (ValAcc={va:.4f})")
    else:
        patience_ctr += 1
        print(f"  -> No improvement ({patience_ctr}/{PATIENCE})")
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {ep+1}")
            break

print(f"Done in {(time.time()-start)/60:.1f}min. BestVal={best_val:.4f}")

model.load_state_dict(best_state); model.eval(); tep, tel = [], []
with torch.no_grad():
    for b in test_dl:
        logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        tep.extend(torch.argmax(logits,1).cpu().numpy()); tel.extend(b["label"].numpy())

acc = accuracy_score(tel, tep)
p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(tel, tep, average=None, labels=[0, 1])
p_w, r_w, f1_w, _ = precision_recall_fscore_support(tel, tep, average="weighted")
cm = confusion_matrix(tel, tep)

print(f"\n{'='*60}\nBanglaBERT Baseline — TEST RESULTS")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Per-class Precision: Real={p_per_class[0]:.4f}, Fake={p_per_class[1]:.4f}")
print(f"Per-class Recall:    Real={r_per_class[0]:.4f}, Fake={r_per_class[1]:.4f}")
print(f"Per-class F1:        Real={f1_per_class[0]:.4f}, Fake={f1_per_class[1]:.4f}")
print(f"Weighted Precision:  {p_w:.4f}")
print(f"Weighted Recall:     {r_w:.4f}")
print(f"Weighted F1:         {f1_w:.4f}")
print(f"Confusion Matrix:    {cm.tolist()}")
print(f"{'='*60}")

res = {
    "experiment": "BanglaBERT Text-Only Baseline",
    "accuracy": acc,
    "accuracy_pct": acc * 100,
    "precision_weighted": p_w,
    "recall_weighted": r_w,
    "f1_weighted": f1_w,
    "precision_per_class": p_per_class.tolist(),
    "recall_per_class": r_per_class.tolist(),
    "f1_per_class": f1_per_class.tolist(),
    "precision_real": p_per_class[0],
    "precision_fake": p_per_class[1],
    "recall_real": r_per_class[0],
    "recall_fake": r_per_class[1],
    "f1_real": f1_per_class[0],
    "f1_fake": f1_per_class[1],
    "confusion_matrix": cm.tolist(),
    "best_val_acc": best_val,
    "epochs": EPOCHS,
    "epochs_actual": ep + 1
}
with open(os.path.join(BASE_DIR,"results_banglabert_baseline.json"),"w") as f: json.dump(res,f,indent=2)
print("Saved results_banglabert_baseline.json")
