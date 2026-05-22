"""
Experiment 4: BanglaBERT Text-Only Ablation (within multimodal framework)
Uses OptimizedMultimodalModel with ZERO images to isolate text contribution.
"""
import os, sys, time, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/mnt/d/all files/ThesisP/BanglaFakeNewsProject/BanglaFakeNewsProject")
from model_defs import OptimizedMultimodalModel

MODEL_NAME = "csebuetnlp/banglabert"
MAX_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
TEXT_LR = 1e-5
OTHER_LR = 1e-4
DEVICE = torch.device("cpu")
BASE_DIR = "/mnt/d/all files/ThesisP/BanglaFakeNewsProject/BanglaFakeNewsProject"

print("=" * 60)
print("EXPERIMENT 4: BanglaBERT Text-Only Ablation (Multimodal Framework)")
print("=" * 60)

class TextOnlyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df.reset_index(drop=True); self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data.iloc[i]
        text = str(r.get("headline","")) + " " + str(r.get("description",""))
        enc = self.tok(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "image": torch.zeros(3, 224, 224), "label": torch.tensor(int(r["label"]), dtype=torch.long)}

train_df = pd.read_csv(os.path.join(BASE_DIR,"Train.csv"), low_memory=False)
val_df = pd.read_csv(os.path.join(BASE_DIR,"Validation.csv"), low_memory=False)
test_df = pd.read_csv(os.path.join(BASE_DIR,"Test.csv"), low_memory=False)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dl = DataLoader(TextOnlyDataset(train_df, tok, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(TextOnlyDataset(val_df, tok, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl = DataLoader(TextOnlyDataset(test_df, tok, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = OptimizedMultimodalModel(text_model_name=MODEL_NAME, num_classes=2).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,} total, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

tp, op = [], []
for n, p in model.named_parameters():
    if 'text_encoder' in n: tp.append(p)
    else: op.append(p)

crit = nn.CrossEntropyLoss()
opt = optim.AdamW([
    {'params': filter(lambda p: p.requires_grad, tp), 'lr': TEXT_LR},
    {'params': filter(lambda p: p.requires_grad, op), 'lr': OTHER_LR},
], weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1*len(train_dl)*EPOCHS), num_training_steps=len(train_dl)*EPOCHS)

best_val, best_state = 0, None
start = time.time()
for ep in range(EPOCHS):
    model.train(); tl, tlp, tlb = 0, [], []
    for b in tqdm(train_dl, desc=f"E{ep+1}/{EPOCHS}"):
        opt.zero_grad()
        logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["image"].to(DEVICE))
        loss = crit(logits, b["label"].to(DEVICE))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tl += loss.item(); tlp.extend(torch.argmax(logits,1).cpu().numpy()); tlb.extend(b["label"].numpy())
    ta = accuracy_score(tlb, tlp)
    model.eval(); vl, vp, vlb = 0, [], []
    with torch.no_grad():
        for b in val_dl:
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["image"].to(DEVICE))
            vl += crit(logits, b["label"].to(DEVICE)).item()
            vp.extend(torch.argmax(logits,1).cpu().numpy()); vlb.extend(b["label"].numpy())
    va = accuracy_score(vlb, vp); _,_,vf1,_ = precision_recall_fscore_support(vlb, vp, average="weighted")
    print(f"  E{ep+1}: TrainAcc={ta:.4f} ValAcc={va:.4f} ValF1={vf1:.4f}")
    if va > best_val: best_val = va; best_state = {k:v.clone() for k,v in model.state_dict().items()}

print(f"Done in {(time.time()-start)/60:.1f}min. BestVal={best_val:.4f}")

model.load_state_dict(best_state); model.eval(); tep, tel = [], []
with torch.no_grad():
    for b in test_dl:
        logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["image"].to(DEVICE))
        tep.extend(torch.argmax(logits,1).cpu().numpy()); tel.extend(b["label"].numpy())

acc = accuracy_score(tel, tep); p,r,f1,_ = precision_recall_fscore_support(tel, tep, average="weighted")
cm = confusion_matrix(tel, tep)
print(f"\n{'='*60}\nBanglaBERT Ablation (Multimodal Framework) — TEST\nAccuracy: {acc:.4f} ({acc*100:.2f}%)\nPrecision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\nCM: {cm.tolist()}\n{'='*60}")

res = {"experiment":"BanglaBERT Text-Only Ablation (Multimodal Framework)","accuracy":acc,"precision":p,"recall":r,"f1":f1,"confusion_matrix":cm.tolist(),"best_val_acc":best_val,"epochs":EPOCHS}
with open(os.path.join(BASE_DIR,"results_banglabert_ablation.json"),"w") as f: json.dump(res,f,indent=2)
print("Saved results_banglabert_ablation.json")
