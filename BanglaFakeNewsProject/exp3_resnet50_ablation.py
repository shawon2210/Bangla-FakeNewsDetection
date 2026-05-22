"""
Experiment 3: ResNet50 Image-Only Ablation (FIXED)
- Full dataset (no .head() truncation)
- 5 epochs with early stopping
- Per-class precision/recall/f1 output
"""
import os, sys, time, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
PATIENCE = 2
DEVICE = torch.device("cpu")
BASE_DIR = r"d:\all files\ThesisP\BanglaFakeNewsProject\BanglaFakeNewsProject"
IMG_DIR = os.path.join(BASE_DIR, "images")
IMG_SIZE = 224

print("=" * 60)
print("EXPERIMENT 3: ResNet50 Image-Only Ablation (FULL DATASET)")
print("=" * 60)

img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def find_img(image_id):
    for ext in ['.png','.jpg','.jpeg','.webp','.gif','.jfif']:
        p = os.path.join(IMG_DIR, str(image_id)+ext)
        if os.path.exists(p): return p
    return None

class ImgDataset(Dataset):
    def __init__(self, df, transform):
        self.data = df.reset_index(drop=True); self.tf = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data.iloc[i]; label = int(r["label"])
        p = find_img(r["image_id"])
        if p:
            try: img = self.tf(Image.open(p).convert("RGB"))
            except: img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        else: img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}

class ResNet50Classifier(nn.Module):
    def __init__(self, freeze_layers=4):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        ch = list(bb.children())
        for i, c in enumerate(ch):
            if i < freeze_layers:
                for p in c.parameters(): p.requires_grad = False
        self.backbone = nn.Sequential(*ch[:-1])
        self.fc = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,2))
    def forward(self, image):
        return self.fc(self.backbone(image).view(image.size(0),-1))

# FULL dataset - no truncation
train_df = pd.read_csv(os.path.join(BASE_DIR,"Train.csv"), low_memory=False)
val_df = pd.read_csv(os.path.join(BASE_DIR,"Validation.csv"), low_memory=False)
test_df = pd.read_csv(os.path.join(BASE_DIR,"Test.csv"), low_memory=False)
print(f"Data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

train_dl = DataLoader(ImgDataset(train_df, img_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(ImgDataset(val_df, img_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl = DataLoader(ImgDataset(test_df, img_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = ResNet50Classifier().to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,} total, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

crit = nn.CrossEntropyLoss()
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)

best_val, best_state, patience_ctr = 0, None, 0
start = time.time()
for ep in range(EPOCHS):
    model.train(); tl, tp, tlb = 0, [], []
    for b in tqdm(train_dl, desc=f"E{ep+1}/{EPOCHS}"):
        opt.zero_grad()
        logits = model(b["image"].to(DEVICE))
        loss = crit(logits, b["label"].to(DEVICE))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item(); tp.extend(torch.argmax(logits,1).cpu().numpy()); tlb.extend(b["label"].numpy())
    ta = accuracy_score(tlb, tp)
    model.eval(); vl, vp, vlb = 0, [], []
    with torch.no_grad():
        for b in val_dl:
            logits = model(b["image"].to(DEVICE))
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
        logits = model(b["image"].to(DEVICE))
        tep.extend(torch.argmax(logits,1).cpu().numpy()); tel.extend(b["label"].numpy())

acc = accuracy_score(tel, tep)
p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(tel, tep, average=None, labels=[0, 1])
p_w, r_w, f1_w, _ = precision_recall_fscore_support(tel, tep, average="weighted")
cm = confusion_matrix(tel, tep)

print(f"\n{'='*60}\nResNet50 Image-Only Ablation — TEST RESULTS")
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
    "experiment": "ResNet50 Image-Only Ablation",
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
with open(os.path.join(BASE_DIR,"results_resnet50_ablation.json"),"w") as f: json.dump(res,f,indent=2)
print("Saved results_resnet50_ablation.json")
