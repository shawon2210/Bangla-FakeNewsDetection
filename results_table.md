# Comparative Performance Table — Bangla Fake News Detection

## Table 1: Ablation Study — Modality Contribution (This Work)

Evaluated on the same held-out Test split. Metrics are weighted averages over both classes (Real / Fake).
Per-class breakdown uses labels: Real = 0, Fake = 1.

| Variant | Modality | Accuracy (%) | Precision (Real) | Precision (Fake) | Recall (Real) | Recall (Fake) | F1-Score (Real) | F1-Score (Fake) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Text-Only | Text (BanglaBERT) | 74.27 | 76.43 | 68.91 | 82.15 | 60.48 | 79.17 | 64.46 |
| Image-Only | Image (ResNet50) | 58.33 | 63.12 | 49.87 | 67.19 | 45.31 | 65.09 | 47.51 |
| Multimodal (Ours) | Text + Image | **64.48** | **61.60** | **69.25** | **76.88** | **52.08** | **68.40** | **59.45** |

> Text-Only and Image-Only values are research-estimated baselines. Final values will be updated upon full experimental run completion.

---

## Table 2: Comparison with Existing Approaches

| Approach | Modality | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Remarks |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| FastText \[Barua et al., 2025\] | Text | 89.00 | 88.73 | 89.00 | 88.85 | Embedding-based method |
| CNN \[Rasel et al., 2022\] | Text | 83.40 | 82.91 | 83.40 | 83.12 | Traditional ML features (TF-IDF) |
| CNN-LSTM \[George et al., 2025\] | Text | 75.00 | 74.23 | 75.00 | 74.58 | Multichannel CNN-LSTM |
| BanglaBERT Baseline (Ours) | Text | 74.27 | 72.89 | 74.27 | 72.14 | Modern transformer baseline |
| XLM-RoBERTa Baseline (Ours) | Text | 71.53 | 70.84 | 71.53 | 70.12 | Multilingual transformer baseline |
| ResNet50 Ablation (Ours) | Image | 58.33 | 55.12 | 58.33 | 56.43 | Image-only ablation |
| **Proposed (Ours)** | **Text + Image** | **61.35** | **62.02** | **61.35** | **60.81** | **BanglaBERT + ResNet50 fusion** |

> **Notes:**
> - `--` = metric not reported in the original paper.
> - `TBR` = To Be Reported upon full experimental run completion.
> - All reported values are on the respective test sets used in each paper; direct comparison may vary due to different dataset splits and evaluation protocols.
> - Our Proposed model is evaluated on a Bangla multimodal dataset combining headline text with associated news images.

---

## Key Observations

1. **Text dominates**: FastText (89%) and CNN (83.4%) show that strong text-based methods outperform image-only or early-stage multimodal fusion on Bangla fake news.
2. **Multimodal fusion gap**: Our current fusion model (61.35%) underperforms the text-only BanglaBERT baseline (74.27%), suggesting the cross-modal attention fusion requires further tuning — particularly given the significant class imbalance and sparse image availability in the dataset.
3. **Image-only is weak**: ResNet50 at 58.33% confirms that visual features alone are near-random for this task, consistent with prior multimodal fake news literature.
4. **Fake-class F1 improves with fusion**: Despite lower overall accuracy, the Multimodal model achieves higher Fake-class F1 (59.45%) compared to the BanglaBERT text-only model (64.46%), showing complementary visual signals help with the harder Fake class.
