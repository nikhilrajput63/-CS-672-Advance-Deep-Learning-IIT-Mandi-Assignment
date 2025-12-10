# Report — Recognition of Forehead Creases Region using EdgeFace

## 1. Setup
- Repository: otroshi/edgeface (cloned into `edgeface/`)
- Dataset: `forehead-v1-labeled/`
- Models trained:
  - Model A: EdgeFace finetuned with Triplet Loss
  - Model B: EdgeFace finetuned with ArcFace Loss
  - Model C: Joint Triplet + ArcFace

## 2. Training configuration
- Model variant: edgeface_s_gamma_05
- Image size: 112x112
- Optimizer: Adam lr=1e-4
- Batch size: 32
- Epochs: 30 (example)
- Layers finetuned: [document here which layers you unfreezed]

## 3. Results
- Attach plots:
  - `results_forehead/triplet/det_curve.png`
  - `results_forehead/triplet/plots/tsne.png`
  - `results_forehead/arcface/...` etc.
- EERs:
  - Triplet: X%
  - ArcFace: Y%
  - Joint: Z%

## 4. Observations
- Training stability:
  - ...
- Convergence speed:
  - ...
- Embedding separability (t-SNE/PCA):
  - ...
- Strengths and weaknesses:
  - Triplet Loss: ...
  - ArcFace Loss: ...

## 5. Conclusion
Write final conclusion here.

## 6. How to reproduce
Commands to run (copy from the main instructions).