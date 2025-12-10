# utils.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve

def save_checkpoint(state, is_best, outdir, filename='checkpoint_latest.pt'):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(outdir, 'checkpoint_best.pt'))

def extract_embeddings(model, loader, device='cpu'):
    model.eval()
    embeddings = []
    labels = []
    paths = []
    with torch.no_grad():
        for batch in loader:
            imgs, batch_labels, batch_paths = batch
            imgs = imgs.to(device)
            embs = model(imgs)
            if isinstance(embs, tuple):  # if model returns (out, something)
                embs = embs[0]
            embs = embs.detach().cpu().numpy()
            embeddings.append(embs)
            labels.extend(batch_labels.numpy().tolist())
            paths.extend(batch_paths)
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels, paths

def cosine_sim(a, b):
    # a: (N,d), b: (M,d) -> (N,M)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def compute_eer(scores, labels):
    # scores: similarity (higher means more similar)
    # labels: 1 for genuine, 0 for impostor
    # compute fpr, tpr
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # EER is where fpr ~= fnr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return eer, thr, fpr, tpr, thresholds