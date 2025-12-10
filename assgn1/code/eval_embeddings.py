# eval_embeddings.py
import argparse
import numpy as np
import os
import torch
from datasets import make_eval_loader
from models_loader import get_edgeface_model
from utils import extract_embeddings, cosine_sim, compute_eer
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--edgeface_dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_name', default='edgeface_s_gamma_05')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # load model
    model = get_edgeface_model(args.edgeface_dir, args.model_name, pretrained=False, device=args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.to(args.device)

    loader = make_eval_loader(args.data, batch_size=32)
    embeddings, labels, paths = extract_embeddings(model, loader, device=args.device)
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)
    # compute all pair similarities
    sims = cosine_sim(embeddings, embeddings)  # NxN
    N = sims.shape[0]

    out_scores = []
    # create score.txt as required : for each pair (i<j) record dissimilarity and label
    score_path = os.path.join(args.output_dir, 'score.txt')
    with open(score_path, 'w') as f:
        for i in range(N):
            for j in range(i+1, N):
                sim = sims[i,j]
                dissim = 1.0 - sim
                label = 1 if labels[i]==labels[j] else 0
                f.write(f"{paths[i]} {paths[j]} {label} {dissim:.6f}\n")
                out_scores.append((sim, label))
    print("Saved score.txt with pairwise dissimilarities.")

    # compute EER and DET components
    scores = np.array([s for s,l in out_scores])
    labs = np.array([l for s,l in out_scores])
    eer, thr, fpr, tpr, thresholds = compute_eer(scores, labs)
    print(f"EER: {eer*100:.4f}% threshold:{thr:.6f}")

    # compute TMR at FMR = 1e-1, 1e-2, 1e-4
    # convert similarity scores to thresholds
    fpr_vals = [1e-1, 1e-2, 1e-4]
    tmr_results = {}
    # get roc curvemodel_name
    fpr_all, tpr_all, thr_all = roc_curve(labs, scores)
    for fmr in fpr_vals:
        # find threshold where fpr <= fmr (closest)
        idx = np.where(fpr_all <= fmr)[0]
        if len(idx) == 0:
            tmr = 0.0
        else:
            tmr = tpr_all[idx[-1]]
        tmr_results[fmr] = tmr

    # Save eval summary
    with open(os.path.join(args.output_dir, 'eval.txt'), 'w') as f:
        f.write(f"EER: {eer}\n")
        f.write(f"EER_percent: {eer*100}\n")
        f.write(f"EER_threshold: {thr}\n")
        for fmr, tmr in tmr_results.items():
            f.write(f"TMR_at_FMR_{fmr}: {tmr}\n")

    # Plot DET/ROC
    plt.figure()
    plt.plot(fpr_all, 1 - tpr_all)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('DET-style curve (log-log)')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'det_curve.png'))
    plt.close()

    print("Evaluation artifacts saved in", args.output_dir)

if __name__ == '__main__':
    main()