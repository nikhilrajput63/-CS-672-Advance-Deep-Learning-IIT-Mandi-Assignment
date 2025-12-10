# visualize_embeddings.py
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--perplexity', type=int, default=30)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    X = np.load(args.embeddings)
    y = np.load(args.labels)
    # PCA (fast)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    for lbl in np.unique(y):
        sel = y==lbl
        plt.scatter(Xp[sel,0], Xp[sel,1], s=20, label=str(lbl))
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small', ncol=1)
    plt.title('PCA of embeddings')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'pca.png'))
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init='pca', random_state=42)
    Xt = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    for lbl in np.unique(y):
        sel = y==lbl
        plt.scatter(Xt[sel,0], Xt[sel,1], s=20, label=str(lbl))
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small', ncol=1)
    plt.title('t-SNE of embeddings')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'tsne.png'))
    plt.close()
    print("Saved PCA and t-SNE plots to", args.outdir)

if __name__ == '__main__':
    main()