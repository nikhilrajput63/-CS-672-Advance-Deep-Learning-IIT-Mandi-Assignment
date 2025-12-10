# train_arcface.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import make_arcface_loader
from models_loader import get_edgeface_model
from utils import save_checkpoint
from arcface_loss import ArcFaceMargin
from tqdm import tqdm

class ArcFaceHead(nn.Module):
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Normalize weight and input, produce cos(theta)
        x_norm = nn.functional.normalize(x, p=2, dim=1)
        w_norm = nn.functional.normalize(self.weight, p=2, dim=1)
        return torch.matmul(x_norm, w_norm.t())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--edgeface_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_name', default='edgeface_s_gamma_05')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    return p.parse_args()

def train():
    args = parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    model = get_edgeface_model(args.edgeface_dir, model_name=args.model_name, pretrained=True, device=device)

    # Unfreeze as needed
    for param in model.parameters():
        param.requires_grad = True

    # determine embedding dim
    model.eval()
    dummy = torch.randn(1,3,112,112).to(device)
    with torch.no_grad():
        emb = model(dummy)
    emb_dim = emb.shape[1]

    # create loader to infer number of classes
    loader = make_arcface_loader(args.data, batch_size=8)
    num_classes = len(loader.dataset.classes)
    print("Detected number of classes:", num_classes)

    head = ArcFaceHead(emb_dim, num_classes).to(device)
    arc_margin = ArcFaceMargin(s=30.0, m=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        head.train()
        running = 0.0
        it = 0
        for imgs, labels, paths in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device); labels = labels.to(device)
            feats = model(imgs)
            logits = head(feats)  # cosine of angles
            logits_margin = arc_margin.forward(logits, labels)
            loss = criterion(logits_margin, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item(); it += 1
        avg_loss = running / max(1, it)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'head': head.state_dict(), 'optimizer': optimizer.state_dict()}, False, args.output_dir)

    print("ArcFace training finished.")

if __name__ == '__main__':
    train()