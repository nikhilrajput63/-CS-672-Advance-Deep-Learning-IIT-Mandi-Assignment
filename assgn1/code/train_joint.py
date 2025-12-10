# train_joint.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import make_triplet_loader, make_arcface_loader
from models_loader import get_edgeface_model
from arcface_loss import ArcFaceMargin
from train_arcface import ArcFaceHead
from utils import save_checkpoint
from tqdm import tqdm
import os

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
    p.add_argument('--alpha', type=float, default=1.0, help='weight for triplet loss')
    p.add_argument('--beta', type=float, default=1.0, help='weight for arcface loss')
    return p.parse_args()

def train():
    args = parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    model = get_edgeface_model(args.edgeface_dir, model_name=args.model_name, pretrained=True, device=device)
    for param in model.parameters():
        param.requires_grad = True

    # loaders
    triplet_loader = make_triplet_loader(args.data, batch_size=args.batch_size)
    arc_loader = make_arcface_loader(args.data, batch_size=args.batch_size)

    # get emb dim
    model.eval()
    with torch.no_grad():
        emb = model(torch.randn(1,3,112,112).to(device))
    emb_dim = emb.shape[1]
    num_classes = len(arc_loader.dataset.classes)
    head = ArcFaceHead(emb_dim, num_classes).to(device)
    arc_margin = ArcFaceMargin(s=30.0, m=0.5)
    triplet_criterion = nn.TripletMarginLoss(margin=0.2)
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)

    # We'll iterate over the smaller of the two loaders per epoch
    for epoch in range(args.epochs):
        model.train(); head.train()
        running_loss = 0.0; it = 0
        t_iter = iter(triplet_loader)
        a_iter = iter(arc_loader)
        steps = min(len(triplet_loader), len(arc_loader))
        for _ in tqdm(range(steps), desc=f"Epoch {epoch+1}/{args.epochs}"):
            try:
                anchor, positive, negative, _ = next(t_iter)
            except StopIteration:
                t_iter = iter(triplet_loader); anchor, positive, negative, _ = next(t_iter)
            try:
                imgs, labels, _ = next(a_iter)
            except StopIteration:
                a_iter = iter(arc_loader); imgs, labels, _ = next(a_iter)

            anchor = anchor.to(device); positive = positive.to(device); negative = negative.to(device)
            imgs = imgs.to(device); labels = labels.to(device)

            emb_a = model(anchor); emb_p = model(positive); emb_n = model(negative)
            triplet_loss = triplet_criterion(emb_a, emb_p, emb_n)

            feats = model(imgs)
            logits = head(feats)
            logits_margin = arc_margin.forward(logits, labels)
            arc_loss = ce(logits_margin, labels)

            loss = args.alpha * triplet_loss + args.beta * arc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item(); it += 1

        avg = running_loss / max(1, it)
        print(f"Epoch {epoch+1} avg joint loss: {avg:.6f}")
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'head': head.state_dict(), 'optimizer': optimizer.state_dict()}, False, args.output_dir)

    print("Joint training finished.")

if __name__ == '__main__':
    train()