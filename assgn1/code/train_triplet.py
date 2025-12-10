# train_triplet.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import make_triplet_loader, make_eval_loader
from models_loader import get_edgeface_model
from utils import save_checkpoint
import os
from tqdm import tqdm

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

    # Decide which layers to freeze / unfreeze:
    # default: freeze backbone except final embedding FC (if present). We'll unfreeze last stage.
    for name, param in model.named_parameters():
        param.requires_grad = False
    # Unfreeze all for simplicity (small dataset) -> comment this line to freeze backbone
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_loader = make_triplet_loader(args.data, batch_size=args.batch_size)
    best_loss = 1e9

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        it = 0
        for anchor, positive, negative, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            it += 1

        avg_loss = running / max(1, it)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        # save checkpoint (simple)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, is_best, args.output_dir)

    print("Training finished. Best loss:", best_loss)

if __name__ == '__main__':
    train()