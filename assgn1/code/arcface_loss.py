# arcface_loss.py
import torch
import torch.nn.functional as F
import math

class ArcFaceMargin:
    def __init__(self, s=30.0, m=0.5):
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        # logits = normalized_features @ normalized_weights
        # labels: long tensor
        # We'll convert logits -> add angular margin
        # This function expects logits are cos(theta)
        cos = logits
        theta = torch.acos(torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7))
        cos_m = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)
        out = cos * (1 - one_hot) + cos_m * one_hot
        out = out * self.s
        return out
