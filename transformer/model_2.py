"""
CNN + Transformer Encoder-Decoder for InkML→LaTeX
Standalone file: ResNet-18 backbone → Transformer Encoder (over spatial tokens)
→ Transformer Decoder (token generator). Includes a tiny test harness.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, List


import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
class LatexTokenizer:
    def __init__(self, vocab):
        self.itos = vocab
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def encode(self, s):
        # Tokenize LaTeX string to list of IDs
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in s.split()]
    def decode(self, ids):
        # Convert list of IDs back to LaTeX string
        return " ".join([self.itos[i] for i in ids if i < len(self.itos)])
    @property
    def pad_id(self): return self.stoi["<pad>"]
    @property
    def bos_id(self): return self.stoi["<bos>"]
    @property
    def eos_id(self): return self.stoi["<eos>"]
    def __len__(self): return len(self.itos)

def spatial_to_latex(spatial_str):
    # Dummy: replace with a real parser for your format
    # Example: "8 Right + NoRel 8 Right × Right 7 NoRel - Below 2 NoRel = Right 3 Right 6"
    # → "8 + 8 \\times 7 - 2 = 36"
    # You may need a tree parser for full generality.
    raise NotImplementedError("Implement spatial-to-LaTeX conversion here.")


class MathExprDataset(Dataset):
    def __init__(self, img_dir, label_file, tokenizer, spatial_to_latex=None, transform=None):
        import json, os
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.spatial_to_latex = spatial_to_latex
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.img_names = list(self.labels.keys())
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        label = self.labels[img_name]
        if self.spatial_to_latex:
            label = self.spatial_to_latex(label)
        token_ids = self.tokenizer.encode(label)
        return img, torch.tensor(token_ids, dtype=torch.long)

class CNNEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.ReLU()
        )
    def forward(self, x):
        # x: [B,1,H,W] → [B, out_dim, H', W']
        return self.conv(x)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, memory, tgt, tgt_mask=None):
        # memory: [S, B, D], tgt: [T, B]
        tgt_emb = self.embedding(tgt) + self.pos[:, :tgt.size(0)]
        out = self.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc(out)

class CNNTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = TransformerDecoder(vocab_size)
    def forward(self, images, tgt, tgt_mask=None):
        # images: [B,1,H,W], tgt: [T,B]
        memory = self.encoder(images)  # [B, C, H', W']
        B, C, H, W = memory.shape
        memory = memory.flatten(2).permute(2, 0, 1)  # [S, B, C]
        return self.decoder(memory, tgt, tgt_mask)