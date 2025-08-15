"""
CNN + Transformer Encoder-Decoder for InkML→LaTeX
Standalone file: ResNet-18 backbone → Transformer Encoder (over spatial tokens)
→ Transformer Decoder (token generator). Includes a tiny test harness.
"""
from __future__ import annotations
import math
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, List
import os 
# from .config import Config
import json
from torchvision import transforms

import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

import os
import xml.etree.ElementTree as ET
import re

def tokenize_latex(s):
    # Remove $ if present
    s = s.replace('$', '')
    # Regex: LaTeX commands, braces, symbols, single digits, letters
    pattern = r'(\\[a-zA-Z]+|[{}^_=+\-*/\[\](),]|[0-9]|[a-zA-Z])'
    tokens = re.findall(pattern, s)
    return tokens

def extract_latex_from_inkml(inkml_path):
    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        for ann in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
            if ann.attrib.get('type') == 'truth':
                return ann.text
    except ET.ParseError as e:
        print(f"Parse error in {inkml_path}: {e}")
    except Exception as e:
        print(f"Error in {inkml_path}: {e}")
    return None

def build_vocab(inkml_dir):
    vocab = set()
    for root, _, files in os.walk(inkml_dir):
        for fname in files:
            if fname.endswith('.inkml'):
                path = os.path.join(root, fname)
                latex = extract_latex_from_inkml(path)
                if latex:
                    tokens = tokenize_latex(latex)
                    vocab.update(tokens)
    # Add special tokens
    special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = special + sorted(vocab)
    return vocab

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

def parse_crohme_txt(txt_path):
    data = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            # Example line: "8 Right + NoRel 8 ...\tcrohme2019/test/UN19_1041_em_590.inkml"
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            expr, inkml_path = parts
            img_name = os.path.splitext(os.path.basename(inkml_path))[0] + '.png'
            data[img_name] = expr
    return data

def _center(box):
    y1,x1,y2,x2 = box
    return ( (y1+y2)/2.0, (x1+x2)/2.0 ), (y2-y1, x2-x1)

""" def reconstruct(preds_for_image, cfg: Config = Config()):
    # Very simple greedy reconstruction: LEFT->RIGHT with super/sub attachment
    # preds_for_image: list of dicts with keys "bbox", "label"
    # returns LaTeX string
    # Sort by x then y
    items = sorted(preds_for_image, key=lambda o: (o["bbox"][1], o["bbox"][0]))
    out = ""
    used = set()
    for i, obj in enumerate(items):
        if i in used: 
            continue
        box = obj["bbox"]; label = obj["label"]
        (cy, cx), (h, w) = _center(box)
        base = label
        # search for super/sub among following few tokens
        sup, sub = None, None
        for j in range(i+1, min(i+6, len(items))):
            if j in used: 
                continue
            b2 = items[j]["bbox"]; (cy2, cx2), (h2, w2) = _center(b2)
            dx = cx2 - cx; dy = cy - cy2
            if dx < -cfg.tau_x_right * w: 
                continue
            if dy > cfg.tau_sup_y * h and abs(cx2 - cx) < 1.2*w:
                sup = items[j]["label"]; used.add(j)
            if -dy > cfg.tau_sub_y * h and abs(cx2 - cx) < 1.2*w:
                sub = items[j]["label"]; used.add(j)
        if sup and sub:
            out += f"{base}^{{{sup}}}_{{{sub}}} "
        elif sup:
            out += f"{base}^{{{sup}}} "
        elif sub:
            out += f"{base}_{{{sub}}} "
        else:
            out += f"{base} "
    return out.strip() """

class MathExprDataset(Dataset):
    def __init__(self, img_dir, inkml_dir, tokenizer, transform=None):
        import os
        self.img_dir = img_dir
        self.inkml_dir = inkml_dir
        self.transform = transform
        self.tokenizer = tokenizer
        # List all PNG images
        self.img_names = [] 
        for dir in img_dir:
            self.img_dir_name = [f for f in os.listdir(dir) if f.endswith('.png')]
            self.img_names.extend(self.img_dir_name)
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        import os
        from latex_extraction import extract_latex_from_inkml
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        # Find corresponding .inkml file
        base = os.path.splitext(img_name)[0]
        inkml_path = os.path.join(self.inkml_dir, base + '.inkml')
        label = extract_latex_from_inkml(inkml_path)
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

# Build vocab from all labels, change this to the crohme2019 labels. 

if __name__ == "__main__":
    inkml_dir = "transformer/data/crohme2019"  # adjust as needed
    vocab = build_vocab(inkml_dir)
    # Save to file
    with open("transformer/data/vocab.txt", "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    print(f"Vocab size: {len(vocab)}")


    tokenizer = LatexTokenizer(vocab)



    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # or whatever your model expects
        transforms.ToTensor(),
    ])

    dataset = MathExprDataset(
        img_dir=['transformer/data/test/pngs', 'transformer/data/valid/pngs', 'transformer/data/train/pngs'],
        inkml_dir='transformer/data/inkml',
        tokenizer=tokenizer,
        transform=transform
    )


    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=None)