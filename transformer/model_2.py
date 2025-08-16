"""
CNN + Transformer Encoder-Decoder for InkML→LaTeX
"""
from __future__ import annotations
import math
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, List
from matplotlib import pyplot as plt
import os 
# from .config import Config
import json
from torchvision import transforms, models 

import torch.optim as optim
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
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
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
        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Change first conv layer to accept 1 channel (grayscale)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the fully connected layer and avgpool
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # output: [B, 512, H', W']
        # Optionally, add a 1x1 conv to change channel dim if needed
        self.out_proj = nn.Conv2d(512, out_dim, kernel_size=1) if out_dim != 512 else nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.out_proj(x)
        return x  # [B, out_dim, H', W']

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(max_seq_len, d_model))  # [max_seq_len, d_model]
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, memory, tgt, tgt_mask=None):
        # memory: [S, B, D], tgt: [T, B]
        tgt_emb = self.embedding(tgt) + self.pos[:tgt.size(0)].unsqueeze(1)
        out = self.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc(out)

class CNNTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model=d_model)
    def forward(self, images, tgt, tgt_mask=None):
        memory = self.encoder(images)  # [B, d_model, H', W']
        B, C, H, W = memory.shape
        memory = memory.flatten(2).permute(2, 0, 1)  # [S, B, d_model]
        return self.decoder(memory, tgt, tgt_mask)
    
def collate_fn(batch):
    imgs, token_ids = zip(*batch)
    imgs = torch.stack(imgs, 0)
    # Pad token sequences
    lengths = [len(t) for t in token_ids]
    max_len = max(lengths)
    pad_id = batch[0][1].new_full((1,), fill_value=0)[0].item()  # or use tokenizer.pad_id
    padded = torch.full((len(token_ids), max_len), tokenizer.pad_id, dtype=torch.long)
    for i, t in enumerate(token_ids):
        padded[i, :len(t)] = t
    return imgs, padded

# Build vocab from all labels, change this to the crohme2019 labels. 

def train(model, train_loader, val_loader, tokenizer, num_epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        for imgs, token_ids in train_loader:
            imgs = imgs.to(device)
            token_ids = token_ids.to(device)
            tgt_in = torch.full((token_ids.size(0), 1), tokenizer.bos_id, dtype=torch.long, device=device)
            tgt_in = torch.cat([tgt_in, token_ids[:, :-1]], dim=1)
            tgt_in = tgt_in.transpose(0, 1)
            tgt_out = token_ids.transpose(0, 1)
            optimizer.zero_grad()
            logits = model(imgs, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Train accuracy
            preds = logits.argmax(-1)
            mask = (tgt_out != tokenizer.pad_id)
            correct = (preds == tgt_out) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_tokens if total_tokens > 0 else 0
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        # Validation
        if 'val_loader' in locals():
            val_loss, val_acc = evaluate(model, val_loader, tokenizer, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} Acc: {train_acc:.4f}")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses: plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    if val_accuracies: plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return model 

def evaluate(model, loader, tokenizer, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for imgs, token_ids in loader:
            imgs = imgs.to(device)
            token_ids = token_ids.to(device)
            tgt_in = torch.full((token_ids.size(0), 1), tokenizer.bos_id, dtype=torch.long, device=device)
            tgt_in = torch.cat([tgt_in, token_ids[:, :-1]], dim=1)
            tgt_in = tgt_in.transpose(0, 1)
            tgt_out = token_ids.transpose(0, 1)
            logits = model(imgs, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            preds = logits.argmax(-1)
            mask = (tgt_out != tokenizer.pad_id)
            correct = (preds == tgt_out) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy

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
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
    ])

    train_dataset = MathExprDataset(
        img_dir='transformer/data/train/pngs',
        inkml_dir='transformer/data/crohme2019/train',
        tokenizer=tokenizer,
        transform=transform
    )
    val_dataset = MathExprDataset(
        img_dir='transformer/data/valid/pngs',
        inkml_dir='transformer/data/crohme2019/valid',
        tokenizer=tokenizer,
        transform=transform
    )
    test_dataset = MathExprDataset(
        img_dir='transformer/data/test/pngs',
        inkml_dir='transformer/data/crohme2019/test',
        tokenizer=tokenizer,
        transform=transform
    )


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = CNNTransformer(len(tokenizer), d_model=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = train(model, train_loader, val_loader, tokenizer, num_epochs=10, lr=1e-4)