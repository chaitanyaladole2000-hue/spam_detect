"""
Train SpamBiLSTM.
Usage:  python model/train.py
        python model/train.py --data spam.tsv --epochs 20
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse, json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from model import SpamBiLSTM
from preprocess import Vocabulary, SpamDataset

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def load_data(path):
    p = Path(path)
    sep = "\t" if p.suffix == ".tsv" else ","
    df  = pd.read_csv(path, sep=sep, on_bad_lines="skip")
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename any known column variants → label, text
    rename = {}
    for c in df.columns:
        if c in ("message","sms","body","v2","text_message"): rename[c] = "text"
        if c in ("v1","category","class","type","target"):    rename[c] = "label"
    if rename: df = df.rename(columns=rename)

    # Drop extra columns like length, punct — keep only label + text
    df = df[["label", "text"]].copy()
    df["label"] = df["label"].map(lambda x: 1 if str(x).lower().strip() == "spam" else 0)
    df = df.dropna()
    print(f"Loaded {len(df)} rows | spam={df['label'].sum()} | ham={(df['label']==0).sum()}")
    return df["text"].tolist(), df["label"].tolist()

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            p = model(X).argmax(1)
            preds.extend(p.cpu().tolist())
            labels.extend(y.cpu().tolist())
    return f1_score(labels, preds, average="binary"), preds, labels

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    texts, labels = load_data(args.data)
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    Xv,  Xt,  yv,  yt  = train_test_split(Xte,  yte,   test_size=0.5, random_state=42, stratify=yte)

    vocab = Vocabulary()
    vocab.build(Xtr)
    vocab.save(str(ARTIFACTS / "vocab.pkl"))
    print(f"Vocab size: {len(vocab)}")

    tr_dl = DataLoader(SpamDataset(Xtr, ytr, vocab, args.max_len), args.batch, shuffle=True)
    vl_dl = DataLoader(SpamDataset(Xv,  yv,  vocab, args.max_len), args.batch)
    te_dl = DataLoader(SpamDataset(Xt,  yt,  vocab, args.max_len), args.batch)

    model = SpamBiLSTM(len(vocab), args.embed_dim, args.hidden_dim, args.layers, args.dropout).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    spam_n = sum(ytr); ham_n = len(ytr) - spam_n
    crit  = nn.CrossEntropyLoss(weight=torch.tensor([1.0, ham_n/spam_n]).to(device))
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_f1, patience = 0.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0
        for X, y in tr_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()

        vf1, _, _ = evaluate(model, vl_dl, device)
        sched.step(loss_sum / len(tr_dl))
        print(f"Epoch {ep:02d}/{args.epochs}  loss={loss_sum/len(tr_dl):.4f}  val_f1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1; patience = 0
            torch.save({"model_state": model.state_dict(), "val_f1": vf1, "epoch": ep,
                        "args": vars(args)}, str(ARTIFACTS / "best_model.pt"))
            print(f"  ✓ Saved (val_f1={vf1:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping."); break

    # Final test
    ckpt = torch.load(str(ARTIFACTS / "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tf1, tp, tl = evaluate(model, te_dl, device)
    print("\n" + classification_report(tl, tp, target_names=["Ham","Spam"]))
    json.dump({"val_f1": best_f1, "test_f1": tf1}, open(str(ARTIFACTS/"results.json"),"w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="spam.tsv")
    ap.add_argument("--epochs",    type=int,   default=20)
    ap.add_argument("--batch",     type=int,   default=32)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int,   default=128)
    ap.add_argument("--hidden_dim",type=int,   default=256)
    ap.add_argument("--layers",    type=int,   default=2)
    ap.add_argument("--dropout",   type=float, default=0.4)
    ap.add_argument("--max_len",   type=int,   default=256)
    ap.add_argument("--patience",  type=int,   default=5)
    train(ap.parse_args())
