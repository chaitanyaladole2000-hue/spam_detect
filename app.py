"""
app.py — SpamShield: Full-Stack Spam Email Detection using BiLSTM
Pages: / (home)  /detect  /analytics  /about
Run:  python app.py
      python app.py --train        (force retrain)
      python app.py --data spam.tsv --epochs 20
"""
import pickle
import sys
import os
import re
import json
import pickle
import sqlite3
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from flask import Flask, request, jsonify, render_template, g


# ══════════════════════════════════════════════════════════════
#  1. PREPROCESSING
# ══════════════════════════════════════════════════════════════

def clean(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"[^\w\s<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return clean(text).split()


class Vocabulary:
    def __init__(self, max_size=30000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token2idx = {"<PAD>": 0, "<UNK>": 1}

    def build(self, texts):
        counter = Counter(t for text in texts for t in tokenize(text))
        for token, freq in counter.most_common(self.max_size - 2):
            if freq < self.min_freq:
                break
            self.token2idx[token] = len(self.token2idx)

    def encode(self, text, max_len=256):
        ids = [self.token2idx.get(t, 1) for t in tokenize(text)[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.token2idx)


class SpamDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.X = [vocab.encode(t, max_len) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.long))


# ══════════════════════════════════════════════════════════════
#  2. BILSTM MODEL
# ══════════════════════════════════════════════════════════════

class SpamBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        emb = self.drop(self.embedding(x))
        _, (h, _) = self.lstm(emb)
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


# ══════════════════════════════════════════════════════════════
#  3. TRAINING
# ══════════════════════════════════════════════════════════════

ARTIFACTS = Path("artifacts")


def train_model(data_path="spam.tsv", epochs=20, batch=32, lr=1e-3,
                max_len=256, patience=5):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report

    ARTIFACTS.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] device={device}")

    sep = "\t" if data_path.endswith(".tsv") else ","
    df = pd.read_csv(data_path, sep=sep, on_bad_lines="skip")
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c in ("message", "sms", "body", "v2", "text_message"):
            rename[c] = "text"
        if c in ("v1", "category", "class", "type", "target"):
            rename[c] = "label"
    if rename:
        df = df.rename(columns=rename)
    df = df[["label", "text"]].copy()
    df["label"] = df["label"].map(
        lambda x: 1 if str(x).lower().strip() == "spam" else 0)
    df = df.dropna()
    print(
        f"[Train] {len(df)} samples | spam={df['label'].sum()} | ham={(df['label'] == 0).sum()}")

    X, y = df["text"].tolist(), df["label"].tolist()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    Xv,  Xt,  yv,  yt = train_test_split(
        Xte, yte, test_size=0.5, random_state=42, stratify=yte)

    vocab = Vocabulary()
    vocab.build(Xtr)
    print(f"[Train] vocab size={len(vocab)}")

    tr_dl = DataLoader(SpamDataset(
        Xtr, ytr, vocab, max_len), batch, shuffle=True)
    vl_dl = DataLoader(SpamDataset(Xv,  yv,  vocab, max_len), batch)
    te_dl = DataLoader(SpamDataset(Xt,  yt,  vocab, max_len), batch)

    model = SpamBiLSTM(len(vocab)).to(device)
    spam_n = sum(ytr)
    ham_n = len(ytr) - spam_n
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, ham_n / spam_n]).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=3, factor=0.5)

    best_f1, wait = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for Xb, yb in vl_dl:
                preds.extend(model(Xb.to(device)).argmax(1).cpu().tolist())
                labs.extend(yb.tolist())
        vf1 = f1_score(labs, preds, average="binary")
        sched.step(total_loss / len(tr_dl))
        print(
            f"[Train] Epoch {ep:02d}/{epochs}  loss={total_loss/len(tr_dl):.4f}  val_f1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1
            wait = 0
            torch.save({"state": model.state_dict(), "val_f1": vf1,
                        "vocab_size": len(vocab)}, str(ARTIFACTS / "model.pt"))
            import app as _app_module


class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


with open(ARTIFACTS / "vocab.pkl", "rb") as f:
    vocab = _VocabUnpickler(f).load()
            print(f"  ✓ Saved (val_f1={vf1:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print("[Train] Early stopping.")
                break

    ckpt = torch.load(str(ARTIFACTS / "model.pt"), map_location=device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for Xb, yb in te_dl:
            tp.extend(model(Xb.to(device)).argmax(1).cpu().tolist())
            tl.extend(yb.tolist())
    print("\n[Train] TEST RESULTS:")
    print(classification_report(tl, tp, target_names=["Ham", "Spam"]))
    return best_f1


# ══════════════════════════════════════════════════════════════
#  4. PREDICTOR
# ══════════════════════════════════════════════════════════════

_predictor = None

def load_predictor():
    global _predictor
    try:
        with open(ARTIFACTS / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        ckpt  = torch.load(str(ARTIFACTS / "model.pt"), map_location="cpu")
        model = SpamBiLSTM(len(vocab), dropout=0.0)
        model.load_state_dict(ckpt["state"])
        model.eval()
        _predictor = {
            "model":  model,
            "vocab":  vocab,
            "val_f1": ckpt.get("val_f1", 0)
        }
        print(f"[App] Model loaded — val_f1={_predictor['val_f1']:.4f}")
    except FileNotFoundError:
        print("[App] No trained model found. Run: python app.py --train")


def predict(body=""):
    if _predictor is None:
        return None
    ids = _predictor["vocab"].encode(body.strip(), 256)
    x   = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        probs = F.softmax(_predictor["model"](x), dim=1).squeeze()
    spam = float(probs[1])
    ham  = float(probs[0])
    return {
        "label":      "spam" if spam >= 0.5 else "ham",
        "spam_score": round(spam, 4),
        "ham_score":  round(ham,  4),
        "confidence": round(max(spam, ham), 4),
    }


def model_info():
    if _predictor is None:
        return {}
    m = _predictor["model"]
    return {
        "architecture": "BiLSTM",
        "vocab_size":   len(_predictor["vocab"]),
        "parameters":   sum(p.numel() for p in m.parameters()),
        "val_f1":       round(_predictor["val_f1"], 4),
        "device":       "cuda" if torch.cuda.is_available() else "cpu",
    }


# ══════════════════════════════════════════════════════════════
#  5. DATABASE
# ══════════════════════════════════════════════════════════════

DB = "history.db"

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db

def init_db():
    with sqlite3.connect(DB) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                body        TEXT,
                label       TEXT,
                spam_score  REAL,
                ham_score   REAL,
                confidence  REAL,
                feedback    TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)


# ══════════════════════════════════════════════════════════════
#  6. FLASK APP + ROUTES
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

# ── Pages ─────────────────────────────────────────────────────
@app.route("/")
def page_home():
    return render_template("home.html")

@app.route("/detect")
def page_detect():
    return render_template("detect.html")

@app.route("/analytics")
def page_analytics():
    return render_template("analytics.html")

@app.route("/about")
def page_about():
    return render_template("about.html")

# ── API ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def route_predict():
    if _predictor is None:
        return jsonify({"error": "Model not ready. Run: python app.py --train"}), 503
    d    = request.get_json()
    body = d.get("body", "").strip()
    if not body:
        return jsonify({"error": "Body is required"}), 400
    r   = predict(body)
    db  = get_db()
    cur = db.execute(
        "INSERT INTO predictions(body,label,spam_score,ham_score,confidence) VALUES(?,?,?,?,?)",
        (body, r["label"], r["spam_score"], r["ham_score"], r["confidence"]))
    db.commit()
    r["id"] = cur.lastrowid
    return jsonify(r)


@app.route("/history")
def route_history():
    label = request.args.get("label")
    page  = int(request.args.get("page", 1))
    pp    = int(request.args.get("per_page", 20))
    db    = get_db()
    base  = "FROM predictions" + (" WHERE label=?" if label else "")
    args  = ([label] if label else [])
    total = db.execute("SELECT COUNT(*) " + base, args).fetchone()[0]
    rows  = db.execute(
        "SELECT * " + base + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
        args + [pp, (page - 1) * pp]).fetchall()
    return jsonify({"predictions": [dict(r) for r in rows], "total": total, "page": page})


@app.route("/feedback", methods=["POST"])
def route_feedback():
    d  = request.get_json()
    db = get_db()
    db.execute("UPDATE predictions SET feedback=? WHERE id=?",
               (d["correct_label"], d["prediction_id"]))
    db.commit()
    return jsonify({"ok": True})


@app.route("/stats")
def route_stats():
    db    = get_db()
    total = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    spam  = db.execute("SELECT COUNT(*) FROM predictions WHERE label='spam'").fetchone()[0]
    avg_c = db.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    return jsonify({
        "total":     total,
        "spam":      spam,
        "ham":       total - spam,
        "spam_rate": round(spam / total, 4) if total else 0,
        "avg_conf":  round(avg_c, 4),
        "model":     model_info(),
    })


@app.route("/history/<int:pid>", methods=["DELETE"])
def route_delete(pid):
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id=?", (pid,))
    db.commit()
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════
#  7. ENTRY POINT  —  works for both local and Render/gunicorn
# ══════════════════════════════════════════════════════════════

# Always initialise the DB and load the model when the module is imported.
# This covers gunicorn (which imports the module directly, never __main__).
init_db()

# Only train / parse CLI args when run directly with `python app.py`
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  action="store_true")
    parser.add_argument("--data",   default="spam.tsv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    if args.train or not (ARTIFACTS / "model.pt").exists():
        if not Path(args.data).exists():
            print(f"[Error] Dataset not found: {args.data}")
            sys.exit(1)
        train_model(data_path=args.data, epochs=args.epochs)

    load_predictor()
    print(f"\n🛡  SpamShield → http://localhost:{args.port}\n")
    port = int(os.environ.get("PORT", args.port))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # Imported by gunicorn — just load the pre-trained model
    load_predictor()