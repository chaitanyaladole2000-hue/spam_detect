from preprocess import Vocabulary
from model import SpamBiLSTM
import torch.nn.functional as F
import torch
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


ARTIFACTS = Path(__file__).parent / "artifacts"


class SpamPredictor:
    _inst = None

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocabulary.load(ARTIFACTS / "vocab.json") as f:
    vocab = Vocabulary()
    vocab.token2idx = json.load(f)
    ckpt = torch.load(str(ARTIFACTS / "best_model.pt"),
    map_location=self.device)
    a = ckpt["args"]
    self.model = SpamBiLSTM(len(self.vocab), a.get("embed_dim", 128),
                            a.get("hidden_dim", 256), a.get("layers", 2), dropout=0.0).to(self.device)
    self.model.load_state_dict(ckpt["model_state"])
    self.model.eval()
    self.max_len = a.get("max_len", 256)
    self.val_f1 = ckpt.get("val_f1", 0)
    print(
        f"[Predictor] Ready — val_f1={self.val_f1:.4f} device={self.device}")

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def predict(self, subject="", body="", sender=""):
        text = f"{subject} {body} {sender}".strip()
        ids = self.vocab.encode(text, self.max_len)
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1).squeeze()
        spam = float(probs[1])
        ham = float(probs[0])
        return {
            "label":      "spam" if spam >= 0.5 else "ham",
            "spam_score": round(spam, 4),
            "ham_score":  round(ham,  4),
            "confidence": round(max(spam, ham), 4),
        }

    def info(self):
        return {
            "architecture": "BiLSTM",
            "vocab_size":   len(self.vocab),
            "parameters":   sum(p.numel() for p in self.model.parameters()),
            "val_f1":       round(self.val_f1, 4),
            "device":       str(self.device),
        }
