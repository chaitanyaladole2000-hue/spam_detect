import re, pickle
from collections import Counter
from typing import List
import torch
from torch.utils.data import Dataset

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"[^\w\s<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> List[str]:
    return clean(text).split()

class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, max_size=30000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        self.token2idx = {self.PAD: 0, self.UNK: 1}
        for token, freq in counter.most_common(self.max_size - 2):
            if freq < self.min_freq:
                break
            self.token2idx[token] = len(self.token2idx)
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode(self, text, max_len=256):
        ids = [self.token2idx.get(t, 1) for t in tokenize(text)[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.token2idx)

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f: return pickle.load(f)


class SpamDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.X = [vocab.encode(t, max_len) for t in texts]
        self.y = labels

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.long))
