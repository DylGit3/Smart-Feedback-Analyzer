import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os

# --- Load and Preprocessing ---

df = pd.read_csv('data/test.csv', names=['label', 'title', 'text'])
df.dropna(inplace=True)
df['label'] = df['label'] - 1
df['combined_text'] = df['title'] + ' ' + df['text']

texts = df['combined_text'].tolist()
labels = df['label'].tolist()

# --- Tokenization and Vocabulary ---


def tokenize(text: str) -> str:
    return text.lower().split()


vocab = Counter()

for text in texts:
    vocab.update(tokenize(text))

word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}
# Pads shorter sequences so they are the same length when passed to the model
word_to_idx['<PAD>'] = 0


def encode(text: str) -> list:
    return [word_to_idx.get(token, 0) for token in tokenize(text)]

# --- Dataset and loader ---


class FeedbackDataset(Dataset):
    def __init__(self, texts, labels):
        self.inputs = [torch.tensor(encode(text)) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def collate_batch(batch):
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    labels = torch.tensor(labels)
    return inputs, labels

# --- Model Definition ---


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)  # Dense vector/lookup table
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return torch.sigmoid(self.fc(pooled)).squeeze(1)

# --- Training ---


X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.1, random_state=40)
train_loader = DataLoader(FeedbackDataset(
    X_train, y_train), batch_size=64, shuffle=True, collate_fn=collate_batch)
