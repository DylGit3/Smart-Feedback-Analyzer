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

# Load and Preprocessing

df = pd.read_csv('data/test.csv', names=['label', 'title', 'text'])
df.dropna(inplace=True)
df['label'] = df['label'] - 1
df['combined_text'] = df['title'] + ' ' + df['text']

texts = df['combined_text'].tolist()
labels = df['label'].tolist()

# Tokenization and Vocabulary


def tokenize(text: str) -> str:
    return text.lower().split()


vocab = Counter()

for text in texts:
    vocab.update(tokenize(text))

word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}
# Pads shorter sequences so they’re the same length when passed to the model
word_to_idx['<PAD>'] = 0


def encode(text: str) -> list:
    return [word_to_idx.get(token, 0) for token in tokenize(text)]

# Dataset and loader
