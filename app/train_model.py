from collections import Counter
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- Load and Preprocessing ---

df = pd.read_csv('data/test.csv', names=['label', 'title', 'text'])
df.dropna(inplace=True)
df['label'] = df['label'].astype(int) - 1
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
    inputs_batch, labels_batch = zip(*batch)
    inputs_batch = pad_sequence(inputs_batch, batch_first=True)
    labels_batch = torch.tensor(labels_batch)
    return inputs_batch, labels_batch

# --- Model Definition ---


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)  # Dense vector/lookup table
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        h = torch.relu(self.fc1(pooled))
        return self.fc2(h).squeeze(1)

# --- Training ---


X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.1, random_state=40)

train_loader = DataLoader(FeedbackDataset(
    X_train, y_train), batch_size=64, shuffle=True, collate_fn=collate_batch)

val_loader = DataLoader(
    FeedbackDataset(X_val, y_val),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_batch
)

model = SentimentModel(vocab_size=len(word_to_idx))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(3):
    total_loss = 0
    total_samples = 0
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1}, batch {batch_idx}, batch loss: {loss.item():.4f}")

    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # === INSERTED VALIDATION PASS ===
    model.eval()
    val_loss = 0
    val_samples = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            logits = model(inputs)
            loss = criterion(logits, targets.float())

            bs = targets.size(0)
            val_loss += loss.item() * bs
            val_samples += bs

            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == targets).sum().item()
            total += bs

    avg_val_loss = val_loss / val_samples
    val_accuracy = correct / total
    print(
        f"Epoch {epoch+1} val loss: {avg_val_loss:.4f}, val acc: {val_accuracy:.4f}\n")

    model.train()

# --- Saving Model and Vocab ---

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/sentiment_model.pt")
torch.save(word_to_idx, "model/word-to-idx.pt")
