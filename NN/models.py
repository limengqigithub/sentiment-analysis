import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=256, num_class=3):
        super(NN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.linear1 = nn.Linear(hidden_size, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, num_class)

    def forward(self, x):
        x1 = self.embedding(x)
        x1 = torch.mean(x1, dim=1)
        x2 = self.linear1(x1)
        x3 = self.relu(x2)
        x4 = self.linear2(x3)
        return x4