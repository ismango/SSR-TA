import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, pad_idx, hidden_dim, n_layers, dropout, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        # text = [batch_size, seq_len]

        # embed = [batch_size, seq_len, embedding_dim]
        embed = self.embedding(text)
        output, _ = self.rnn(embed)
        pred = self.fc(output[:, -1, :])
        return pred

