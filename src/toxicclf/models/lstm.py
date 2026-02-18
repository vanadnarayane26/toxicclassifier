import torch
import torch.nn as nn

from toxicclf.constants import NUM_LABELS

class LSTMClassifier(nn.Module):
    def __init__(self, 
                 vocab_size:int,
                 embedding_dim:int,
                 hidden_dim:int,
                 num_layers:int,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size =hidden_dim, 
                            num_layers = num_layers, 
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, NUM_LABELS)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden,cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            # hidden shape: (num_layers * num_directions, batch, hidden_size)
            # take the last layer's forward and backward hidden states
            h_forward = hidden[-2, :, :]  # (batch, hidden_size)
            h_backward = hidden[-1, :, :]  # (batch, hidden_size)
            hidden_cat = torch.cat((h_forward, h_backward), dim=1)  # (batch, hidden*2)
        else:
            hidden_cat = hidden[-1, :, :]  # (batch, hidden_size)

        out = self.dropout(hidden_cat)
        logits = self.fc(out)
        return logits