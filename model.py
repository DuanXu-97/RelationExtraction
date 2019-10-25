import torch
from torch import nn
import torch.nn.functional as F


class AttentionBasedBiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.sequence_len = config.sequence_len
        self.embedding_size = config.embedding_size
        self.lstm_dim = config.lstm_dim
        self.lstm_concated_dim = config.lstm_concated_dim
        self.lstm_n_layer = config.lstm_n_layer
        self.n_linear = config.n_linear
        self.n_directions = config.n_directions
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.lstm_dim,
                            num_layers=self.lstm_n_layer,
                            bidirectional=True,
                            batch_first=True)

        self.lstm_dropout = nn.Dropout(p=self.dropout_rate)
        self.attention_weights = nn.Parameter(torch.randn(1, self.lstm_concated_dim, 1))
        self.linear_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.lstm_concated_dim, self.lstm_concated_dim),
                                                          nn.Dropout(p=self.dropout_rate),
                                                          F.relu())
                                            for _ in range(self.n_linear - 1)])
        self.linear_dropout = nn.Dropout(p=self.dropout_rate)

        self.sm_layer = nn.Linear(self.lstm_concated_dim, self.num_classes)

    def attention(self, lstm_output, h_n, x):
        # # (batch_size, n_directions*lstm_n_layer, lstm_dim)->
        # # (n_directions, batch_size, lstm_dim)
        # h_n = h_n.view(self.lstm_n_layer, self.n_directions, self.batch_size, self.lstm_dim)[-1]
        # # (n_directions, batch_size, lstm_dim)->(batch_size, n_directions, lstm_dim)
        # h_n = h_n.permute(1, 0, 2)
        # # (batch_size, n_directions, lstm_dim)->(batch_size, 1, lstm_dim)
        # h_n = h_n.sum(dim=1)

        # (batch_size, sequence_len, lstm_concated_dim),(batch_size, lstm_concated_dim, 1)
        # ->(batch_size, sequence_len, 1)
        att = torch.bmm(torch.tanh(lstm_output), self.att_w.repeat(self.batch_size, 1, 1))
        att = F.softmax(att, dim=1)
        # (batch_size, sequence_len, lstm_concated_dim)->(batch_size, lstm_concated_dim, sequence_len)
        lstm_output = lstm_output.transpose(1, 2)
        # (batch_size, lstm_concated_dim, sequence_len),(batch_size, sequence_len, 1)
        # ->(batch_size, lstm_concated_dim)
        att = torch.bmm(lstm_output, att).squeeze(2)

        output = torch.tanh(att)

        return output

    def forward(self, x):
        # (batch_size, sequence_len, embedding_dim)
        x = self.embedding_layer(x)

        # lstm_output.shape = (batch_size, sequence_len, lstm_concated_dim)
        # h_n.shape = (batch_size, n_directions*lstm_n_layer, lstm_dim)
        lstm_output, (h_n, c_n) = self.lstm(x)
        lstm_output = self.lstm_dropout(lstm_output)

        x = self.attention(lstm_output, h_n, x)

        x = self.linear_layers(x)

        logits = self.sm_layer(x)

        return logits












