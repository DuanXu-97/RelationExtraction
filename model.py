import torch
from torch import nn
import torch.nn.functional as F


class AttBiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.sequence_len = config.sequence_len
        self.embedding_size = config.embedding_size
        self.lstm_dim = config.lstm_dim
        self.lstm_combined_dim = config.lstm_combined_dim
        self.lstm_n_layer = config.lstm_n_layer
        self.lstm_combine = config.lstm_combine
        self.n_linear = config.n_linear
        self.n_linear_ent = config.n_linear_ent
        self.hidden_dim_ent = config.hidden_dim_ent
        self.n_directions = config.n_directions
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate
        self.attention_type = config.attention_type
        self.linear_concated_dim = config.linear_concated_dim
        self.with_ent = config.with_ent

        self.criterion = nn.CrossEntropyLoss()

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.lstm_dim,
                            num_layers=self.lstm_n_layer,
                            bidirectional=True,
                            batch_first=True)

        self.lstm_dropout = nn.Dropout(p=self.dropout_rate)

        self.attention_weights_1 = nn.Parameter(torch.randn(1, self.lstm_combined_dim, 1))
        self.attention_weights_2 = nn.Parameter(torch.randn(1, self.lstm_combined_dim, 1))

        self.linear_layers = nn.ModuleList([nn.Linear(self.linear_concated_dim, self.linear_concated_dim)
                                            for _ in range(self.n_linear - 1)])

        self.linear_layer_ent = nn.Linear(self.embedding_size, self.hidden_dim_ent)

        self.linear_dropout = nn.Dropout(p=self.dropout_rate)

        self.sm_layer = nn.Linear(self.linear_concated_dim, self.num_classes)

    def attention(self, lstm_output):
        # # (batch_size, n_directions*lstm_n_layer, lstm_dim)->
        # # (n_directions, batch_size, lstm_dim)
        # h_n = h_n.view(self.lstm_n_layer, self.n_directions, self.batch_size, self.lstm_dim)[-1]
        # # (n_directions, batch_size, lstm_dim)->(batch_size, n_directions, lstm_dim)
        # h_n = h_n.permute(1, 0, 2)
        # # (batch_size, n_directions, lstm_dim)->(batch_size, 1, lstm_dim)
        # h_n = h_n.sum(dim=1)

        if self.lstm_combine == 'add':
            # (batch_size, sequence_len, lstm_combined_dim)->(batch_size, sequence_len, 2, lstm_dim)
            lstm_output = lstm_output.view(self.batch_size, self.sequence_len, 2, self.lstm_dim)
            # (batch_size, sequence_len, 2, lstm_dim)->(batch_size, sequence_len, 1, lstm_dim)
            lstm_output = lstm_output.sum(dim=2)

        if self.attention_type == 'double':
            # (batch_size, sequence_len, lstm_combined_dim),(batch_size, lstm_combined_dim, 1)
            # ->(batch_size, sequence_len, 1)
            att_1 = torch.bmm(torch.tanh(lstm_output), self.attention_weights_1.repeat(self.batch_size, 1, 1))
            att_2 = torch.bmm(torch.tanh(lstm_output), self.attention_weights_2.repeat(self.batch_size, 1, 1))
            att = torch.mean(torch.stack((att_1, att_2), dim=0), dim=0)
        else:
            # (batch_size, sequence_len, lstm_combined_dim),(batch_size, lstm_combined_dim, 1)
            # ->(batch_size, sequence_len, 1)
            att = torch.bmm(torch.tanh(lstm_output), self.attention_weights_1.repeat(self.batch_size, 1, 1))

        att = F.softmax(att, dim=1)
        # (batch_size, sequence_len, lstm_combined_dim)->(batch_size, lstm_combined_dim, sequence_len)
        lstm_output = lstm_output.transpose(1, 2)
        # (batch_size, lstm_combined_dim, sequence_len),(batch_size, sequence_len, 1)
        # ->(batch_size, lstm_combined_dim)
        att = torch.bmm(lstm_output, att).squeeze(2)

        output = torch.tanh(att)

        return output

    def forward(self, x, ent1, ent2):
        self.batch_size, self.sequence_len = x.shape
        # (batch_size, sequence_len, embedding_dim)
        x = self.embedding_layer(x)

        # lstm_output.shape = (batch_size, sequence_len, lstm_combined_dim)
        # h_n.shape = (batch_size, n_directions*lstm_n_layer, lstm_dim)
        lstm_output, (h_n, c_n) = self.lstm(x)
        lstm_output = self.lstm_dropout(lstm_output)

        x = self.attention(lstm_output)

        if self.with_ent is True:
            ent1 = self.embedding_layer(ent1)
            ent2 = self.embedding_layer(ent2)
            ent1 = self.linear_layer_ent(ent1)
            ent2 = self.linear_layer_ent(ent2)

            ent1 = ent1.squeeze(dim=1)
            ent2 = ent2.squeeze(dim=1)


            x = torch.cat((x, ent1, ent2), dim=-1)

        for layer in self.linear_layers:
            x = layer(x)
            x = self.linear_dropout(x)
            x = F.relu(x)

        logits = self.sm_layer(x)

        return logits

    def loss(self, input, target, ent1, ent2):
        logits = self.forward(input, ent1, ent2)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.criterion(logits_flat, target_flat)  # mean_score per batch
        return loss

    def predict(self, input, ent1, ent2):
        logits = self.forward(input, ent1, ent2)
        logits[:, :2] = float('-inf')
        preds = logits.max(dim=-1)[1]
        preds = preds.detach().cpu().numpy().tolist()
        return preds

    def loss_n_acc(self, input, target, ent1, ent2):
        logits = self.forward(input, ent1, ent2)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.criterion(logits_flat, target_flat)  # mean_score per batch

        pred_flat = logits_flat.max(dim=-1)[1]
        acc = (pred_flat == target_flat).sum()
        return loss, acc.item()













