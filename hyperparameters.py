

class AttBiLSTMHP:
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.dropout_rate = 0.1
        self.epochs = 500
        self.num_classes = 10

        self.vocab_size = 100000
        self.sequence_len = 100
        self.embedding_size = 100
        self.lstm_dim = 100
        self.lstm_n_layer = 1
        self.n_linear = 1
        self.n_directions = 2
        self.lstm_combine = 'add'
        self.attention_type = 'double'
        self.lstm_combined_dim = self.n_directions * self.lstm_dim if self.lstm_combine == 'concat' else self.lstm_dim


