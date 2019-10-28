

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
        self.lstm_dim = 128
        self.lstm_n_layer = 2
        self.n_linear = 2
        self.n_directions = 2
        self.lstm_combine = 'add'
        self.lstm_concated_dim = self.n_directions * self.lstm_dim
