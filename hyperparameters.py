

class AttBiLSTMHP:
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.epochs = 500
        self.num_classes = 10

        self.embedding_dropout_rate = 0.3
        self.lstm_dropout_rate = 0.3
        self.linear_dropout_rate = 0.5

        self.vocab_size = 100000
        self.sequence_len = 100
        self.embedding_size = 100
        self.embedding_vectors = None
        self.lstm_dim = 100
        self.lstm_n_layer = 2
        self.n_linear = 2
        self.n_linear_ent = 1
        self.hidden_dim_ent = 16
        self.n_directions = 2
        self.lstm_combine = 'add'
        self.attention_type = 'single'
        self.lstm_combined_dim = self.n_directions * self.lstm_dim if self.lstm_combine == 'concat' else self.lstm_dim

        self.with_ent = True
        self.linear_concated_dim = self.lstm_combined_dim + 2*self.hidden_dim_ent if self.with_ent is True else self.lstm_combined_dim



