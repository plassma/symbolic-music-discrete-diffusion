import torch
import torch.nn as nn


class SimpleNoteDensityEstimator(nn.Module):
    def __init__(self, classes=17):
        nn.Module.__init__(self)
        self.sequential = nn.Sequential(nn.Linear(90, 128), nn.ReLU(), nn.Linear(128, 1024), nn.ReLU(),
                                        nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, classes), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(16 * classes, classes))

    def forward(self, x):
        x = self.sequential(x)
        return self.head(x.view(x.shape[0], -1))


class NoteDensityEstimator(nn.Module):
    """
    A Simple Recurrent Model
    """

    def __init__(self,
                 input_size,
                 recurrent_size, hidden_size,
                 dropout=0.0,
                 classes=24,
                 batch_first=True,
                 dtype=torch.float32,
                 rnn_layer=nn.LSTM,
                 device=None):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.n_layers = 1
        self.batch_first = batch_first
        self.device = device if device is not None else torch.device('cpu')
        self.to(self.device)
        self.dtype = dtype
        self.rnn = rnn_layer(hidden_size,
                             self.recurrent_size,
                             self.n_layers,
                             batch_first=batch_first,
                             dropout=dropout,
                             bidirectional=False)
        dense_in_features = self.recurrent_size
        self.dense = nn.Linear(in_features=dense_in_features,
                               out_features=self.hidden_size)
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=self.output_size)
        self.head = nn.Linear(self.input_size, classes)
        self.emb = nn.Linear(input_size, hidden_size)

    def init_hidden(self, batch_size):

        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.n_layers, batch_size, self.recurrent_size).to(self.dtype).to(self.device)
            c0 = torch.zeros(self.n_layers, batch_size, self.recurrent_size).to(self.dtype).to(self.device)
            return (h0, c0)
        else:
            return torch.zeros(self.n_layers, batch_size, self.recurrent_size).to(self.dtype)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = self.init_hidden(batch_size)
        x = self.emb(x)
        output, h = self.rnn(x, h0)
        flatten_shape = self.recurrent_size
        dense = self.dense(output.contiguous().view(-1, flatten_shape))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)
        y = self.head(y[:, -1])
        return y.squeeze()

    def forward_h(self, x):
        batch_size = x.size(0)
        h0 = self.init_hidden(batch_size)
        output, (h, c) = self.rnn(x, h0)

        return h