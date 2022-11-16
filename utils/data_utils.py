import numpy as np


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class SubseqSampler:
    def __init__(self, dataset, seq_len):
        self.dataset = dataset
        self.seq_len = seq_len
    def __getitem__(self, item):
        if self.seq_len == self.dataset.shape[1]:
            return self.dataset[item]
        seq_start = np.random.randint(0, self.dataset.shape[1] - self.seq_len)
        return self.dataset[item][:, seq_start:seq_start+self.seq_len]

    def __len__(self):
        return len(self.dataset)
