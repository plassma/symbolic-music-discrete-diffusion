import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.latent_shape = H.latent_shape
        self.emb_dim = H.emb_dim
        self.codebook_size = H.codebook_size
        self.n_samples = H.n_samples

    def train_iter(self, x, x_target, step):
        raise NotImplementedError()

    def sample(self, x_T=None):
        raise NotImplementedError()

    def class_conditional_train_iter(self, x, y):
        raise NotImplementedError()

    def class_conditional_sample(n_samples, y):
        raise NotImplementedError()
