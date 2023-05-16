import numpy as np
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def optim_warmup(H, step, optim):
    lr = H.lr * float(step) / H.warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def augment_note_tensor(H, batch):
    if H.augment:
        for i in range(len(batch)):
            x = batch[i]
            x = x[:, :2]
            x = x[x > 1]
            mi, ma = -x.min() + 2, H.codebook_size[0] - x.max()
            shift = np.random.randint(mi, ma)
            batch[i, :, :2][batch[i, :, :2] > 1] += shift
    return batch
