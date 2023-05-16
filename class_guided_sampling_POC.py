import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from hparams import get_sampler_hparams
from utils import get_sampler, load_model, EMA, SubseqSampler
from utils.eval_utils import get_rand_dataset_subset, evaluate_consistency_variance
from utils.sampler_utils import get_samples, np_to_ns
from note_seq import note_sequence_to_midi_file
from note_density_estimator.key_estimator import SimpleNoteDensityEstimator


MAGENTA_CLASSES = 90
MAGENTA_PITCH_OFFSET = -19
def to_features(npy):
    if npy.shape[-1] > 1:
        npy = npy[:, :, 0]
    pitches = [torch.nn.functional.one_hot(torch.tensor(n).to(torch.int64), MAGENTA_CLASSES) for n in npy[0, :]]
    return torch.stack(pitches).unsqueeze(0)

def get_guided_samples(sampler, sample_steps, guide, x_T=None, temp=1.0, b=None, eta=0.3):
    sampler.eval()

    if x_T is not None and not torch.is_tensor(x_T):
        x_T = torch.tensor(x_T).to(next(sampler.parameters()).device)

    result = sampler.guided_sample(guide, eta=eta, sample_steps=sample_steps, x_T=x_T, temp=temp, B=b)
    return result.cpu().numpy()


if __name__ == '__main__':
    H = get_sampler_hparams('sample')
    H.sample_schedule = "rand"
    sampler = get_sampler(H).cuda()
    sampler = load_model(
                sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, strict=True)

    midi_data = np.load(H.dataset_path, allow_pickle=True)
    midi_data = SubseqSampler(midi_data, H.NOTES)

    ema = EMA(H.ema_beta)
    ema_sampler = copy.deepcopy(sampler)

    n_samples = 0
    sampler.sampling_batch_size = 1

    REC_IN_OUT_SIZE = 128
    input_size = 90
    BATCH_SIZE = 12
    sampling_batch_size = 4
    model = SimpleNoteDensityEstimator().to("cuda")
    model.load_state_dict(torch.load('note_density_estimator.pth'))

    def guide(target, x):
        b = x.shape[0]
        x = x.view(-1, 16, x.shape[-1])
        y = model(x)
        loss = torch.nn.functional.cross_entropy(y, torch.tensor(target).view(x.shape[0]).cuda(), reduction='none')
        #loss = y[target]
        return loss.view(b, -1).mean(-1)

    targets = [i for i in range(1, 17) for _ in range(2)]
    targets = targets + list(reversed(targets))

    unguided_samples = []#acc bs 1 = .6745 | 6 = .53
    unguided_accs = []
    diffs = []
    for _ in tqdm(range(int(np.ceil(BATCH_SIZE / sampling_batch_size)))):
        targets = np.random.randint(6, 12, sampling_batch_size)#np.array([8] * sampling_batch_size)
        targets = targets.repeat(64)
        sa = get_samples(ema_sampler, 1024, b=sampling_batch_size)
        ns = np_to_ns(sa)

        actual_nd = sa[:, :, 0].reshape(-1, 16)
        actual_nd = (actual_nd > 1).sum(-1)
        acc = (actual_nd == targets).mean()

        diffs.append((actual_nd.mean() - targets.mean()) ** 2)
        unguided_samples.append(sa)
        unguided_accs.append(acc)

    samples = np.concatenate(unguided_samples)
    samples = np_to_ns(samples[:BATCH_SIZE])

    originals = get_rand_dataset_subset(midi_data, BATCH_SIZE)
    originals = np_to_ns(originals)

    c, v = evaluate_consistency_variance(originals, samples)

    print("C", c, "V", v, "accs:", np.array(unguided_accs).mean(), "diffs:", np.array(diffs).mean())
