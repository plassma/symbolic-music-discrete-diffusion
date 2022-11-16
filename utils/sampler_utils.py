import os
import torch
from tqdm import tqdm
import numpy as np
#from .log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion, ConVormer
from torch.nn import DataParallel
import torch.distributions as dists
from transformers import BigBirdConfig, BigBirdModel

from preprocessing import OneHotMelodyConverter, TrioConverter


def get_sampler(H):

    if H.sampler == 'absorbing':
        denoise_fn = ConVormer(H).cuda()

        denoise_fn = DataParallel(denoise_fn).cuda()
        sampler = AbsorbingDiffusion(
            H, denoise_fn, H.codebook_size)

    return sampler

@torch.no_grad()
def get_samples(sampler, sample_steps, x_T=None, temp=1.0):
    sampler.eval()

    if x_T is not None and not torch.is_tensor(x_T):
        x_T = torch.tensor(x_T).to(next(sampler.parameters()).device)

    result = sampler.sample(sample_steps=sample_steps, x_T=x_T, temp=temp)
    return result.cpu().numpy()

def np_to_ns(x):
    if x.shape[-1] == 1:
        converter = OneHotMelodyConverter()
        return converter.from_tensors(x.squeeze())
    elif x.shape[-1] == 3:
        converter = TrioConverter()
        return converter.from_tensors(x)
    else:
        raise Exception(f"unsupported number of tracks: {x.shape[-1]}")

@torch.no_grad()
def sample_interleaved(b, samplers, x_T=None, temp=1.0, sample_steps=None, unmasked=None, strides=None):

    #assert maskid equal, first should be largest sampler

    if strides is None:
        strides = [s.shape[0] // 4 for s in samplers]
    for i, s in enumerate(samplers):
        assert s.shape[0] % strides[i] == 0

    device ='cuda'
    x_t = x_T if x_T is not None else torch.ones((b, *samplers[0].shape), device=device).long() * samplers[0].mask_id

    if unmasked is None:
        unmasked = torch.zeros_like(x_t, device=device).bool()

    sample_steps = list(range(1, sample_steps+1))

    for timestep in reversed(sample_steps):
        print(f'Sample timestep {timestep:4d}', end='\r')

        t = torch.full((b,), timestep, device=device, dtype=torch.long)

        # where to unmask
        changes = torch.rand(x_t.shape, device=device) < 1/t.float().view(-1, *((1,) * (len(x_t.shape) - 1)))
        # don't unmask somewhere already unmasked
        changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
        # update mask with changes
        unmasked = torch.bitwise_or(unmasked, changes)

        self = samplers[timestep % len(samplers)]
        windows = int(np.ceil((x_t.shape[1] - self.shape[0]) / strides[timestep % len(samplers)])) + 1
        x_0_window_logits = []
        for i in range(windows):
            end = min(x_t.shape[1], self.shape[0] + i * strides[timestep % len(samplers)])
            x_0_window_logits.append(self._denoise_fn(x_t[:, end - self.shape[0]:end], t=t))

        x_0_logits = [torch.zeros((b, x_t.shape[1], c), device=device) for c in self.codebook_size]

        for c in range(len(self.codebook_size)):
            counts = torch.zeros_like(x_0_logits[c][0])
            for w in range(windows):
                start = w * strides[timestep % len(samplers)]
                end = min(start + self.shape[0], x_t.shape[1])
                x_0_logits[c][:, start:end] += x_0_window_logits[w][c][:, :end-start]  # joint probability over all windows
                counts[start:end] += 1
            x_0_logits[c] /= counts  # maintain softmax temperature


        # scale by temperature
        x_0_logits = [x / temp for x in x_0_logits]
        x_0_dist = [dists.Categorical(
            logits=x) for x in x_0_logits]
        x_0_hat = torch.stack([xd.sample().long() for xd in x_0_dist], -1)

        x_t[changes] = x_0_hat[changes]

    return x_t.cpu().numpy()