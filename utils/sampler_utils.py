import numpy as np
import torch
import torch.distributions as dists
from torch.nn import DataParallel

# from .log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion, ConVormer
from preprocessing import OneHotMelodyConverter, TrioConverter


def get_sampler(H):
    if H.model == 'transformer':
        denoise_fn = Transformer(H).cuda()
    else:
        denoise_fn = ConVormer(H).cuda()

    denoise_fn = DataParallel(denoise_fn).cuda()
    sampler = AbsorbingDiffusion(
        H, denoise_fn, H.codebook_size)

    return sampler


@torch.no_grad()
def get_samples(sampler, sample_steps, x_T=None, temp=1.0, b=None, progress_handler=None):
    sampler.eval()

    if x_T is not None and not torch.is_tensor(x_T):
        x_T = torch.tensor(x_T).to(next(sampler.parameters()).device)

    result = sampler.sample(sample_steps=sample_steps, x_T=x_T, temp=temp, B=b, progress_handler=progress_handler)
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


def ns_to_np(ns, bars, mode='melody'):
    if mode == 'melody':
        converter = OneHotMelodyConverter(slice_bars=bars)
    else:
        converter = TrioConverter(slice_bars=bars)
    return converter.to_tensors(ns)
