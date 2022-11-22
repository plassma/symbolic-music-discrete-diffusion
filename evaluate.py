import argparse
import copy

from utils import evaluate, get_sampler, load_model, EMA, log
from hparams import get_sampler_hparams


if __name__ == '__main__':
    H = get_sampler_hparams('eval')

    if (H.gap_start < 0 or H.gap_end < 0) and H.mode == 'infilling':
        H.gap_start = H.NOTES // 4
        H.gap_end = (H.NOTES * 3) // 4
        log(f'Gap not specified - masking out {H.gap_start} to {H.gap_end}')

    sampler = get_sampler(H).cuda()
    sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
    ema = EMA(H.ema_beta)
    ema_sampler = copy.deepcopy(sampler)

    try:
        ema_sampler = load_model(
            ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
    except Exception as e:
        ema_sampler = copy.deepcopy(sampler)

    evaluate(H, ema_sampler if H.ema else sampler)
