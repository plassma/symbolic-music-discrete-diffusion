import copy

import hparams
from utils import evaluate, get_sampler, load_model, EMA

N_SAMPLES = 256


if __name__ == '__main__':
    H = hparams.HparamsAbsorbingConv('Lakh', 64)
    H.n_samples = N_SAMPLES

    H.ema = True
    H.gap_start = H.NOTES // 4
    H.gap_end = (H.NOTES * 3) // 4

    sampler = get_sampler(H).cuda()
    load_step = 690000
    sampler = load_model(sampler, H.sampler, load_step, H.load_dir).cuda()
    sampler.eval()
    ema = EMA(H.ema_beta)
    ema_sampler = copy.deepcopy(sampler)

    try:
        ema_sampler = load_model(
            ema_sampler, f'{H.sampler}_ema', load_step, H.load_dir)
    except Exception as e:
        ema_sampler = copy.deepcopy(sampler)

    evaluate(H, ema_sampler if H.ema else sampler, 'data/lakh_melody_64_1MIO.npy', mode='unconditional',
             batches=5, evaluations_per_batch=10, batch_size=1000)
