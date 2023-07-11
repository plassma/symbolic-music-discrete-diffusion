import time

import numpy as np
from note_seq import note_sequence_to_midi_file, midi_file_to_note_sequence

from hparams import get_sampler_hparams
from utils import get_sampler, load_model
from utils.sampler_utils import get_samples, np_to_ns, ns_to_np


def sample_cont(sampler, H):
    x_T = None
    batch = 1
    if H.piece:
        ns = midi_file_to_note_sequence(H.piece)
        bars = min(64, int(max([n.end_time for n in ns.notes]) // 2))
        npy = ns_to_np(ns, bars, 'melody').outputs[0]

        x_T = np.zeros((batch, H.NOTES, 3), dtype=int)

        x_T[:, :] = H.codebook_size
        x_T[:, :npy.shape[0], 0] = npy[:, 0]

        if npy.shape[1] == 3:
            x_T[:, :npy.shape[0], 1] = npy[:, 1]
            x_T[:, :npy.shape[0], 2] = npy[:, 2]

    n_samples = 0
    sampler.sampling_batch_size = batch
    piece = None
    while n_samples < H.n_samples:
        sa = get_samples(sampler, H.sample_steps, x_T)

        if piece is None:
            piece = sa
        else:
            piece = np.append(piece, sa[:, sa.shape[1] // 2:], axis=1)

        x_T = np.zeros((batch, H.NOTES, 3), dtype=int)
        x_T[:, :] = H.codebook_size
        x_T[:, :sa.shape[1] // 2] = sa[:, sa.shape[1] // 2:]
        ns = np_to_ns(piece)

        for _ in ns:
            n_samples += 1
        print(f'{n_samples}/{H.n_samples}')
    for n in ns:
        note_sequence_to_midi_file(n, f'data/out/conti{time.time()}.mid')


if __name__ == '__main__':
    H = get_sampler_hparams('sample')
    H.sample_schedule = "rand"
    sampler = get_sampler(H).cuda()
    sampler = load_model(
                sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)

    sample_cont(sampler, H)