import copy
import itertools
from statistics import NormalDist

import numpy as np
import torch
from note_seq import midi_to_note_sequence, quantize_note_sequence
from note_seq import sequences_lib

import hparams
from preprocessing import TrioConverter
from utils import load_model
from utils.sampler_utils import get_samples, get_sampler
from utils.train_utils import EMA

N_SAMPLES = 4


def prep_piece(bars, path="mario_theme.mid"):
    converter = TrioConverter(bars)
    ns = midi_to_note_sequence(open(path, 'rb').read())
    tensors = list(converter.to_tensors(ns).outputs)
    tensors = [t.squeeze() for t in tensors]

    melody = tensors[0]
    x_T = torch.ones((N_SAMPLES, melody.shape[0], 3), dtype=torch.long) * torch.tensor((90, 90, 512))
    x_T[:, :, 0] = torch.tensor(melody[:, 0])

    x_T[:, :256] = 90
    x_T[:, 768:] = 90

    mask = torch.zeros_like(x_T)
    mask[:, :, 0] = 1
    mask[:, 256:768] = 0

    return x_T, mask.bool(), tensors[0]


def frame_statistics(bars):
    bars = list(itertools.chain(*bars))
    stats = lambda x: NormalDist(np.mean(x), np.std(x))
    return stats([n.pitch for n in bars]), stats([n.quantized_end_step - n.quantized_start_step for n in bars])


def framewise_overlap_areas(ns, width=4, hop=2):
    qns = quantize_note_sequence(ns, 4)
    steps_per_bar = sequences_lib.steps_per_bar_in_quantized_sequence(qns)
    assert steps_per_bar == 16.
    steps_per_bar = 16
    print(steps_per_bar)

    by_bar = [[] for _ in range(max([n.quantized_end_step for n in qns.notes]) // steps_per_bar + 1)]

    for note in qns.notes:
        k = note.quantized_start_step // steps_per_bar
        by_bar[k].append(note)
        if note.quantized_end_step // steps_per_bar != k:#todo: how 2 handle bar crossing notes?
            by_bar[note.quantized_end_step // steps_per_bar].append(note)

    frames = []

    for f in range((len(by_bar) - width) // hop):
        start_bar = hop * f
        frames.append(frame_statistics(by_bar[start_bar:start_bar+width]))

    OAs = []
    for i in range(len(frames) - 1):
        if all([(frames[i][j].variance and frames[i+1][j].variance) for j in [0, 1]]):
            OAs.append([frames[i][j].overlap(frames[i+1][j]) for j in [0, 1]])

    return np.array(OAs)


def evaluate_consistency_variance(target, pred):
    OA_t, OA_p = framewise_overlap_areas(target), framewise_overlap_areas(pred)

    consistency = 1 - np.abs(OA_t.mean(0) - OA_p.mean(0)) / OA_t.mean(0)
    variance = 1 - np.abs(OA_t.var(0) - OA_p.var(0)) / OA_t.var(0)

    return np.clip(consistency, 0, 1), np.clip(variance, 0, 1)



if __name__ == '__main__':
    H = hparams.HparamsAbsorbing('Lakh')
    H.n_samples = N_SAMPLES

    sampler = get_sampler(H).cuda()

    sampler = load_model(sampler, H.sampler, 420000, H.load_dir).cuda()

    ema = EMA(H.ema_beta)
    ema_sampler = copy.deepcopy(sampler)

    if H.ema:
        try:
            ema_sampler = load_model(
                ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
        except Exception:
            ema_sampler = copy.deepcopy(sampler)


    x_T, mask, original = prep_piece(64, 'data/long_eval.mid')

    samples = get_samples(H, ema_sampler if H.ema else sampler, x_T.cuda(), mask.cuda())
    converter = TrioConverter(16)  # todo: Hparams, async
    samples = converter.from_tensors(samples)
    original = converter.from_tensors(np.expand_dims(original, 0))[0]

    for i in range(len(samples)):
        c, v = evaluate_consistency_variance(original, samples[i])
        print(f'Consistency: {c}, Variance: {v}')
