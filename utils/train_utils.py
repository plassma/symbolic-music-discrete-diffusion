import itertools
from statistics import NormalDist

import numpy as np
from note_seq import quantize_note_sequence
from note_seq import sequences_lib
from tqdm import tqdm

from preprocessing import OneHotMelodyConverter
from .sampler_utils import get_samples


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

def frame_statistics(bars):
    bars = list(itertools.chain(*bars))
    stats = lambda x: NormalDist(np.mean(x), np.std(x) + 1e-6)
    return stats([n.pitch for n in bars]), stats([n.quantized_end_step - n.quantized_start_step for n in bars])


def framewise_overlap_areas(ns, width=4, hop=2):
    if not len(ns.notes):
        return [np.nan, np.nan]

    qns = quantize_note_sequence(ns, 4)
    steps_per_bar = sequences_lib.steps_per_bar_in_quantized_sequence(qns)
    assert steps_per_bar == 16.
    steps_per_bar = 16

    by_bar = [[] for _ in range(max([n.quantized_end_step for n in qns.notes]) // steps_per_bar + 1)]

    for note in qns.notes:
        k = note.quantized_start_step // steps_per_bar
        by_bar[k].append(note)
        #if note.quantized_end_step // steps_per_bar != k:#todo: how 2 handle bar crossing notes?
        #    by_bar[note.quantized_end_step // steps_per_bar].append(note)

    frames = []

    for f in range((len(by_bar) - width) // hop):
        start_bar = hop * f
        frames.append(frame_statistics(by_bar[start_bar:start_bar+width]))

    OAs = []
    for i in range(len(frames) - 1):
        OAs.append([frames[i][j].overlap(frames[i+1][j]) for j in [0, 1]])

    return np.array(OAs).mean(0)


def evaluate_consistency_variance(targets, preds):
    OA_t = [framewise_overlap_areas(t) for t in targets]
    OA_p = [framewise_overlap_areas(p) for p in preds]
    OA_t, OA_p = [oa for oa in OA_t if not np.isnan(oa).any()], [oa for oa in OA_p if not np.isnan(oa).any()]
    OA_t, OA_p = np.stack(OA_t), np.stack(OA_p)


    consistency = 1 - np.abs(OA_t.mean(0) - OA_p.mean(0)) / OA_t.mean(0)
    variance = 1 - np.abs(OA_t.var(0) - OA_p.var(0)) / OA_t.var(0)

    return np.clip(consistency, 0, 1), np.clip(variance, 0, 1)
