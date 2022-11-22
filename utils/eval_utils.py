import itertools
from statistics import NormalDist

import numpy as np
from note_seq import quantize_note_sequence
from note_seq import sequences_lib
from tqdm import tqdm

from .data_utils import SubseqSampler
from .log_utils import log
from .sampler_utils import get_samples, np_to_ns


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


def get_rand_dataset_subset(midi_data, size=1000):
    idx = np.random.choice(midi_data.dataset.shape[0], size)
    return midi_data[idx]


def get_samples_for_eval(H, sampler, mode, dataset, size=1000):
    b = H.sampling_batch_size
    originals = get_rand_dataset_subset(dataset, size)
    if mode == 'unconditional':
        samples = []
        for _ in tqdm(range(int(np.ceil(size / H.sampling_batch_size)))):
            sampler.sampling_batch_size = min(b, size - b * len(samples))
            sa = get_samples(sampler, sample_steps=H.sample_steps, temp=H.temp)
            samples.append(sa)
        samples = np.concatenate(samples)
    elif mode == 'infilling':
        samples = originals.copy()
        samples[:, H.gap_start:H.gap_end] = np.array(H.codebook_size)
        for i in tqdm(range(int(np.ceil(size / H.sampling_batch_size)))):
            l = i * b
            u = min(len(samples), l + b)
            samples[l:u] = \
                get_samples(sampler, sample_steps=H.sample_steps, x_T=samples[l:u], temp=H.temp)
    else:  # self
        samples = get_rand_dataset_subset(dataset, size)

    return np_to_ns(samples[:size]), np_to_ns(originals)


def evaluate(H, sampler):
    if H.mode == 'infilling':
        if H.evals_per_batch != 1:
            log("Infilling only allows one evaluation per batch")
        H.evals_per_batch = 1

    midi_data = np.load(H.dataset_path, allow_pickle=True)
    midi_data = SubseqSampler(midi_data, H.NOTES)

    result = []
    for i in range(H.num_evals):
        samples, originals = get_samples_for_eval(H, sampler, H.mode, midi_data, H.eval_batch_size)

        CVs = []
        for e in range(H.evals_per_batch):
            c, v = evaluate_consistency_variance(originals, samples)
            CVs.append((c, v))
            originals = get_rand_dataset_subset(midi_data, H.eval_batch_size)
            originals = np_to_ns(originals)
        CVs = np.array(CVs)
        result.append(CVs.mean(0))
    return np.array(result).mean(0)
