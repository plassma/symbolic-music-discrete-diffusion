import copy
import itertools
from statistics import NormalDist

import itertools
from statistics import NormalDist

import numpy as np
from note_seq import quantize_note_sequence
from note_seq import sequences_lib
from tqdm import tqdm

import hparams
from utils import SubseqSampler, np_to_ns, log, get_sampler, load_model, EMA
from utils.sampler_utils import get_samples

N_SAMPLES = 256

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


def get_samples_for_eval(mode, dataset, size=1000):
    sampler.n_samples = min(N_SAMPLES, size)
    originals = get_rand_dataset_subset(dataset, size)
    if mode == 'unconditional':
        samples = []
        for _ in tqdm(range(int(np.ceil(size / H.n_samples)))):
            sa = get_samples(ema_sampler if H.ema else sampler, 1024)
            samples.append(sa)
        samples = np.stack(samples)
    elif mode == 'infilling':
        samples = originals.copy()
        samples[:, H.gap_start:H.gap_end] = np.array(H.codebook_size)
        for i in tqdm(range(int(np.ceil(size / H.n_samples)))):
            samples[i * N_SAMPLES:(i + 1) * N_SAMPLES] = \
                get_samples(ema_sampler if H.ema else sampler, 1024, x_T=samples[i * N_SAMPLES:(i + 1) * N_SAMPLES])
    else:  # self
        samples = get_rand_dataset_subset(dataset, size)

    return np_to_ns(samples[:size]), np_to_ns(originals)


def evaluate(eval_dataset_path, mode='unconditional', batches=100, evaluations_per_batch=10, batch_size=1000, notes=1024):

    if mode == 'infilling':
        if evaluations_per_batch != 1:
            log("Infilling only allows one evaluation per batch")
        evaluations_per_batch = 1

    midi_data = np.load(eval_dataset_path, allow_pickle=True)
    midi_data = SubseqSampler(midi_data, notes)

    for i in range(batches):
        samples, originals = get_samples_for_eval(mode, midi_data, batch_size)

        CVs = []
        for e in range(evaluations_per_batch):
            c, v = evaluate_consistency_variance(originals, samples)
            CVs.append((c, v))
            print(c, v)
            originals = get_rand_dataset_subset(midi_data, batch_size)
            originals = np_to_ns(originals)

        CVs = np.array(CVs)
        print(f"avg for samples {i}:", CVs.mean(0))


if __name__ == '__main__':
    H = hparams.HparamsAbsorbingConv('Lakh', 64)
    H.n_samples = N_SAMPLES

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

    evaluate('data/lakh_melody_64_1MIO.npy', mode='infilling', batches=5, evaluations_per_batch=10, batch_size=1000)
