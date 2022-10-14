import copy
import itertools
from statistics import NormalDist

import numpy as np
import torch
from note_seq import midi_to_note_sequence, quantize_note_sequence, note_sequence_to_midi_file
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
    x_T = torch.tensor(melody).tile((N_SAMPLES, 1, 1))

    x_T[:, 256:768, 0] = 90
    x_T[:, 256:768, 1] = 90
    x_T[:, 256:768, 2] = 512

    mask = torch.ones_like(x_T)
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
        if all([(frames[i][j].variance and frames[i+1][j].variance) for j in [0, 1]]):
            OAs.append([frames[i][j].overlap(frames[i+1][j]) for j in [0, 1]])

    return np.array(OAs).mean(0)


def evaluate_consistency_variance(targets, preds):
    OA_t, OA_p = np.array([framewise_overlap_areas(t) for t in targets]), np.array([framewise_overlap_areas(p) for p in preds])


    consistency = 1 - np.abs(OA_t.mean(0) - OA_p.mean(0)) / OA_t.mean(0)
    variance = 1 - np.abs(OA_t.var(0) - OA_p.var(0)) / OA_t.var(0)

    return np.clip(consistency, 0, 1), np.clip(variance, 0, 1)

def evaluate_unconditional(batches = 100, batch_size=1000):
    midi_data = np.load('data/full_lakh.npy', allow_pickle=True)

    samples = []

    while len(samples) < batch_size:
        sa = get_samples(H, ema_sampler if H.ema else sampler)
        for s in sa:
            samples.append(s)

    idx = np.random.choice(midi_data.shape[0], batch_size)
    originals = midi_data[idx]

    converter = TrioConverter()
    to_ns = lambda x: converter.from_tensors(np.expand_dims(x, 0))[0]

    samples, originals = [to_ns(s) for s in samples], [to_ns(o) for o in originals]

    c, v = evaluate_consistency_variance(originals, samples)

    print(c, v)


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

    evaluate_unconditional(2, 100)

    #x_T, mask, original = prep_piece(64, 'data/long_eval.mid')

    #converter = TrioConverter(16)  # todo: Hparams, async

    #if False:
    #    samples = get_samples(H, ema_sampler if H.ema else sampler, x_T.cuda(), mask.cuda())
    #    samples = converter.from_tensors(samples)
    #else:
    #    samples = [midi_to_note_sequence(open(f'sample_{i}.mid', 'rb').read()) for i in range(4)]
    #original = converter.from_tensors(np.expand_dims(original, 0))[0]
    #if False:
    #    note_sequence_to_midi_file(original, 'sample_original.mid')
    #    [note_sequence_to_midi_file(s, f'sample_{i}.mid') for i, s in enumerate(samples)]
    #for i in range(len(samples)):
    #    c, v = evaluate_consistency_variance(original, samples[i])
    #    print(f'Consistency: {c}, Variance: {v}')
