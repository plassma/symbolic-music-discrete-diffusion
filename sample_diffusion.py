import copy

import torch

import hparams
from utils import load_model
from utils.sampler_utils import get_sampler, get_samples
from utils.train_utils import EMA
from preprocessing import TrioConverter, OneHotMelodyConverter
from note_seq import fluidsynth, note_sequence_to_midi_file, midi_to_note_sequence

N_SAMPLES = 4
def synth_samples(samples):
    converter = TrioConverter(16)#todo: Hparams, async
    samples = converter.from_tensors(samples)
    audios = [fluidsynth(s, 44100., 'soundfont.sf2') for s in samples]
    for i in range(len(samples)):
        note_sequence_to_midi_file(samples[i], f'data/out_{i}.mid')
    return audios


def prep_theme(path="mario_theme.mid"):
    converter = OneHotMelodyConverter(slice_bars=32)
    ns = midi_to_note_sequence(open(path, 'rb').read())
    tensors = list(converter.to_tensors(ns).outputs)
    tensors = [t.squeeze() for t in tensors]
    back_2_ns = converter.from_tensors(tensors)
    note_sequence_to_midi_file(back_2_ns[0], 'out.mid')

    melody = tensors[0]
    x_T = torch.ones((N_SAMPLES, melody.shape[0], 3), dtype=torch.long) * torch.tensor((90, 90, 512))
    x_T[:, :, 0] = torch.tensor(melody)

    mask = torch.zeros_like(x_T)
    mask[:, :, 0] = 1

    return x_T, mask.bool()

if __name__ == '__main__':

    theme, mask = prep_theme()

    H = hparams.HparamsAbsorbing('Lakh')
    H.n_samples = N_SAMPLES

    sampler = get_sampler(H).cuda()

    sampler = load_model(sampler, H.sampler, 270000, H.load_dir).cuda()

    ema = EMA(H.ema_beta)
    ema_sampler = copy.deepcopy(sampler)

    if H.ema:
        try:
            ema_sampler = load_model(
                ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
        except Exception:
            ema_sampler = copy.deepcopy(sampler)

    samples = get_samples(H, ema_sampler if H.ema else sampler, theme.cuda(), mask.cuda())
    synth_samples(samples)
