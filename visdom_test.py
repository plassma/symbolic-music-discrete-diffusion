from types import SimpleNamespace

import numpy as np
import partitura
import fluidsynth
from utils import set_up_visdom
from utils import synth_tracks
import partitura.performance as performance

SAMPLE_RATE = 44100
DURATION = 30
F = 1000

def synth_notearray_sample():
    s = []

    fl = fluidsynth.Synth()

    # Initial silence is 1 second
    s = np.append(s, fl.get_samples(SAMPLE_RATE * 1))

    sfid = fl.sfload("/home/plassma/snap/vlc/GeneralUser GS v1.471.sf2")
    fl.program_select(0, sfid, 0, 0)

    fl.noteon(0, 60, 100)
    fl.noteon(0, 67, 100)
    fl.noteon(0, 76, 100)

    # Chord is held for 2 seconds
    s = np.append(s, fl.get_samples(SAMPLE_RATE * 2))

    fl.noteoff(0, 60)
    fl.noteoff(0, 67)
    fl.noteoff(0, 76)

    # Decay of chord is held for 1 second
    s = np.append(s, fl.get_samples(SAMPLE_RATE * 1))

    fl.delete()

    return s


def synth_notearray(note_array):
    s = []

    fl = fluidsynth.Synth(samplerate=float(SAMPLE_RATE))

    sfid = fl.sfload("/home/plassma/snap/vlc/GeneralUser GS v1.471.sf2")
    fl.program_select(0, sfid, 0, 0)

    t = 0
    for note in note_array:
        if t < note['onset_sec']:
            s = np.append(s, fl.get_samples(int(SAMPLE_RATE / 2 * (note['onset_sec'] - t))))#todo: why only half sample rate?
            t = note['onset_sec']

        fl.noteon(0, note['pitch'], 100)
        s = np.append(s, fl.get_samples(int(SAMPLE_RATE / 2 * note['duration_sec'])))
        t += note['duration_sec']
    return s


def samples_to_midi():
    samples = list(np.load('logs/logs/log/samples/samples_20001.npz.npy'))

    note_arrays = [quantized_sequence_to_note_array([s]) for s in samples]

    pparts = [performance.PerformedPart.from_note_array(n) for n in note_arrays]

    for i, ppart in enumerate(pparts):
        partitura.save_performance_midi(ppart, f'midi_{i}.mid')


if __name__ == '__main__':
    H = SimpleNamespace()
    H.visdom_port = 8097
    H.visdom_server = 'localhost'
    vis = set_up_visdom(H)

    #t = np.arange(0, DURATION, 1 / SAMPLE_RATE)
    #tensor = np.sin(t * F * 2 * np.pi)

    #vis.audio(tensor, win="1 kHz Sin")

    samples = list(np.load('logs/logs/log/samples/samples_155000.npz.npy'))

    audios = synth_tracks(samples)

    for i, audio in enumerate(audios):
        vis.audio(audio, win=f"synthesized sample {i}", opts=dict(title=f'sample {i}'))

    samples = list(np.load('logs/logs/log/samples/samples_0.npz.npy'))
    audios = synth_tracks(samples)
    for i, audio in enumerate(audios):
        vis.audio(audio, win=f"synthesized sample {i}", opts=dict(title=f'sample {i}'), env='untrained')
