import fluidsynth
import numpy as np
from preprocessing.read_midi import PPART_FIELDS, TIME_RESOLUTION

SAMPLE_RATE = 44100


def synth_notearray(note_array):
    s = []

    fl = fluidsynth.Synth(samplerate=float(SAMPLE_RATE))

    sfid = fl.sfload("soundfont.sf2")
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


def merge_track_notes(track):
    result = []
    cp = 0
    cd = 0
    co = 0
    for i, p in enumerate(track):
        if cp == p:
            cd += 1
        else:
            if cp:
                result.append((co, cp, cd))
            co = i
            cp = p
            cd = 1
    result.append((co, cp, cd))
    return result


def quantized_sequence_to_note_array(quantized_tracks):
    #todo: assert all tracks same len

    quantized_tracks = [merge_track_notes(qt) for qt in quantized_tracks]
    n_notes = sum([len(qt) for qt in quantized_tracks])
    note_array = np.zeros(n_notes, dtype=PPART_FIELDS)

    n = 0
    for i, track in enumerate(quantized_tracks):
        for o, p, d in track:
            note_array[n]['onset_sec'] = o / TIME_RESOLUTION
            note_array[n]['duration_sec'] = d / TIME_RESOLUTION
            note_array[n]['pitch'] = p
            note_array[n]['velocity'] = 70
            note_array[n]['track'] = i
            note_array[n]['channel'] = i
            note_array[n]['id'] = f'n-{n}'

            n += 1
    idxs = note_array['onset_sec'].argsort()  # todo: slightly inefficient, could be merged linearly
    note_array = note_array[idxs]

    return note_array


def synth_tracks(samples):
    note_arrays = [quantized_sequence_to_note_array([s]) for s in samples]
    audios = [synth_notearray(na) for na in note_arrays]
    return audios