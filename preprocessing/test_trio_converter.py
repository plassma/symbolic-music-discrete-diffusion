import preprocessing
from preprocessing import TrioConverter
#from data import TrioConverter
from note_seq import midi_to_note_sequence
import note_seq
from note_seq import fluidsynth
from pathlib import Path
import random
import os

BARS = 16
NUMBER_PIECES = 5

if __name__ == '__main__':
    root_dir = Path("/media/plassma/Data/Lakh/lmd_full/")
    converter = TrioConverter(BARS)
    tensors = []
    max_drum_pitch = 0
    for midi in root_dir.rglob("*.mid"):
        try:
            ns = midi_to_note_sequence(open(midi, 'rb').read())
            tensor = converter.to_tensors(ns)
            #tensor-shape: (len, 90 + 90 + 512) = melody, bass, drums
            if len(tensor.lengths):
                tensors.append(tensor.outputs[random.randint(0, len(tensor.outputs))])#only take one rand sample 4 diversity
        except Exception as e:
            print(e)

        if len(tensors) == NUMBER_PIECES:
            break

    seqs = converter.from_tensors(tensors)
    wave = fluidsynth(seqs[0], 44100., 'soundfont.sf2')
    os.makedirs('data/trio_converter', exist_ok=True)
    for i, ns in enumerate(seqs):
        note_seq.sequence_proto_to_midi_file(ns, f'data/trio_converter/sample_{i}.mid')
