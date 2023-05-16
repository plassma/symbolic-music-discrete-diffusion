import csv
import glob
import os

import numpy as np
from note_seq import midi_to_note_sequence
import torch
from torch.utils.data import Dataset
from utils.sampler_utils import ns_to_np

RNG = np.random.RandomState(42)

MAJOR_KEYS = [
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
    "E#"
]

MINOR_KEYS_u = [k+"m" for k in MAJOR_KEYS]
MINOR_KEYS = MINOR_KEYS_u[3:] + MINOR_KEYS_u[:3]

ALL_KEYS = MAJOR_KEYS + MINOR_KEYS

def clean_dict(labels_dict):
    if labels_dict["key"] == "F":
        labels_dict["key"] = "E#"
    if labels_dict["key"] == "Fm":
        labels_dict["key"] = "E#m"
    if labels_dict["key"] == "Bb":
        labels_dict["key"] = "A#"
    if labels_dict["key"] == "Bbm":
        labels_dict["key"] = "A#m"
    if labels_dict["key"] == "Gb":
        labels_dict["key"] = "F#"
    if labels_dict["key"] == "Gbm":
        labels_dict["key"] = "F#m"
    if labels_dict["key"] == "Eb":
        labels_dict["key"] = "D#"
    if labels_dict["key"] == "Ebm":
        labels_dict["key"] = "D#m"
    if labels_dict["key"] == "Ab":
        labels_dict["key"] = "G#"
    if labels_dict["key"] == "Abm":
        labels_dict["key"] = "G#m"
def get_transposes(seq, labels_dict, add_octaves=0):
    if "m" in labels_dict["key"]:
        KEYS = MINOR_KEYS
    else:
        KEYS = MAJOR_KEYS

    clean_dict(labels_dict)

    index = KEYS.index(labels_dict["key"])
    result = []
    labels = []

    for i, l in enumerate(KEYS):
        if l == labels_dict:
            result.append(seq)
        else:
            diff = (i - index)
            new_seq = seq.copy()
            new_seq[new_seq > 1] += diff
            result.append(new_seq)
        labels.append(l)

    return result, labels

MAGENTA_PITCH_OFFSET = 19
def load_data(min_seq_length=10, add_octaves=0, par_dir=".", l_quant=1024):
    labels_csv = csv.reader(open(f"{par_dir}/key_meter_estimation/key_meter_ground_truth.txt"), delimiter=',')
    next(labels_csv)
    labels_dict = {line[0]: {"key": line[1], "num": line[2], "denom": line[3], "tempo": line[4]} for line in labels_csv}
    files = glob.glob(os.path.join(f"{par_dir}/key_meter_estimation", "*.mid"))
    files.sort()
    sequences = []
    labels = []
    for fn in files:
        seq = midi_to_note_sequence(open(fn, 'rb').read())
        out = ns_to_np(seq, bars=None)
        out = out.outputs[0].squeeze()

        npy = np.full(l_quant, MAGENTA_PITCH_OFFSET)
        l = min(l_quant, out.shape[0])
        npy[:l] = out[:l]

        if len(seq.notes) > min_seq_length:
            seqs, labls = get_transposes(npy, labels_dict[fn.split('/')[-1]], add_octaves)
            sequences.extend(seqs)
            labels.extend(labls)

    return sequences, labels



MAGENTA_CLASSES = 90

def to_features(x):
    pitches = [torch.nn.functional.one_hot(torch.tensor(s).to(torch.int64), MAGENTA_CLASSES) for s in x]
    return pitches


class PieceDataset(Dataset):
    """
    Dataset for sequential predictions.
    In this case, if data is a sequence of datapoints,
    the inputs (x) will be x[t:t+seq_len] and outputs would
    be (y) x[t+1:t+seq_len+1] (i.e., the next events)
    """

    def __init__(self, data, seq_len=10):
        self.data = data
        self.seq_len = seq_len

    @property
    def piecewise(self):
        return self.seq_len == -1

    def __getitem__(self, i):
        if self.piecewise:
            return self._get_item_piecewise(i)
        else:
            return self._get_item_sequencewise(i)

    def _get_item_piecewise(self, i):
        if i > 0:
            raise IndexError
        x = self.data[:-1]
        y = self.data[1:]
        return x, y

    def _get_item_sequencewise(self, i):
        if i + self.seq_len - 1 > len(self.data):
            raise IndexError
        x = self.data[i:i + self.seq_len]
        y = self.data[i + 1: i + self.seq_len + 1]
        return x, y

    def __len__(self):
        if self.piecewise:
            return 1
        else:
            return max(0, len(self.data) - self.seq_len)

