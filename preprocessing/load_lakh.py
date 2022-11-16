from data import TrioConverter, OneHotMelodyConverter
import argparse
import itertools
import logging
import os
from functools import partial
from itertools import chain
from pathlib import Path

import numpy as np
from multiprocessing import Pool
from note_seq import midi_to_note_sequence
from tqdm import tqdm
import warnings



def _load_midi_trio(midi):
    result = []
    converter = TrioConverter(slice_bars=64, max_tensors_per_notesequence=100)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ns = midi_to_note_sequence(open(midi, 'rb').read())
            result = list(converter.to_tensors(ns).outputs)  # tensor-shape: (len, 3) = len x (melody, bass, drums)
    except Exception as e:
        pass
        #logging.info(e) todo: make this not destroy tqdm
    return result


def _load_midi_melody(midi):
    result = []
    converter = OneHotMelodyConverter(slice_bars=64, max_tensors_per_notesequence=100)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ns = midi_to_note_sequence(open(midi, 'rb').read())
            result = list(converter.to_tensors(ns).outputs)  # tensor-shape: (len, 3) = len x (melody, bass, drums)
    except Exception as e:
        pass
        #logging.info(e) todo: make this not destroy tqdm
    return result

def load_lakh_trio(path="/media/plassma/Data/Lakh/lmd_full/", bars=16, max_tensors_per_ns=5, cache_path='data/lakh_trio_BIG_64.npy', limit=0):
    if os.path.exists(cache_path):
        return np.load(cache_path)

    root_dir = Path(path)
    converter = TrioConverter(bars, max_tensors_per_notesequence=max_tensors_per_ns)
    p = Pool(40)
    if limit:
        result = list(tqdm(p.imap(_load_midi_trio, itertools.islice(sorted(root_dir.rglob("*.mid")), limit)), total=limit))
    else:
        midis = sorted(root_dir.rglob("*.mid"))
        result = list(tqdm(p.imap(_load_midi_trio, midis), total=len(midis), miniters=1))

    ##begin dbg
    #result = []
    #for midi in tqdm(sorted(root_dir.rglob("*.mid"))):
    #    result.append(_load_midi(midi))
    #end dbg
    result = list(chain(*result))
    np.save(cache_path, result)

    return result


def load_lakh_melody(path="/media/plassma/Data/Lakh/lmd_full/", bars=16, max_tensors_per_ns=5, cache_path='data/lakh_melody_BIG_64.npy', limit=0):
    if os.path.exists(cache_path):
        return np.load(cache_path)

    root_dir = Path(path)
    #converter = OneHotMelodyConverter(bars, max_tensors_per_notesequence=max_tensors_per_ns)
    #p = Pool(40)
    #if limit:
    #    result = list(tqdm(p.imap(_load_midi_melody, itertools.islice(sorted(root_dir.rglob("*.mid")), limit)), total=limit))
    #else:
    #    midis = sorted(root_dir.rglob("*.mid"))
    #   result = list(tqdm(p.imap(_load_midi_melody, midis), total=len(midis), miniters=1))

    ##begin dbg
    result = []
    for midi in tqdm(sorted(root_dir.rglob("*.mid"))):
        result.append(_load_midi_melody(midi))
    #end dbg
    result = list(chain(*result))
    np.save(cache_path, result)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", nargs='?', type=str, default="/media/plassma/Data/Lakh/lmd_full/")

    args = parser.parse_args()
    load_lakh_melody(path=args.root_dir)
