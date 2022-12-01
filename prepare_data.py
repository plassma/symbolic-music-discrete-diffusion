import argparse
import itertools
import os
import warnings
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from note_seq import midi_to_note_sequence
from tqdm import tqdm

from preprocessing.data import TrioConverter, OneHotMelodyConverter


def _load_midi_trio(bars, max_t_per_ns, midi):
    result = []
    converter = TrioConverter(slice_bars=bars, max_tensors_per_notesequence=max_t_per_ns)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ns = midi_to_note_sequence(open(midi, 'rb').read())
            result = list(converter.to_tensors(ns).outputs)  # tensor-shape: (len, 3) = len x (melody, bass, drums)
    except Exception as e:
        pass
        #logging.info(e) todo: make this not destroy tqdm
    return result


def _load_midi_melody(bars, max_t_per_ns, midi):
    result = []
    converter = OneHotMelodyConverter(slice_bars=bars, max_tensors_per_notesequence=max_t_per_ns)
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
    p = Pool(40)
    if limit:
        result = list(tqdm(p.imap(partial(_load_midi_trio, bars, max_tensors_per_ns), itertools.islice(sorted(root_dir.rglob("*.mid")), limit)), total=limit))
    else:
        midis = sorted(root_dir.rglob("*.mid"))
        result = list(tqdm(p.imap(partial(_load_midi_trio, bars, max_tensors_per_ns), midis), total=len(midis), miniters=1))

    ##begin dbg
    #result = []
    #for midi in tqdm(sorted(root_dir.rglob("*.mid"))):
    #    result.append(_load_midi(midi))
    #end dbg
    result = list(chain(*result))
    np.save(cache_path, result)

    return result


def load_lakh_melody(path="lmd_full/", bars=16, max_tensors_per_ns=5, cache_path='data/lakh_melody_BIG_64.npy', limit=0):
    if os.path.exists(cache_path):
        return np.load(cache_path)

    root_dir = Path(path)
    p = Pool(40)
    if limit:
        result = list(tqdm(p.imap(partial(_load_midi_melody, bars, max_tensors_per_ns), itertools.islice(sorted(root_dir.rglob("*.mid")), limit)), total=limit))
    else:
        midis = sorted(root_dir.rglob("*.mid"))
        result = list(tqdm(p.imap(partial(_load_midi_melody, bars, max_tensors_per_ns), midis), total=len(midis), miniters=1))

    np.save(cache_path, result)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="lmd_full/")
    parser.add_argument("--mode", type=str, default="melody")
    parser.add_argument("--target", type=str, default="data/lakh.npy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bars", type=int, default=64)

    args = parser.parse_args()

    if args.mode == 'melody':
        load_lakh_melody(path=args.root_dir, bars=args.bars, cache_path=args.target, limit=args.limit)
    else:
        load_lakh_trio(path=args.root_dir, bars=args.bars, cache_path=args.target, limit=args.limit)
