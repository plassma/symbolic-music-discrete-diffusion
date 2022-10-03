import argparse
import io
import os
import sys
import warnings
from pathlib import Path

import note_seq
import numpy as np
import partitura
import pretty_midi
from note_seq import music_pb2
from partitura.utils import partition
from tqdm import tqdm
from multiprocessing import Pool
import itertools

N_PROC = 40 if os.getlogin() == 'matthiaspl' else 12
DBG = False
MIDI_LOADER = "magenta"

if DBG:
    N_PROC = 1


class MIDIConversionError(Exception):
  pass


PPART_FIELDS = [
    ("onset_sec", "f4"),
    ("duration_sec", "f4"),
    ("pitch", "i4"),
    ("velocity", "i4"),
    ("track", "i4"),
    ("channel", "i4"),
    ("id", "U256"),
]


def extract_melodies_magenta(note_sequence, keep_longest_split=False):
    """Extracts all melodies in a polyphonic note sequence.

    Args:
      note_sequence: A polyphonic NoteSequence object.
      keep_longest_split: Whether to discard all subsequences with tempo changes
          other than the longest one.

    Returns:
      List of monophonic NoteSequence objects.
    """
    splits = note_seq.sequences_lib.split_note_sequence_on_time_changes(
        note_sequence)

    if keep_longest_split:
        ns = max(splits, key=lambda x: len(x.notes))
        splits = [ns]

    melodies = []
    for split_ns in splits:
        qs = note_seq.sequences_lib.quantize_note_sequence(split_ns,
                                                           steps_per_quarter=4)

        instruments = list(set([note.instrument for note in qs.notes]))

        for instrument in instruments:
            melody = note_seq.melodies_lib.Melody()
            try:
                melody.from_quantized_sequence(qs,
                                               ignore_polyphonic_notes=True,
                                               instrument=instrument,
                                               gap_bars=np.inf)
            except note_seq.NonIntegerStepsPerBarError:
                continue
            melody_ns = melody.to_sequence()
            melodies.append(melody_ns)

    return melodies


def midi_to_note_sequence(midi_data):
  """Convert MIDI file contents to a NoteSequence.

  Converts a MIDI file encoded as a string into a NoteSequence. Decoding errors
  are very common when working with large sets of MIDI files, so be sure to
  handle MIDIConversionError exceptions.

  Args:
    midi_data: A string containing the contents of a MIDI file or populated
        pretty_midi.PrettyMIDI object.

  Returns:
    A NoteSequence.

  Raises:
    MIDIConversionError: An improper MIDI mode was supplied.
  """
  # In practice many MIDI files cannot be decoded with pretty_midi. Catch all
  # errors here and try to log a meaningful message. So many different
  # exceptions are raised in pretty_midi.PrettyMidi that it is cumbersome to
  # catch them all only for the purpose of error logging.
  # pylint: disable=bare-except
  if isinstance(midi_data, pretty_midi.PrettyMIDI):
    midi = midi_data
  else:
    try:
      midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_data))
    except:
      raise MIDIConversionError('Midi decoding error %s: %s' %
                                (sys.exc_info()[0], sys.exc_info()[1]))
  # pylint: enable=bare-except

  sequence = music_pb2.NoteSequence()

  # Populate header.
  sequence.ticks_per_quarter = midi.resolution
  sequence.source_info.parser = music_pb2.NoteSequence.SourceInfo.PRETTY_MIDI
  sequence.source_info.encoding_type = (
      music_pb2.NoteSequence.SourceInfo.MIDI)

  # Populate time signatures.
  for midi_time in midi.time_signature_changes:
    time_signature = sequence.time_signatures.add()
    time_signature.time = midi_time.time
    time_signature.numerator = midi_time.numerator
    try:
      # Denominator can be too large for int32.
      time_signature.denominator = midi_time.denominator
    except ValueError:
      raise MIDIConversionError('Invalid time signature denominator %d' %
                                midi_time.denominator)

  # Populate key signatures.
  for midi_key in midi.key_signature_changes:
    key_signature = sequence.key_signatures.add()
    key_signature.time = midi_key.time
    key_signature.key = midi_key.key_number % 12
    midi_mode = midi_key.key_number // 12
    if midi_mode == 0:
      key_signature.mode = key_signature.MAJOR
    elif midi_mode == 1:
      key_signature.mode = key_signature.MINOR
    else:
      raise MIDIConversionError('Invalid midi_mode %i' % midi_mode)

  # Populate tempo changes.
  tempo_times, tempo_qpms = midi.get_tempo_changes()
  for time_in_seconds, tempo_in_qpm in zip(tempo_times, tempo_qpms):
    tempo = sequence.tempos.add()
    tempo.time = time_in_seconds
    tempo.qpm = tempo_in_qpm

  # Populate notes by gathering them all from the midi's instruments.
  # Also set the sequence.total_time as the max end time in the notes.
  midi_notes = []
  midi_pitch_bends = []
  midi_control_changes = []
  for num_instrument, midi_instrument in enumerate(midi.instruments):
    # Populate instrument name from the midi's instruments
    if midi_instrument.name:
      instrument_info = sequence.instrument_infos.add()
      instrument_info.name = midi_instrument.name
      instrument_info.instrument = num_instrument
    for midi_note in midi_instrument.notes:
      if not sequence.total_time or midi_note.end > sequence.total_time:
        sequence.total_time = midi_note.end
      midi_notes.append((midi_instrument.program, num_instrument,
                         midi_instrument.is_drum, midi_note))
    for midi_pitch_bend in midi_instrument.pitch_bends:
      midi_pitch_bends.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_pitch_bend))
    for midi_control_change in midi_instrument.control_changes:
      midi_control_changes.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_control_change))

  for program, instrument, is_drum, midi_note in midi_notes:
    note = sequence.notes.add()
    note.instrument = instrument
    note.program = program
    note.start_time = midi_note.start
    note.end_time = midi_note.end
    note.pitch = midi_note.pitch
    note.velocity = midi_note.velocity
    note.is_drum = is_drum

  for program, instrument, is_drum, midi_pitch_bend in midi_pitch_bends:
    pitch_bend = sequence.pitch_bends.add()
    pitch_bend.instrument = instrument
    pitch_bend.program = program
    pitch_bend.time = midi_pitch_bend.time
    pitch_bend.bend = midi_pitch_bend.pitch
    pitch_bend.is_drum = is_drum

  for program, instrument, is_drum, midi_control_change in midi_control_changes:
    control_change = sequence.control_changes.add()
    control_change.instrument = instrument
    control_change.program = program
    control_change.time = midi_control_change.time
    control_change.control_number = midi_control_change.number
    control_change.control_value = midi_control_change.value
    control_change.is_drum = is_drum

  # TODO(douglaseck): Estimate note type (e.g. quarter note) and populate
  # note.numerator and note.denominator.

  return sequence, midi


TIME_RESOLUTION = 16


def encode_notearray_partitura(notes):
    # todo: notes overlap, a single track can be polyphonic too, up 128 notes simultaneously...
    # channel can have multiple tracks
    #n_channels = len(set(notes['channel']))
    #assert n_channels == max(notes['channel']) + 1

    notes['onset_sec'] = np.rint(notes['onset_sec'] * TIME_RESOLUTION)
    notes['duration_sec'] = np.rint(notes['duration_sec'] * TIME_RESOLUTION)

    by_channel = partition(lambda x: x['channel'], notes)

    length = int(notes[-1]['onset_sec'] + notes[-1]['duration_sec'])
    result = []

    for channel in by_channel.values():
        quantized_pitches = np.zeros(length)

        current_note = 0
        for i in range(length):
            pitch = 0

            while current_note < len(channel) and \
                    (channel[current_note]['onset_sec'] + channel[current_note]['duration_sec'] <= i or
                     (current_note < len(channel) - 1 and channel[current_note + 1]['onset_sec'] <= i)):
                current_note += 1

            if current_note == len(channel):
                break

            if channel[current_note]['onset_sec'] <= i <\
                    channel[current_note]['onset_sec'] + channel[current_note]['duration_sec']:
                pitch = channel[current_note]['pitch']

            quantized_pitches[i] = pitch
        result.append(quantized_pitches)
    return result

def encode_notearray_magenta(tracks):
    result = []

    for track in tracks:
        if not len(track.notes):
            continue
        for i in range(len(track.notes)):
            if i < len(track.notes) - 1:
                assert track.notes[i].end_time <= track.notes[i + 1].start_time
            track.notes[i].start_time *= TIME_RESOLUTION
            track.notes[i].end_time *= TIME_RESOLUTION

        length = int(track.notes[-1].end_time)
        quantized_pitches = np.zeros(length)

        current_note = 0
        for i in range(length):
            pitch = 0

            while current_note < len(track.notes) and \
                    (track.notes[current_note].end_time <= i or
                     (current_note < len(track.notes) - 1 and track.notes[current_note + 1].start_time <= i)):
                current_note += 1

            if current_note == len(track.notes):
                break

            if track.notes[current_note].start_time <= i <\
                    track.notes[current_note].end_time:
                pitch = track.notes[current_note].pitch

            quantized_pitches[i] = pitch
        result.append(quantized_pitches)
    return result


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

def quantized_sequence_to_performance_midi(quantized_tracks):
    #todo: assert all tracks same len

    n_notes = sum([sum([1 for p in qt if p]) for qt in quantized_tracks])
    note_array = np.zeros(n_notes, dtype=PPART_FIELDS)

    n = 0
    for i, track in enumerate(quantized_tracks):
        for p in track:
            if p:
                note_array[n]['onset_sec'] = (n // len(quantized_tracks)) / TIME_RESOLUTION
                note_array[n]['duration_sec'] = 1 / TIME_RESOLUTION
                note_array[n]['pitch'] = p
                note_array[n]['velocity'] = 70
                note_array[n]['track'] = i
                note_array[n]['channel'] = i
                note_array[n]['id'] = f'n-{n}'

                n += 1
    idxs = note_array['onset_sec'].argsort()  # todo: slightly inefficient, could be merged linearly
    note_array = note_array[idxs]

    return note_array


def magenta(midi):
    ns, pm = midi_to_note_sequence(open(midi, 'rb').read())
    melodies = extract_melodies_magenta(ns)
    return melodies


def _load_midi(midi):
    encoded = []
    try:
        if MIDI_LOADER != 'magenta':
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                performance_midi = partitura.load_performance_midi(midi)
                encoded = encode_notearray_partitura(performance_midi.note_array())
        else:
            melodies = magenta(midi)
            encoded = encode_notearray_magenta(melodies)
    except Exception as e:
        print(f"could not process {midi}, {e}")
    return encoded


def load_midis(root_dir, limit=0):
    with Pool(N_PROC) as p:
        if DBG:
            result = [_load_midi(m) for m in itertools.islice(sorted(root_dir.rglob("*.mid")), limit)]
        else:
            if limit:
                result = list(tqdm(p.imap(_load_midi, itertools.islice(sorted(root_dir.rglob("*.mid")), limit)), total=limit))
            else:
                midis = sorted(root_dir.rglob("*.mid"))
                result = list(tqdm(p.imap(_load_midi, midis), total=len(midis)))
    empty_lists = sum([1 for l in result if len(l) == 0])
    print(f'Did not get melodies from {empty_lists} files')
    result = [t for s in result for t in s]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", nargs='?', type=str, default="/media/plassma/Data/Lakh/lmd_full/")

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    quantized_note_arrays = load_midis(root_dir)
    np.save('../data/quantized_note_arrays.npy', quantized_note_arrays, allow_pickle=True)
    quantized_note_arrays = np.load('../data/quantized_note_arrays.npy', allow_pickle=True)
