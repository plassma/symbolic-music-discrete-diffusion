# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MusicVAE data library."""
import abc
import collections
import copy
import functools
import itertools

import note_seq
import numpy as np
from note_seq import chords_lib
from note_seq import drums_encoder_decoder
from note_seq import sequences_lib

from preprocessing import drum_pipelines
from preprocessing import melody_pipelines

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127

MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar
BASS_PROGRAMS = range(32, 40)
ELECTRIC_BASS_PROGRAM = 33

# 9 classes: kick, snare, closed_hh, open_hh, low_tom, mid_tom, hi_tom, crash,
# ride
REDUCED_DRUM_PITCH_CLASSES = drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES
# 61 classes: full General MIDI set
FULL_DRUM_PITCH_CLASSES = [
    [p] for p in  # pylint:disable=g-complex-comprehension
    [36, 35, 38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85, 42, 44,
     54, 68, 69, 70, 71, 73, 78, 80, 46, 67, 72, 74, 79, 81, 45, 29, 41, 61, 64,
     84, 48, 47, 60, 63, 77, 86, 87, 50, 30, 43, 62, 76, 83, 49, 55, 57, 58, 51,
     52, 53, 59, 82]
]
ROLAND_DRUM_PITCH_CLASSES = [
    # kick drum
    [36],
    # snare drum
    [38, 37, 40],
    # closed hi-hat
    [42, 22, 44],
    # open hi-hat
    [46, 26],
    # low tom
    [43, 58],
    # mid tom
    [47, 45],
    # high tom
    [50, 48],
    # crash cymbal
    [49, 52, 55, 57],
    # ride cymbal
    [51, 53, 59]
]

OUTPUT_VELOCITY = 80

CHORD_SYMBOL = note_seq.NoteSequence.TextAnnotation.CHORD_SYMBOL


def _maybe_pad_seqs(seqs, dtype, depth):
  """Pads sequences to match the longest and returns as a numpy array."""
  if not len(seqs):  # pylint:disable=g-explicit-length-test,len-as-condition
    return np.zeros((0, 0, depth), dtype)
  lengths = [len(s) for s in seqs]
  if len(set(lengths)) == 1:
    return np.array(seqs, dtype)
  else:
    length = max(lengths)
    return (np.array([np.pad(s, [(0, length - len(s)), (0, 0)], mode='constant')
                      for s in seqs], dtype))


def _extract_instrument(note_sequence, instrument):
  extracted_ns = copy.copy(note_sequence)
  del extracted_ns.notes[:]
  extracted_ns.notes.extend(
      n for n in note_sequence.notes if n.instrument == instrument)
  return extracted_ns


def maybe_sample_items(seq, sample_size, randomize):
  """Samples a seq if `sample_size` is provided and less than seq size."""
  if not sample_size or len(seq) <= sample_size:
    return seq
  if randomize:
    indices = set(np.random.choice(len(seq), size=sample_size, replace=False))
    return [seq[i] for i in indices]
  else:
    return seq[:sample_size]


def combine_converter_tensors(converter_tensors, max_num_tensors=None,
                              randomize_sample=True):
  """Combines multiple `ConverterTensors` into one and samples if required."""
  results = []
  for result in converter_tensors:
    results.extend(zip(*result))
  sampled_results = maybe_sample_items(results, max_num_tensors,
                                       randomize_sample)
  if sampled_results:
    return ConverterTensors(*zip(*sampled_results))
  else:
    return ConverterTensors()


def np_onehot(indices, depth, dtype=bool):
  """Converts 1D array of indices to a one-hot 2D array with given depth."""
  #onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
  #onehot_seq[np.arange(len(indices)), indices] = 1.0
  #return onehot_seq
  return np.expand_dims(np.array(indices), 1)



class ConverterTensors(collections.namedtuple(
    'ConverterTensors', ['inputs', 'outputs', 'controls', 'lengths'])):
  """Tuple of tensors output by `to_tensors` method in converters.

  Attributes:
    inputs: Input tensors to feed to the encoder.
    outputs: Output tensors to feed to the decoder.
    controls: (Optional) tensors to use as controls for both encoding and
        decoding.
    lengths: Length of each input/output/control sequence.
  """

  def __new__(cls, inputs=None, outputs=None, controls=None, lengths=None):
    if inputs is None:
      inputs = []
    if outputs is None:
      outputs = []
    if lengths is None:
      lengths = [len(i) for i in inputs]
    if not controls:
      controls = [np.zeros([l, 0]) for l in lengths]
    return super(ConverterTensors, cls).__new__(
        cls, inputs, outputs, controls, lengths)


class BaseNoteSequenceConverter(object):
  """Base class for data converters between items and tensors.

  Inheriting classes must implement the following abstract methods:
    -`to_tensors`
    -`from_tensors`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               input_depth,
               input_dtype,
               output_depth,
               output_dtype,
               control_depth=0,
               control_dtype=bool,
               end_token=None,
               max_tensors_per_notesequence=None,
               length_shape=(),
               presplit_on_time_changes=True):
    """Initializes BaseNoteSequenceConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      control_depth: Depth of final dimension of control tensors, or zero if not
          conditioning on control tensors.
      control_dtype: DType of control tensors.
      end_token: Optional end token.
      max_tensors_per_notesequence: The maximum number of outputs to return for
          each input.
      length_shape: Shape of length returned by `to_tensor`.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
    """
    self._input_depth = input_depth
    self._input_dtype = input_dtype
    self._output_depth = output_depth
    self._output_dtype = output_dtype
    self._control_depth = control_depth
    self._control_dtype = control_dtype
    self._end_token = end_token
    self._max_tensors_per_input = max_tensors_per_notesequence
    self._str_to_item_fn = note_seq.NoteSequence.FromString
    self._mode = None
    self._length_shape = length_shape
    self._presplit_on_time_changes = presplit_on_time_changes

  def set_mode(self, mode):
    if mode not in ['train', 'eval', 'infer']:
      raise ValueError('Invalid mode: %s' % mode)
    self._mode = mode

  @property
  def is_training(self):
    return self._mode == 'train'

  @property
  def is_inferring(self):
    return self._mode == 'infer'

  @property
  def str_to_item_fn(self):
    return self._str_to_item_fn

  @property
  def max_tensors_per_notesequence(self):
    return self._max_tensors_per_input

  @max_tensors_per_notesequence.setter
  def max_tensors_per_notesequence(self, value):
    self._max_tensors_per_input = value

  @property
  def end_token(self):
    """End token, or None."""
    return self._end_token

  @property
  def input_depth(self):
    """Dimension of inputs (to encoder) at each timestep of the sequence."""
    return self._input_depth

  @property
  def input_dtype(self):
    """DType of inputs (to encoder)."""
    return self._input_dtype

  @property
  def output_depth(self):
    """Dimension of outputs (from decoder) at each timestep of the sequence."""
    return self._output_depth

  @property
  def output_dtype(self):
    """DType of outputs (from decoder)."""
    return self._output_dtype

  @property
  def control_depth(self):
    """Dimension of control inputs at each timestep of the sequence."""
    return self._control_depth

  @property
  def control_dtype(self):
    """DType of control inputs."""
    return self._control_dtype

  @property
  def length_shape(self):
    """Shape of length returned by `to_tensor`."""
    return self._length_shape

  @abc.abstractmethod
  def to_tensors(self, item):
    """Python method that converts `item` into list of `ConverterTensors`."""
    pass

  @abc.abstractmethod
  def from_tensors(self, samples, controls=None):
    """Python method that decodes model samples into list of items."""
    pass


class LegacyEventListOneHotConverter(BaseNoteSequenceConverter):
  """Converts NoteSequences using legacy OneHotEncoding framework.

  Quantizes the sequences, extracts event lists in the requested size range,
  uniquifies, and converts to encoding. Uses the OneHotEncoding's
  output encoding for both the input and output.

  Attributes:
    event_list_fn: A function that returns a new EventSequence.
    event_extractor_fn: A function for extracing events into EventSequences. The
      sole input should be the quantized NoteSequence.
    legacy_encoder_decoder: An instantiated OneHotEncoding object to use.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    slice_bars: Optional size of window to slide over raw event lists after
      extraction.
    steps_per_quarter: The number of quantization steps per quarter note.
      Mututally exclusive with `steps_per_second`.
    steps_per_second: The number of quantization steps per second.
      Mututally exclusive with `steps_per_quarter`.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
      before converting.
    chord_encoding: An instantiated OneHotEncoding object to use for encoding
      chords on which to condition, or None if not conditioning on chords.
    condition_on_key: If True, condition on key; key is represented as a
      depth-12 one-hot encoding.
    dedupe_event_lists: If True, only keep unique events in the extracted
      event list.
  """

  def __init__(self, event_list_fn, event_extractor_fn,
               legacy_encoder_decoder, add_end_token=False, slice_bars=None,
               slice_steps=None, steps_per_quarter=None, steps_per_second=None,
               quarters_per_bar=4, pad_to_total_time=False,
               max_tensors_per_notesequence=None,
               presplit_on_time_changes=True, chord_encoding=None,
               condition_on_key=False, dedupe_event_lists=True):
    if (steps_per_quarter, steps_per_second).count(None) != 1:
      raise ValueError(
          'Exactly one of `steps_per_quarter` and `steps_per_second` should be '
          'provided.')
    if (slice_bars, slice_steps).count(None) == 0:
      raise ValueError(
          'At most one of `slice_bars` and `slice_steps` should be provided.')
    self._event_list_fn = event_list_fn
    self._event_extractor_fn = event_extractor_fn
    self._legacy_encoder_decoder = legacy_encoder_decoder
    self._chord_encoding = chord_encoding
    self._condition_on_key = condition_on_key
    self._steps_per_quarter = steps_per_quarter
    if steps_per_quarter:
      self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._steps_per_second = steps_per_second
    if slice_bars:
      self._slice_steps = self._steps_per_bar * slice_bars
    else:
      self._slice_steps = slice_steps
    self._pad_to_total_time = pad_to_total_time
    self._dedupe_event_lists = dedupe_event_lists

    depth = legacy_encoder_decoder.num_classes + add_end_token
    control_depth = (
        chord_encoding.num_classes if chord_encoding is not None else 0
    ) + (
        12 if condition_on_key else 0
    )
    super(LegacyEventListOneHotConverter, self).__init__(
        input_depth=depth,
        input_dtype=bool,
        output_depth=depth,
        output_dtype=bool,
        control_depth=control_depth,
        control_dtype=bool,
        end_token=legacy_encoder_decoder.num_classes if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _dedupe_and_sample(self, all_sliced_lists):
    """Dedupe lists of events, chords, and keys, then optionally sample."""
    # This function expects the following lists (when chords & keys are present;
    # when absent those lists will simply be missing):
    #
    # all_sliced_lists = [
    #   [
    #     [event_00, event_01, ..., event_0a],
    #     [event_10, event_11, ..., event_1b],
    #     [event_20, event_21, ..., event_2c]
    #   ],
    #   [
    #     [chord_00, chord_01, ..., chord_0a],
    #     [chord_10, chord_11, ..., chord_1b],
    #     [chord_20, chord_21, ..., chord_2c]
    #   ],
    #   [
    #     [key_00, key_01, ..., key_0a],
    #     [key_10, key_11, ..., key_1b],
    #     [key_20, key_21, ..., key_2c]
    #   ]
    # ]

    sliced_multievent_lists = [zip(*lists) for lists in zip(*all_sliced_lists)]

    # Now we have:
    #
    # sliced_multievent_lists = [
    #   [(event_00, chord_00, key_00), ..., (event_0a, chord_0a, key_0a)],
    #   [(event_10, chord_10, key_10), ..., (event_1b, chord_1b, key_1b)],
    #   [(event_20, chord_20, key_20), ..., (event_2c, chord_2c, key_2c)]
    # ]

    # TODO(adarob): Consider handling the fact that different event lists can
    # be mapped to identical tensors by the encoder_decoder (e.g., Drums).

    if self._dedupe_event_lists:
      multievent_tuples = list(
          set(tuple(l) for l in sliced_multievent_lists))
    else:
      multievent_tuples = [tuple(l) for l in sliced_multievent_lists]
    multievent_tuples = maybe_sample_items(
        multievent_tuples,
        self.max_tensors_per_notesequence,
        self.is_training)

    # Now multievent_tuples is structured like sliced_multievent_lists
    # above, with duplicates optionally removed and sampled.

    if multievent_tuples:
      # Return lists structured like input all_sliced_lists.
      return list(zip(*[zip(*t) for t in multievent_tuples if t]))
    else:
      return []

  def _chords_and_keys_to_controls(self, chord_tuples, key_tuples):
    """Map chord and/or key tuples to control tensors."""
    control_seqs = []

    # Use zip_longest here because chord_tuples and key_tuples should either be:
    #   a) a list of tuples, with chord_tuples and key_tuples the same shape
    #   b) the empty list
    for ct, kt in itertools.zip_longest(chord_tuples, key_tuples):
      controls = []

      if ct is not None:
        try:
          chord_tokens = [self._chord_encoding.encode_event(e) for e in ct]
          if self.end_token:
            # Repeat the last chord instead of using a special token;
            # otherwise the model may learn to rely on the special token to
            # detect endings.
            if chord_tokens:
              chord_tokens.append(chord_tokens[-1])
            else:
              chord_tokens.append(
                  self._chord_encoding.encode_event(note_seq.NO_CHORD))
        except (note_seq.ChordSymbolError, note_seq.ChordEncodingError):
          return []
        controls.append(np_onehot(
            chord_tokens, self._chord_encoding.num_classes,
            self.control_dtype))

      if kt is not None:
        if self.end_token:
          # Repeat the last key. If the sequence is empty, just pick randomly.
          if kt:
            kt.append(kt[-1])
          else:
            kt.append(np.random.choice(range(12)))
        controls.append(np_onehot(kt, 12, self.control_dtype))

      # Concatenate controls (chord and/or key) depthwise. The resulting control
      # tensor should be one of:
      #   a) a one-hot-encoded chord (if not conditioning on key)
      #   b) a one-hot-encoded key (if not conditioning on chord)
      #   c) both (a) and (b), concatenated depthwise
      control_seqs.append(np.concatenate(controls, axis=1))

    return control_seqs

  def to_tensors(self, item):
    """Converts NoteSequence to unique, one-hot tensor sequences."""
    note_sequence = item
    try:
      if self._steps_per_quarter:
        quantized_sequence = note_seq.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
        if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
            self._steps_per_bar):
          return ConverterTensors()
      else:
        quantized_sequence = note_seq.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
    except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError):
      return ConverterTensors()

    if (self._chord_encoding and not any(
        ta.annotation_type == CHORD_SYMBOL
        for ta in quantized_sequence.text_annotations)) or (
            self._condition_on_key and not quantized_sequence.key_signatures):
      # We are conditioning on chords and/or key but sequence does not have
      # them. Try to infer chords and optionally key.
      try:
        note_seq.infer_chords_for_sequence(
            quantized_sequence, add_key_signatures=self._condition_on_key)
      except note_seq.ChordInferenceError:
        return ConverterTensors()

    event_lists, unused_stats = self._event_extractor_fn(quantized_sequence)
    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_lists = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_lists.append(l[i - self._slice_steps: i])
    else:
      sliced_event_lists = event_lists

    # We are going to dedupe the event lists. However, when conditioning on
    # chords and/or key, we want to include the same event list multiple times
    # if it appears with different chords or keys.
    all_sliced_lists = [sliced_event_lists]

    if self._chord_encoding:
      # Extract chord lists that correspond to event lists, i.e. for each event
      # we find the chord active at that time step.
      try:
        sliced_chord_lists = chords_lib.event_list_chords(
            quantized_sequence, sliced_event_lists)
      except chords_lib.CoincidentChordsError:
        return ConverterTensors()
      all_sliced_lists.append(sliced_chord_lists)

    if self._condition_on_key:
      # Extract key lists that correspond to event lists, i.e. for each event
      # we find the key active at that time step.
      if self._steps_per_second:
        steps_per_second = self._steps_per_second
      else:
        qpm = quantized_sequence.tempos[0].qpm
        steps_per_second = self._steps_per_quarter * qpm / 60.0
      sliced_key_lists = chords_lib.event_list_keys(
          quantized_sequence, sliced_event_lists, steps_per_second)
      all_sliced_lists.append(sliced_key_lists)

    all_unique_tuples = self._dedupe_and_sample(all_sliced_lists)
    if not all_unique_tuples:
      return ConverterTensors()

    unique_event_tuples = all_unique_tuples[0]
    unique_chord_tuples = all_unique_tuples[1] if self._chord_encoding else []
    unique_key_tuples = all_unique_tuples[-1] if self._condition_on_key else []

    if self._chord_encoding or self._condition_on_key:
      # We need to encode control sequences consisting of chords and/or keys.
      control_seqs = self._chords_and_keys_to_controls(
          unique_chord_tuples, unique_key_tuples)
      if not control_seqs:
        return ConverterTensors()
    else:
      control_seqs = []

    seqs = []
    for t in unique_event_tuples:
      seqs.append(np_onehot(
          [self._legacy_encoder_decoder.encode_event(e) for e in t] +
          ([] if self.end_token is None else [self.end_token]),
          self.output_depth, self.output_dtype))

    return ConverterTensors(inputs=seqs, outputs=seqs, controls=control_seqs)

  def from_tensors(self, samples, controls=None):
    """Converts model samples to a list of `NoteSequence`s."""
    output_sequences = []
    for i, sample in enumerate(samples):
      s = sample  # np.argmax(sample, axis=-1)
      if self.end_token is not None and self.end_token in s.tolist():
        end_index = s.tolist().index(self.end_token)
      else:
        end_index = len(s)
      s = s[:end_index]
      event_list = self._event_list_fn()
      for e in s:
        assert e != self.end_token
        event_list.append(self._legacy_encoder_decoder.decode_event(e))
      if self._steps_per_quarter:
        qpm = note_seq.DEFAULT_QUARTERS_PER_MINUTE
        seconds_per_step = 60.0 / (self._steps_per_quarter * qpm)
        sequence = event_list.to_sequence(velocity=OUTPUT_VELOCITY, qpm=qpm)
      else:
        seconds_per_step = 1.0 / self._steps_per_second
        sequence = event_list.to_sequence(velocity=OUTPUT_VELOCITY)
      if self._chord_encoding and controls is not None:
        chords = [self._chord_encoding.decode_event(e)
                  for e in np.argmax(controls[i][:, :-12], axis=-1)[:end_index]]
        chord_times = [step * seconds_per_step for step in event_list.steps]
        chords_lib.add_chords_to_sequence(sequence, chords, chord_times)
      if self._condition_on_key and controls is not None:
        keys = np.argmax(controls[i][:, -12:], axis=-1)[:end_index]
        key_times = [step * seconds_per_step for step in event_list.steps]
        chords_lib.add_keys_to_sequence(sequence, keys, key_times)
      output_sequences.append(sequence)
    return output_sequences


class OneHotMelodyConverter(LegacyEventListOneHotConverter):
  """Converter for legacy MelodyOneHotEncoding.

  Attributes:
    melody_fn: A function that takes no arguments and returns an empty Melody.
    melody_encoding: The MelodyOneHotEncoding object used to encode/decode
        individual melody events.
  """

  def __init__(self, min_pitch=PIANO_MIN_MIDI_PITCH,
               max_pitch=PIANO_MAX_MIDI_PITCH, valid_programs=None,
               skip_polyphony=False, max_bars=None, slice_bars=None,
               gap_bars=1.0, steps_per_quarter=4, quarters_per_bar=4,
               add_end_token=False, pad_to_total_time=False,
               max_tensors_per_notesequence=5, presplit_on_time_changes=True,
               chord_encoding=None, condition_on_key=False,
               dedupe_event_lists=True):
    """Initialize a OneHotMelodyConverter object.

    Args:
      min_pitch: The minimum pitch to model. Those below this value will be
          ignored.
      max_pitch: The maximum pitch to model. Those above this value will be
          ignored.
      valid_programs: Optional set of program numbers to allow.
      skip_polyphony: Whether to skip polyphonic instruments. If False, the
          highest pitch will be taken in polyphonic sections.
      max_bars: Optional maximum number of bars per extracted melody, before
          slicing.
      slice_bars: Optional size of window to slide over raw Melodies after
          extraction.
      gap_bars: If this many bars or more of non-events follow a note event, the
          melody is ended. Disabled when set to 0 or None.
      steps_per_quarter: The number of quantization steps per quarter note.
      quarters_per_bar: The number of quarter notes per bar.
      add_end_token: Whether to add an end token at the end of each sequence.
      pad_to_total_time: Pads each input/output tensor to the total time of the
          NoteSequence.
      max_tensors_per_notesequence: The maximum number of outputs to return
          for each NoteSequence.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
          before converting.
      chord_encoding: An instantiated OneHotEncoding object to use for encoding
          chords on which to condition, or None if not conditioning on chords.
      condition_on_key: If True, condition on key; key is represented as a
          depth-12 one-hot encoding.
      dedupe_event_lists: If True, only keep unique events in the extracted
          event list.
    """
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch
    self._valid_programs = valid_programs
    steps_per_bar = steps_per_quarter * quarters_per_bar
    max_steps_truncate = steps_per_bar * max_bars if max_bars else None

    def melody_fn():
      return note_seq.Melody(
          steps_per_bar=steps_per_bar, steps_per_quarter=steps_per_quarter)

    self._melody_fn = melody_fn
    self._melody_encoding = note_seq.MelodyOneHotEncoding(
        min_pitch, max_pitch + 1)

    melody_extractor_fn = functools.partial(
        melody_pipelines.extract_melodies,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=max_steps_truncate,
        min_unique_pitches=1,
        ignore_polyphonic_notes=not skip_polyphony,
        pad_end=True)
    super(OneHotMelodyConverter, self).__init__(
        melody_fn,
        melody_extractor_fn,
        self._melody_encoding,
        add_end_token=add_end_token,
        slice_bars=slice_bars,
        pad_to_total_time=pad_to_total_time,
        steps_per_quarter=steps_per_quarter,
        quarters_per_bar=quarters_per_bar,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes,
        chord_encoding=chord_encoding,
        condition_on_key=condition_on_key,
        dedupe_event_lists=dedupe_event_lists)

  @property
  def melody_fn(self):
    return self._melody_fn

  @property
  def melody_encoding(self):
    return self._melody_encoding

  def _to_tensors_fn(self, note_sequence):
    def is_valid(note):
      if (self._valid_programs is not None and
          note.program not in self._valid_programs):
        return False
      return self._min_pitch <= note.pitch <= self._max_pitch
    notes = list(note_sequence.notes)
    del note_sequence.notes[:]
    note_sequence.notes.extend([n for n in notes if is_valid(n)])
    return super(OneHotMelodyConverter, self).to_tensors(note_sequence)

  def to_tensors(self, item):
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, self._to_tensors_fn)


class DrumsConverter(BaseNoteSequenceConverter):
  """Converter for legacy drums with either pianoroll or one-hot tensors.

  Inputs/outputs are either a "pianoroll"-like encoding of all possible drum
  hits at a given step, or a one-hot encoding of the pianoroll.

  The "roll" input encoding includes a final NOR bit (after the optional end
  token).

  Attributes:
    max_bars: Optional maximum number of bars per extracted drums, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more follow a non-empty drum event, the
      drum track is ended. Disabled when set to 0 or None.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by. By
      default, groups valid drum pitches into 9 different classes.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    roll_input: Whether to use a pianoroll-like representation as the input
      instead of a one-hot encoding.
    roll_output: Whether to use a pianoroll-like representation as the output
      instead of a one-hot encoding.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
      before converting.
  """

  def __init__(self, max_bars=None, slice_bars=None, gap_bars=1.0,
               pitch_classes=None, add_end_token=False, steps_per_quarter=4,
               quarters_per_bar=4, pad_to_total_time=False, roll_input=False,
               roll_output=False, max_tensors_per_notesequence=5,
               presplit_on_time_changes=True):
    self._pitch_classes = pitch_classes or REDUCED_DRUM_PITCH_CLASSES
    self._pitch_class_map = {}
    for i, pitches in enumerate(self._pitch_classes):
      self._pitch_class_map.update({p: i for p in pitches})
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._slice_steps = self._steps_per_bar * slice_bars if slice_bars else None
    self._pad_to_total_time = pad_to_total_time
    self._roll_input = roll_input
    self._roll_output = roll_output

    self._drums_extractor_fn = functools.partial(
        drum_pipelines.extract_drum_tracks,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=self._steps_per_bar * max_bars if max_bars else None,
        pad_end=True)

    num_classes = len(self._pitch_classes)

    self._pr_encoder_decoder = note_seq.PianorollEncoderDecoder(
        input_size=num_classes + add_end_token)
    # Use pitch classes as `drum_type_pitches` since we have already done the
    # mapping.
    self._oh_encoder_decoder = note_seq.MultiDrumOneHotEncoding(
        drum_type_pitches=[(i,) for i in range(num_classes)])

    if self._roll_output:
      output_depth = num_classes + add_end_token
    else:
      output_depth = self._oh_encoder_decoder.num_classes + add_end_token

    if self._roll_input:
      input_depth = num_classes + 1 + add_end_token
    else:
      input_depth = self._oh_encoder_decoder.num_classes + add_end_token

    super(DrumsConverter, self).__init__(
        input_depth=input_depth,
        input_dtype=bool,
        output_depth=output_depth,
        output_dtype=bool,
        end_token=output_depth - 1 if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors_fn(self, note_sequence):
    """Converts NoteSequence to unique sequences."""
    try:
      quantized_sequence = note_seq.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
    except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError):
      return ConverterTensors()

    new_notes = []
    for n in quantized_sequence.notes:
      if not n.is_drum:
        continue
      if n.pitch not in self._pitch_class_map:
        continue
      n.pitch = self._pitch_class_map[n.pitch]
      new_notes.append(n)
    del quantized_sequence.notes[:]
    quantized_sequence.notes.extend(new_notes)

    event_lists, unused_stats = self._drums_extractor_fn(quantized_sequence)

    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_tuples = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_tuples.append(tuple(l[i - self._slice_steps: i]))
    else:
      sliced_event_tuples = [tuple(l) for l in event_lists]

    unique_event_tuples = list(set(sliced_event_tuples))
    unique_event_tuples = maybe_sample_items(unique_event_tuples,
                                             self.max_tensors_per_notesequence,
                                             self.is_training)

    rolls = []
    oh_vecs = []
    for t in unique_event_tuples:
      if self._roll_input or self._roll_output:
        if self.end_token is not None:
          t_roll = list(t) + [(self._pr_encoder_decoder.input_size - 1,)]
        else:
          t_roll = t
        rolls.append(np.vstack([
            self._pr_encoder_decoder.events_to_input(t_roll, i).astype(bool)
            for i in range(len(t_roll))]))
      if not (self._roll_input and self._roll_output):
        labels = [self._oh_encoder_decoder.encode_event(e) for e in t]
        if self.end_token is not None:
          labels += [self._oh_encoder_decoder.num_classes]
        oh_vecs.append(np_onehot(
            labels,
            self._oh_encoder_decoder.num_classes + (self.end_token is not None),
            bool))

    if self._roll_input:
      input_seqs = [
          np.append(roll, np.expand_dims(np.all(roll == 0, axis=1), axis=1),
                    axis=1) for roll in rolls]
    else:
      input_seqs = oh_vecs

    output_seqs = rolls if self._roll_output else oh_vecs

    return ConverterTensors(inputs=input_seqs, outputs=output_seqs)

  def to_tensors(self, item):
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, self._to_tensors_fn)

  def from_tensors(self, samples, unused_controls=None):
    output_sequences = []
    for s in samples:
      if self._roll_output:
        if self.end_token is not None:
          end_i = np.where(s[:, self.end_token])
          if len(end_i):  # pylint: disable=g-explicit-length-test,len-as-condition
            s = s[:end_i[0]]
        events_list = [frozenset(np.where(e)[0]) for e in s]
      else:
        s = s  # np.argmax(s, axis=-1)
        if self.end_token is not None and self.end_token in s:
          s = s[:s.tolist().index(self.end_token)]
        events_list = [self._oh_encoder_decoder.decode_event(e) for e in s]
      # Map classes to exemplars.
      events_list = [
          frozenset(self._pitch_classes[c][0] for c in e) for e in events_list]
      track = note_seq.DrumTrack(
          events=events_list,
          steps_per_bar=self._steps_per_bar,
          steps_per_quarter=self._steps_per_quarter)
      output_sequences.append(track.to_sequence(velocity=OUTPUT_VELOCITY))
    return output_sequences


class TrioConverter(BaseNoteSequenceConverter):
  """Converts to/from 3-part (mel, drums, bass) multi-one-hot events.

  Extracts overlapping segments with melody, drums, and bass (determined by
  program number) and concatenates one-hot tensors from OneHotMelodyConverter
  and OneHotDrumsConverter. Takes the cross products from the sets of
  instruments of each type.

  Attributes:
    slice_bars: Optional size of window to slide over full converted tensor.
    gap_bars: The number of consecutive empty bars to allow for any given
      instrument. Note that this number is effectively doubled for internal
      gaps.
    max_bars: Optional maximum number of bars per extracted sequence, before
      slicing.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    chord_encoding: An instantiated OneHotEncoding object to use for encoding
      chords on which to condition, or None if not conditioning on chords.
    condition_on_key: If True, condition on key; key is represented as a
      depth-12 one-hot encoding.
  """

  class InstrumentType(object):
    UNK = 0
    MEL = 1
    BASS = 2
    DRUMS = 3
    INVALID = 4

  def __init__(
      self, slice_bars=None, gap_bars=2, max_bars=1024, steps_per_quarter=4,
      quarters_per_bar=4, max_tensors_per_notesequence=5,
      chord_encoding=None, condition_on_key=False):
    self._melody_converter = OneHotMelodyConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None, chord_encoding=chord_encoding,
        condition_on_key=condition_on_key)
    self._drums_converter = DrumsConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None)
    self._slice_bars = slice_bars
    self._gap_bars = gap_bars
    self._max_bars = max_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._chord_encoding = chord_encoding
    self._condition_on_key = condition_on_key

    self._split_output_depths = (
        self._melody_converter.output_depth,
        self._melody_converter.output_depth,
        self._drums_converter.output_depth)
    output_depth = sum(self._split_output_depths)

    self._program_map = dict(
        [(i, TrioConverter.InstrumentType.MEL) for i in MEL_PROGRAMS] +
        [(i, TrioConverter.InstrumentType.BASS) for i in BASS_PROGRAMS])

    super(TrioConverter, self).__init__(
        input_depth=output_depth,
        input_dtype=bool,
        output_depth=output_depth,
        output_dtype=bool,
        control_depth=self._melody_converter.control_depth,
        control_dtype=self._melody_converter.control_dtype,
        end_token=False,
        presplit_on_time_changes=True,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors_fn(self, note_sequence):
    """Converts a `NoteSequence` to `ConverterTensors` obj."""
    try:
      quantized_sequence = note_seq.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
    except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError):
      return ConverterTensors()

    if (self._chord_encoding and not any(
        ta.annotation_type == CHORD_SYMBOL
        for ta in quantized_sequence.text_annotations)) or (
            self._condition_on_key and not quantized_sequence.key_signatures):
      # We are conditioning on chords and/or key but sequence does not have
      # them. Try to infer chords and optionally key.
      try:
        note_seq.infer_chords_for_sequence(
            quantized_sequence, add_key_signatures=self._condition_on_key)
      except note_seq.ChordInferenceError:
        return ConverterTensors()

      # The trio parts get extracted from the original NoteSequence, so copy the
      # inferred chords and keys back to that one.
      for qta in quantized_sequence.text_annotations:
        if qta.annotation_type == CHORD_SYMBOL:
          ta = note_sequence.text_annotations.add()
          ta.annotation_type = CHORD_SYMBOL
          ta.time = qta.time
          ta.text = qta.text
      for qks in quantized_sequence.key_signatures:
        ks = note_sequence.key_signatures.add()
        ks.time = qks.time
        ks.key = qks.key

    total_bars = int(
        np.ceil(quantized_sequence.total_quantized_steps / self._steps_per_bar))
    total_bars = min(total_bars, self._max_bars)

    # Assign an instrument class for each instrument, and compute its coverage.
    # If an instrument has multiple classes, it is considered INVALID.
    instrument_type = np.zeros(MAX_INSTRUMENT_NUMBER + 1, np.uint8)
    coverage = np.zeros((total_bars, MAX_INSTRUMENT_NUMBER + 1), bool)
    for note in quantized_sequence.notes:
      i = note.instrument
      if i > MAX_INSTRUMENT_NUMBER:
        #logging.warning('Skipping invalid instrument number: %d', i)
        continue
      if note.is_drum:
        inferred_type = self.InstrumentType.DRUMS
      else:
        inferred_type = self._program_map.get(
            note.program, self.InstrumentType.INVALID)
      if not instrument_type[i]:
        instrument_type[i] = inferred_type
      elif instrument_type[i] != inferred_type:
        instrument_type[i] = self.InstrumentType.INVALID

      start_bar = note.quantized_start_step // self._steps_per_bar
      end_bar = int(np.ceil(note.quantized_end_step / self._steps_per_bar))

      if start_bar >= total_bars:
        continue
      coverage[start_bar:min(end_bar, total_bars), i] = True

    # Group instruments by type.
    instruments_by_type = collections.defaultdict(list)
    for i, type_ in enumerate(instrument_type):
      if type_ not in (self.InstrumentType.UNK, self.InstrumentType.INVALID):
        instruments_by_type[type_].append(i)
    if len(instruments_by_type) < 3:
      # This NoteSequence doesn't have all 3 types.
      return ConverterTensors()

    # Encode individual instruments.
    # Set total time so that instruments will be padded correctly.
    note_sequence.total_time = (
        total_bars * self._steps_per_bar *
        60 / note_sequence.tempos[0].qpm / self._steps_per_quarter)
    encoded_instruments = {}
    encoded_controls = None
    for i in (instruments_by_type[self.InstrumentType.MEL] +
              instruments_by_type[self.InstrumentType.BASS]):
      tensors = self._melody_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if tensors.outputs:
        encoded_instruments[i] = tensors.outputs[0]
        if encoded_controls is None:
          encoded_controls = tensors.controls[0]
        elif not np.array_equal(encoded_controls, tensors.controls[0]):
          #logging.warning('Trio controls disagreement between instruments.')
          pass
      else:
        coverage[:, i] = False
    for i in instruments_by_type[self.InstrumentType.DRUMS]:
      tensors = self._drums_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if tensors.outputs:
        encoded_instruments[i] = tensors.outputs[0]
      else:
        coverage[:, i] = False

    # Fill in coverage gaps up to self._gap_bars.
    og_coverage = coverage.copy()
    for j in range(total_bars):
      coverage[j] = np.any(
          og_coverage[
              max(0, j-self._gap_bars):min(total_bars, j+self._gap_bars) + 1],
          axis=0)

    # Take cross product of instruments from each class and compute combined
    # encodings where they overlap.
    seqs = []
    control_seqs = []
    for grp in itertools.product(
        instruments_by_type[self.InstrumentType.MEL],
        instruments_by_type[self.InstrumentType.BASS],
        instruments_by_type[self.InstrumentType.DRUMS]):
      # Consider an instrument covered within gap_bars from the end if any of
      # the other instruments are. This allows more leniency when re-encoding
      # slices.
      grp_coverage = np.all(coverage[:, grp], axis=1)
      grp_coverage[:self._gap_bars] = np.any(coverage[:self._gap_bars, grp])
      grp_coverage[-self._gap_bars:] = np.any(coverage[-self._gap_bars:, grp])
      slice_bars = self._slice_bars if self._slice_bars else total_bars
      for j in range(total_bars - slice_bars + 1):
        if (np.all(grp_coverage[j:j + slice_bars]) and
            all(i in encoded_instruments for i in grp)):
          start_step = j * self._steps_per_bar
          end_step = (j + slice_bars) * self._steps_per_bar
          seqs.append(np.concatenate(
              [encoded_instruments[i][start_step:end_step] for i in grp],
              axis=-1))
          if encoded_controls is not None:
            control_seqs.append(encoded_controls[start_step:end_step])

    return ConverterTensors(inputs=seqs, outputs=seqs, controls=control_seqs)

  def to_tensors(self, item):
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, self._to_tensors_fn)

  def from_tensors(self, samples, controls=None):
    output_sequences = []
    dim_ranges = np.cumsum(self._split_output_depths)
    for i, s in enumerate(samples):
      mel_ns = self._melody_converter.from_tensors(
          [s[:, 0]],
          [controls[i]] if controls is not None else None)[0]
      bass_ns = self._melody_converter.from_tensors(
          [s[:, 1]])[0]
      drums_ns = self._drums_converter.from_tensors(
          [s[:, 2]])[0]

      for n in bass_ns.notes:
        n.instrument = 1
        n.program = ELECTRIC_BASS_PROGRAM
      for n in drums_ns.notes:
        n.instrument = 9

      ns = mel_ns
      ns.notes.extend(bass_ns.notes)
      ns.notes.extend(drums_ns.notes)
      ns.total_time = max(
          mel_ns.total_time, bass_ns.total_time, drums_ns.total_time)
      output_sequences.append(ns)
    return output_sequences

def split_process_and_combine(note_sequence, split, sample_size, randomize,
                              to_tensors_fn):
  """Splits a `NoteSequence`, processes and combines the `ConverterTensors`.

  Args:
    note_sequence: The `NoteSequence` to split, process and combine.
    split: If True, the given note_sequence is split into multiple based on time
      changes, and the tensor outputs are concatenated.
    sample_size: Outputs are sampled if size exceeds this value.
    randomize: If True, outputs are randomly sampled (this is generally done
      during training).
    to_tensors_fn: A fn that converts a `NoteSequence` to `ConverterTensors`.

  Returns:
    A `ConverterTensors` obj.
  """
  note_sequences = sequences_lib.split_note_sequence_on_time_changes(
      note_sequence) if split else [note_sequence]
  results = []
  for ns in note_sequences:
    tensors = to_tensors_fn(ns)
    sampled_results = maybe_sample_items(
        list(zip(*tensors)), sample_size, randomize)
    if sampled_results:
      results.append(ConverterTensors(*zip(*sampled_results)))
    else:
      results.append(ConverterTensors())
  return combine_converter_tensors(results, sample_size, randomize)