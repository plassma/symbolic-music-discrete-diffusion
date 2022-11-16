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

"""For running data processing pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect
import os.path

import statistics
import six


class InvalidTypeSignatureError(Exception):
  """Thrown when `Pipeline.input_type` or `Pipeline.output_type` is not valid.
  """
  pass


class InvalidStatisticsError(Exception):
  """Thrown when stats produced by a `Pipeline` are not valid."""
  pass


class PipelineKey(object):
  """Represents a get operation on a Pipeline type signature.

  If a pipeline instance `my_pipeline` has `output_type`
  {'key_1': Type1, 'key_2': Type2}, then PipelineKey(my_pipeline, 'key_1'),
  represents the output type Type1. And likewise
  PipelineKey(my_pipeline, 'key_2') represents Type2.

  Calling __getitem__ on a pipeline will return a PipelineKey instance.
  So my_pipeline['key_1'] returns PipelineKey(my_pipeline, 'key_1'), and so on.

  PipelineKey objects are used for assembling a directed acyclic graph of
  Pipeline instances. See dag_pipeline.py.
  """

  def __init__(self, unit, key):
    if not isinstance(unit, Pipeline):
      raise ValueError('Cannot take key of non Pipeline %s' % unit)
    if not isinstance(unit.output_type, dict):
      raise KeyError(
          'Cannot take key %s of %s because output type %s is not a dictionary'
          % (key, unit, unit.output_type))
    if key not in unit.output_type:
      raise KeyError('PipelineKey %s is not valid for %s with output type %s'
                     % (key, unit, unit.output_type))
    self.key = key
    self.unit = unit
    self.output_type = unit.output_type[key]

  def __repr__(self):
    return 'PipelineKey(%s, %s)' % (self.unit, self.key)


def _guarantee_dict(given, default_name):
  if not isinstance(given, dict):
    return {default_name: list}
  return given


def _assert_valid_type_signature(type_sig, type_sig_name):
  """Checks that the given type signature is valid.

  Valid type signatures are either a single Python class, or a dictionary
  mapping string names to Python classes.

  Throws a well formatted exception when invalid.

  Args:
    type_sig: Type signature to validate.
    type_sig_name: Variable name of the type signature. This is used in
        exception descriptions.

  Raises:
    InvalidTypeSignatureError: If `type_sig` is not valid.
  """
  if isinstance(type_sig, dict):
    for k, val in type_sig.items():
      if not isinstance(k, six.string_types):
        raise InvalidTypeSignatureError(
            '%s key %s must be a string.' % (type_sig_name, k))
      if not inspect.isclass(val):
        raise InvalidTypeSignatureError(
            '%s %s at key %s must be a Python class.' % (type_sig_name, val, k))
  else:
    if not inspect.isclass(type_sig):
      raise InvalidTypeSignatureError(
          '%s %s must be a Python class.' % (type_sig_name, type_sig))


class Pipeline(object):
  """An abstract class for data processing pipelines that transform datasets.

  A Pipeline can transform one or many inputs to one or many outputs. When there
  are many inputs or outputs, each input/output is assigned a string name.

  The `transform` method converts a given input or dictionary of inputs to
  a list of transformed outputs, or a dictionary mapping names to lists of
  transformed outputs for each name.

  The `get_stats` method returns any Statistics that were collected during the
  last call to `transform`. These Statistics can give feedback about why any
  data was discarded and what the input data is like.

  `Pipeline` implementers should call `_set_stats` from within `transform` to
  set the Statistics that will be returned by the next call to `get_stats`.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_type, output_type, name=None):
    """Constructs a `Pipeline` object.

    Subclass constructors are expected to call this constructor.

    A type signature is a Python class or primative collection containing
    classes. Valid type signatures for `Pipeline` inputs and outputs are either
    a Python class, or a dictionary mapping string names to classes. An object
    matches a type signature if its type equals the type signature
    (i.e. type('hello') == str) or, if its a collection, the types in the
    collection match (i.e. {'hello': 'world', 'number': 1234} matches type
    signature {'hello': str, 'number': int})

    `Pipeline` instances have (preferably unique) string names. These names act
    as name spaces for the Statistics produced by them. The `get_stats` method
    will automatically prepend `name` to all of the Statistics names before
    returning them.

    Args:
      input_type: The type signature this pipeline expects for its inputs.
      output_type: The type signature this pipeline promises its outputs will
          have.
      name: The string name for this instance. This name is accessible through
          the `name` property. Names should be unique across `Pipeline`
          instances. If None (default), the string name of the implementing
          subclass is used.
    """
    # Make sure `input_type` and `output_type` are valid.
    if name is None:
      # This will get the name of the subclass, not "Pipeline".
      self._name = type(self).__name__
    else:
      assert isinstance(name, six.string_types)
      self._name = name
    _assert_valid_type_signature(input_type, 'input_type')
    _assert_valid_type_signature(output_type, 'output_type')
    self._input_type = input_type
    self._output_type = output_type
    self._stats = []

  def __getitem__(self, key):
    return PipelineKey(self, key)

  @property
  def input_type(self):
    """What type or types does this pipeline take as input.

    Returns:
      A class, or a dictionary mapping names to classes.
    """
    return self._input_type

  @property
  def output_type(self):
    """What type or types does this pipeline output.

    Returns:
      A class, or a dictionary mapping names to classes.
    """
    return self._output_type

  @property
  def output_type_as_dict(self):
    """Returns a dictionary mapping names to classes.

    If `output_type` is a single class, then a default name will be created
    for the output and a dictionary containing `output_type` will be returned.

    Returns:
      Dictionary mapping names to output types.
    """
    return _guarantee_dict(self._output_type, 'dataset')

  @property
  def name(self):
    """The string name of this pipeline."""
    return self._name

  @abc.abstractmethod
  def transform(self, input_object):
    """Runs the pipeline on the given input.

    Args:
      input_object: An object or dictionary mapping names to objects.
          The object types must match `input_type`.

    Returns:
      If `output_type` is a class, `transform` returns a list of objects
      which are all that type. If `output_type` is a dictionary mapping
      names to classes, `transform` returns a dictionary mapping those
      same names to lists of objects that are the type mapped to each name.
    """
    pass

  def _set_stats(self, stats):
    """Overwrites the current Statistics returned by `get_stats`.

    Implementers of Pipeline should call `_set_stats` from within `transform`.

    Args:
      stats: An iterable of Statistic objects.

    Raises:
      InvalidStatisticsError: If `stats` is not iterable, or if any
          object in the list is not a `Statistic` instance.
    """
    if not hasattr(stats, '__iter__'):
      raise InvalidStatisticsError(
          'Expecting iterable, got type %s' % type(stats))
    self._stats = [self._prepend_name(stat) for stat in stats]

  def _prepend_name(self, stat):
    """Returns a copy of `stat` with `self.name` prepended to `stat.name`."""
    if not isinstance(stat, statistics.Statistic):
      raise InvalidStatisticsError(
          'Expecting Statistic object, got %s' % stat)
    stat_copy = stat.copy()
    stat_copy.name = self._name + '_' + stat_copy.name
    return stat_copy

  def get_stats(self):
    """Returns Statistics about pipeline runs.

    Call `get_stats` after each call to `transform`.
    `transform` computes Statistics which will be returned here.

    Returns:
      A list of `Statistic` objects.
    """
    return list(self._stats)