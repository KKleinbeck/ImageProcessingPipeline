"""Sums up a list of provided input numbers."""

import re

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import RegexInput, Option, Deliverable


class NumberAdder(AbstractProcessStep):
  """Adds up provided input numbers.

  Adds all inputs of the form `number_{x: int}` together with an optional
  offset `extra_summand` (default 0), and returns the result via `sum`.
  """

  number: RegexInput[float | int, r"number_\d+"]
  """Regex input numbers, each representing one summand."""

  sum: Deliverable[float | int]
  """Result of the summation."""

  extra_summand: Option[int] = 0
  """Offset added to the result."""

  def _execute(self):
    self.sum = self.extra_summand

    for field_name in self.inputs_actual:
      if not re.match(r"number_\d+", field_name):
        continue
      self.sum += getattr(self, field_name)


process_steps["NumberAdder"] = NumberAdder
