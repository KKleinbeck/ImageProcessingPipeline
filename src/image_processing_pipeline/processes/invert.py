import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)
from image_processing_pipeline._types import Input, Deliverable


class Invert(AbstractProcessStep):
  """Inverts the image stack.

  Takes an input in range [0, 1]. Then the image is inverted by calculating 1 - image.
  """

  input_stack: Input[np.ndarray]
  """An ndarray of [0, 1] values."""

  inverted_stack: Deliverable[np.ndarray]
  """An ndarray of of inverted values, i.e., 1 - input_stack."""

  def _on_set_inputs(self):
    assert np.all((self.input_stack >= 0) & (self.input_stack <= 1)), "Input stack must be in [0, 1] range."

  def _execute(self):
    self.inverted_stack = 1 - self.input_stack


process_steps["Invert"] = Invert
