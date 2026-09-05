import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable

class Normalise(AbstractProcessStep):
  """Normalises the image stack to [0, 1] range."""

  input_stack: Input[np.ndarray]
  '''A ndarray containing the pixel values from 1-max value'''

  normalised_stack: Deliverable[np.ndarray]
  '''A ndarray contating pixel values normalised to between 0 - 1'''

  def _execute(self):
    min_vals = self.input_stack.min(axis=(1, 2), keepdims=True)
    max_vals = self.input_stack.max(axis=(1, 2), keepdims=True)
    self.normalised_stack = (self.input_stack - min_vals) / (max_vals - min_vals + 1e-8)


process_steps["Normalise"] = Normalise
