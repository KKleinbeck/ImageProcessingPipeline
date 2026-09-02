import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable

class GenerateFullMask(AbstractProcessStep):
  """Generate a mask that is '1' everywhere, and has the shape of the input image."""
  
  input_stack: Input[np.ndarray]
  """A single/stack of tiff image/s"""
  
  mask_stack: Deliverable[np.ndarray]
  """A mask containing a 1 in every postion. It has the same shape as the input stack"""

  def _execute(self):
    if self.input_stack is None or self.input_stack.size == 0:
      raise ValueError("Input stack is empty, cannot generate mask")
    self.mask_stack = np.ones_like(self.input_stack, dtype=np.uint8)


process_steps["GenerateFullMask"] = GenerateFullMask
