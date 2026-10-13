import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input,Option, Deliverable


class GenerateFullMask(AbstractProcessStep):
  """Generate a mask that is '1' everywhere, and has the shape of the input image."""

  input_stack: Input[np.ndarray]
  """A ndarray conating pixel values corresponding to a tiff image/stack"""
  
  mask_stack: Deliverable[np.ndarray]
  """A mask containing either a 1 or a 0 in every postion. It has the same shape as the input stack"""

  boolean: Option[int] = 1

  frame: Option[int|None] = None
  "If suppplied the mask goes to this frame"

  def _execute(self):
    self.mask_stack = np.ones_like(self.input_stack, dtype=np.uint8)*self.boolean
    if self.frame is not None:
      self.mask_stack[self.frame:] = self.input_stack[self.frame:]

    




process_steps["GenerateFullMask"] = GenerateFullMask
