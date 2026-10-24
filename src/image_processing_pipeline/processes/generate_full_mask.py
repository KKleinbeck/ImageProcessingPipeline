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
  """A mask containing either a 1 in every postion. It has the same shape as the input stack"""

  start_frame: Option[int] = 0
  "Set all prior frames in the mask stack to 0 instead of 1."

  def _execute(self):
    self.mask_stack = np.ones_like(self.input_stack, dtype=np.uint8)
    self.mask_stack[:self.start_frame,:,:] = 0

    


process_steps["GenerateFullMask"] = GenerateFullMask
