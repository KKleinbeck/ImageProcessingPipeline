import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)
from image_processing_pipeline._types import Input, Deliverable


class ShrinkToContent(AbstractProcessStep):
  """Shrinp the input stack to the smallest footprint, that contains all non-zero values.

  Iterate through all the arrays in the input stack and identifies a crop, which shrinks the
  stack to the smallest footprint which contains all non-zero/ non-false values.

  Returns the cropped stack and the applied offset (crop width & height implicitly
  communicated through the dimensions of the cropped stack).
  """

  input_stack: Input[np.ndarray]
  """A 0/1 mask stack, defining a region of interest (1s), the size of the array is based on the original image
  or on crop inputs defined in CullingMixIns"""

  output_stack: Deliverable[np.ndarray]
  """A 0/1 mask stack, defining a region of interest (1s), the size of the array is defined by the smallest footprint
  within the array which contains all of the 1 values."""
  offset: Deliverable[tuple]
  """Offset value which defines the position of the top left point of the crop"""

  def _execute(self):
    projection = np.any(self.input_stack, axis=0)

    a0_projection = np.any(projection, axis=0)
    a0_nonzero = a0_projection.nonzero()[0]
    a0_offset, a0_max = a0_nonzero[0], a0_nonzero[-1]
    a1_projection = np.any(projection, axis=1)
    a1_nonzero = a1_projection.nonzero()[0]
    a1_offset, a1_max = a1_nonzero[0], a1_nonzero[-1]

    self.offset = (a1_offset, a0_offset)
    self.output_stack = self.input_stack[:, a1_offset : a1_max + 1, a0_offset : a0_max + 1]


process_steps["ShrinkToContent"] = ShrinkToContent
