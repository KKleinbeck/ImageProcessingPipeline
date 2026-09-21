import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps
from image_processing_pipeline.processes.mixins.culling import CullingMixin

from image_processing_pipeline._types import Input, Deliverable


class CullBoundary(CullingMixin, AbstractProcessStep):
  """Binerises the image stack based on a threshold.

  Assume input is normalised to [0,1]. For this every pixel value below the threshold
  is set to 0, every pixel value above or equal to the threshold is set to 1.
  """  #########DOES THIS REALLY DO THIS ? DOESN'T IT CROP THE IMAGE ??

  input_stack: Input[np.ndarray]

  culled_stack: Deliverable[np.ndarray]
  culled_image_offset: Deliverable[tuple]

  def _on_set_inputs(self):
    self.former_image_shape = self.input_stack.shape[1:]

  def _execute(self):
    top, bottom = self.top, self.bottom
    left, right = self.left, self.right

    self.culled_stack = self.input_stack[
      :, top : (bottom if bottom is None else -bottom), left : (right if right is None else -right)
    ]
    self.culled_image_offset = (top, left)


process_steps["CullBoundary"] = CullBoundary
