import numpy as np

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps
from image_processing_pipeline.processes.mixins.culling import CullingMixin

from image_processing_pipeline._types import Input, Deliverable


class CullBoundary(CullingMixin, AbstractProcessStep):
  """Crops the input_stack to the specified dimensions (see `CullingMixin`).

  The values from the cropping are set via the options. Following combinations are allowed
   - Setting `top`, `right`, `bottom`, `left` individually.
   - Setting `top` or `bottom` and `height`, and setting `left` or `right` and `width`.
   - Setting `offset` and either `right` (`bottom`) or `width` (`height`).
  """

  input_stack: Input[np.ndarray]
  """A ndarray containing pixel values corresponding to a full tiff image/stack."""

  culled_stack: Deliverable[np.ndarray]
  """A ndarray containing pixel values corresponding to a specified region of a tiff image/stack."""
  culled_image_offset: Deliverable[tuple]
  """Values which correspond to the cropping parameters."""

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
