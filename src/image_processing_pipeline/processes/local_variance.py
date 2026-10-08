"""Sums up a list of provided input numbers."""

import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, Deliverable


class LocalVariance(AbstractProcessStep):
  """Calculates the local variance of the input stack.

  For each frame in the input stack calculate the local variance. The local
  variance of each pixel is determined by calculating the variance of the
  pixels in the vicinity. The area that is included in this calculation can
  be set via the option `kernel_size`.
  """

  input_stack: Input[np.ndarray]
  """Input stack to be analysed."""

  local_variance_stack: Deliverable[np.ndarray]
  """Stack of local variance maps of the input frames."""

  kernel_size: Option[int] = 5
  """Size of the kernel (width and height)."""
  mode: Option[str] = "constant"
  """Parameter determines how input frame is extended beyond its boundaries.

  See `scipy.ndimage.convolve`."""

  def _execute(self):
    kernel = np.ones((self.kernel_size, self.kernel_size))

    local_mean = nd.convolve(self.input_stack, kernel, mode=self.mode, axes=(1, 2)) / self.kernel_size**2
    local_mean_of_squares = nd.convolve(self.input_stack**2, kernel, mode=self.mode, axes=(1, 2)) / self.kernel_size**2
    self.local_variance_stack = local_mean_of_squares - local_mean**2


process_steps["LocalVariance"] = LocalVariance
