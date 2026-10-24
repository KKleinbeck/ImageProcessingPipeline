"""Sums up a list of provided input numbers."""

import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, RegexDeliverable, Deliverable


class LocalModes(AbstractProcessStep):
  """Calculates the local modes of the input stack.

  For each frame in the input stack calculate the modes
  $$
    m_n(x,y) = E_{x,y}(i^n),
  $$
  where $i$ is the input stack, $i^n$ the pixelwise power and $E_{x,y}$ calculates the local
  average. The area that is included in this calculation can be set via the option `kernel_size`.
  Averaging can be performed with constant or gaussian pixel weights.
  """

  input_stack: Input[np.ndarray]
  """Input stack to be analysed."""

  mask_stack: Input[np.ndarray]
  """Input stack to be analysed."""

  m_: RegexDeliverable[np.ndarray, r"m_\d+"]
  """Stack of local variance maps of the input frames."""

  kernel_type: Option[str] = "gaussian"
  """Averaging kernel. Supports "gaussian" and "constant"."""
  kernel_size: Option[int] = 11
  """Pixel size (width and hight) of the kernel.

  For a Gaussian kernel sigma will be roughly a third of the kernel_size and the kernel size will
  always be rounded to an odd number."""
  mode: Option[str] = "constant"
  """Parameter determines how input frame is extended beyond its boundaries.

  See `scipy.ndimage.convolve`."""

  def _on_set_options(self):
    self.kernel_type = self.kernel_type.lower()
    if self.kernel_type not in ["constant", "gaussian"]:
      raise ValueError(
        f"LocalModes: `kernel_type` can only be 'constant' or 'gaussian'.\n\tGot {self.kernel_type} instead."
      )

    if self.kernel_size <= 0:
      raise ValueError(
        f"LocalModes: `kernel_size` Must be larger than 0.\n\tGot {self.kernel_size} instead."
      )


  def _get_kernel(self) -> np.ndarray:
    if self.kernel_type == "constant":
      return np.ones(self.kernel_size) / self.kernel_size

    x_max = self.kernel_size // 2
    sigma = x_max / 3
    xs = np.linspace(-x_max, x_max, 2*x_max + 1)
    gaussian_kernel =  np.exp(- xs**2 / (2. * sigma**2))
    return gaussian_kernel / gaussian_kernel.sum()


  def _execute(self):
    kernel = self._get_kernel()
    mask_float = self.mask_stack.astype(np.float32)
    denominator = nd.convolve1d(mask_float, kernel, axis=1, mode=self.mode)
    denominator = nd.convolve1d(denominator, kernel, axis=2, mode=self.mode)

    for mode_key in self.deliverables_actual.keys():
      n = int(mode_key.split("_")[1])
      powered = self.input_stack ** n

      numerator = nd.convolve1d(powered, kernel, axis=1, mode=self.mode)
      numerator = nd.convolve1d(numerator,     kernel, axis=2, mode=self.mode)
      setattr(self, mode_key, numerator / np.maximum(denominator, 1e-9))


process_steps["LocalModes"] = LocalModes
