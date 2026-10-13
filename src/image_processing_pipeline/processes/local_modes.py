"""Sums up a list of provided input numbers."""

import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, RegexDeliverable


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

  m_: RegexDeliverable[np.ndarray, r"m_\d+"]
  """Stack of local variance maps of the input frames."""

  kernel_type: Option[str] = "gaussian"
  """Averaging kernel. Supports "gaussian" and "constant"."""
  kernel_size: Option[int] = 7
  """Pixel size (width and hight) of the kernel.

  For a Gaussian kernel sigma will be roughly a third of the kernel_size and the kernel size will
  always be rounded to an odd number.
  """
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
      return np.ones((self.kernel_size, self.kernel_size)) / self.kernel_size**2

    x_max = self.kernel_size // 2
    sigma = x_max / 3
    xs = np.linspace(-x_max, x_max, 2*x_max + 1)
    gaussian_kernel_1d =  np.exp(- xs**2 / (2. * sigma**2))
    return np.outer(gaussian_kernel_1d, gaussian_kernel_1d) / np.sum(gaussian_kernel_1d)**2


  def _execute(self):
    kernel = self._get_kernel()

    for mode in self.deliverables_actual.keys():
      n = int(mode.split("_")[1])
      setattr(
        self,
        mode,
        nd.convolve(self.input_stack**n, kernel, mode=self.mode, axes=(1, 2))
      )


process_steps["LocalModes"] = LocalModes
