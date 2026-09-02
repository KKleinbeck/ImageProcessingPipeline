import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option

class FourierDenoise(AbstractProcessStep):
  """Suppress small Fourier amplitudes.

  Transforms the stack into Fourier space, then sets all Fourier amplitudes smaller than
  `denoise_level * max(abs(amplitudes))` to 0.
  Finally transforms back to real space and delivers the result stack.
  """
  
  input_stack: Input[np.ndarray]
  """Stack of 0/1 masks, with potentially missing masks at the beginning or end.""" ########OR A TIFF STACK

  denoised_stack: Deliverable[np.ndarray]
  
  denoise_level: Option[float] = 1.0

  options = {"denoise_level": (float, 1.0)}

  def _on_set_options(self):
    assert 0.0 < self.denoise_level <= 1.0, "Denoise level must be in the range (0, 1]."

  def _execute(self):
    ft = np.fft.fft2(self.input_stack)
    ft[np.abs(ft) < self.denoise_level * np.max(np.abs(ft))] = 0
    self.denoised_stack = np.abs(np.fft.ifft2(ft))


process_steps["FourierDenoise"] = FourierDenoise
