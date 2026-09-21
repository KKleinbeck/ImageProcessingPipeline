import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option
class ThresholdBinarise(AbstractProcessStep):
  input_stack: Input[np.ndarray]

  binary_stack: Deliverable[np.ndarray]

  threshold: Option[float] = 0.5
  allow_non_normalised: Option[bool] = False

  def _on_set_options(self):
    if (
      not ( np.all((self.input_stack >= 0) & (self.input_stack <= 1)) )
      and
      not self.allow_non_normalised
    ):
      raise ValueError("ThresholdBinarise: Input stack must be in [0, 1] range.")

    if not (0 <= self.threshold <= 1) and not self.allow_non_normalised:
      raise ValueError("ThresholdBinarise: Threshold must be in [0, 1] range.")

  def _execute(self):
    """Binerises the image stack based on a threshold.

    Assume input is normalised to [0,1]. For this every pixel value below the threshold
    is set to 0, every pixel value above or equal to the threshold is set to 1.
    """
    self.binary_stack = self.input_stack > self.threshold


process_steps["ThresholdBinarise"] = ThresholdBinarise
