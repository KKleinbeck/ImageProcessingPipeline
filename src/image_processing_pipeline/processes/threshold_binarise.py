import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option  # WE ALSO NEVER USE THIS STEP, DO WE NEED IT


class ThresholdBinarise(AbstractProcessStep):
  """Binerises the image stack based on a threshold.

  Assume input is normalised to [0,1]. For this every pixel value below the threshold
  is set to 0, every pixel value above or equal to the threshold is set to 1.
  """

  input_stack: Input[np.ndarray]
  """A ndarray that has pixel values that have been normalised to [0,1]"""  # THIS IS NOT A MASK CORRECT ??

  binary_stack: Deliverable[np.ndarray]
  """A ndarray that has been binarised to 0 or 1, based on a threshold"""

  threshold: Option[float] = 0.5
  """Threshold value. All values above this will be converted to 1, all values below this will be converted to 0."""
  allow_non_normalised: Option[bool] = False

  def _on_set_options(self):
    if not (np.all((self.input_stack >= 0) & (self.input_stack <= 1))) and not self.allow_non_normalised:
      raise ValueError("ThresholdBinarise: Input stack must be in [0, 1] range.")

    if not (0 <= self.threshold <= 1) and not self.allow_non_normalised:
      raise ValueError("ThresholdBinarise: Threshold must be in [0, 1] range.")

  def _execute(self):
    self.binary_stack = self.input_stack > self.threshold


process_steps["ThresholdBinarise"] = ThresholdBinarise
