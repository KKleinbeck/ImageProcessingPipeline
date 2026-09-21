import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)
from image_processing_pipeline._types import Input, Deliverable, Option


class RemoveOutliers(AbstractProcessStep):
  """Remove outliers from the ndarray.

  or this every pixel value below or above the quantile specified in the options parameter
  is set to the respective quantile values.
  """

  input_stack: Input[np.ndarray]
  """A ndarray containing the entire range of pixel values"""

  filtered_stack: Deliverable[np.ndarray]
  """A ndarray, containing values from the initial input_stack that have been stripped of outliers.
  Values above or below the specified quantile are set to the quantile value """

  lower_quantile: Option[float] = 0.0
  """The value for the lower quantile. Any values below this will instead be set to this value"""
  upper_quantile: Option[float] = 1.0
  """The value for the upper quantile. Any values above this will instead be set to this value"""

  def _execute(self):
    # Get quantiles for each slice
    ql = self.lower_quantile
    qh = self.upper_quantile
    qs = np.quantile(self.input_stack, [ql, qh], axis=(1, 2))

    # Extract low/high, reshape to broadcast over the input stack height
    low = qs[0, :, None, None]
    high = qs[1, :, None, None]
    self.filtered_stack = np.clip(self.input_stack, low, high)


process_steps["RemoveOutliers"] = RemoveOutliers
