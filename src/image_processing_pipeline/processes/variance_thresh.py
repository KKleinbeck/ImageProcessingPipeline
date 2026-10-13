import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, Deliverable

class VarianceThresh(AbstractProcessStep):
  """Computes frame at which the standard deviation of the region of interest meets a given threshold.

  The threshold is based upon
  """
  variance: Input[list]

  std_sustain : Option[int] = 5
  std_threshold : Option[float] = 0.1  

  variance_thresh: Deliverable[int]
  """Mean intensity of the masked region for each frame"""

  def _execute(self):
    normalised_variance = np.asarray(self.variance, dtype=float)
    normalised_variance /= np.max(normalised_variance)
    T = len(normalised_variance)
    onset = 0  # default: no onset detected
    print(normalised_variance)
    for t in range(T - self.std_sustain + 1):
      if np.all(normalised_variance[t : t + self.std_sustain] > self.std_threshold):
        onset = t
        break
    print(onset)

    self.variance_thresh = onset
      
process_steps["VarianceThresh"] = VarianceThresh
