import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, Deliverable

class ThresholdDetector(AbstractProcessStep):
  """Computes frame at which the standard deviation of the region of interest meets a given threshold.

  The threshold is based upon
  """
  values: Input[list]

  sustain : Option[int] = 5
  threshold : Option[float] = 0.1  

  first_frame_over_threshold: Deliverable[int]
  """Mean intensity of the masked region for each frame"""

  def _execute(self):
    normalised_values = np.asarray(self.values, dtype=float)
    normalised_values /= np.max(normalised_values)
    print(normalised_values)
    T = len(normalised_values)
    onset = 0  # default: no onset detected
    
    for t in range(T - self.sustain + 1):
      if np.all(normalised_values[t : t + self.sustain] > self.threshold):
        onset = t
        break
    print(onset)    
    self.first_frame_over_threshold = onset
      
process_steps["ThresholdDetector"] = ThresholdDetector
