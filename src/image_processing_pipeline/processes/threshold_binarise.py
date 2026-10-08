import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option
class ThresholdBinarise(AbstractProcessStep):
  """Binerises the image stack based on a threshold.
  
      Assume input is normalised to [0,1]. For this every pixel value below the threshold
      is set to 0, every pixel value above or equal to the threshold is set to 1.
      """
  
  input_stack: Input[np.ndarray]
  """A ndarray containing values that have been normalised to between 0 and 1"""
        
  binary_stack: Deliverable[np.ndarray]
  """An ndarray containing a 0/1 mask, where any 0 values were below the threshold, and 1 values were above"""

  threshold : Option[float] = 0.5
  """ Threshold value which defines what values will be considered 0 (below), and what values will be considered 1 (above)."""

  def _on_set_inputs(self):
    assert np.all((self.input_stack >= 0) & (self.input_stack <= 1)), "Input stack must be in [0, 1] range."

  def _on_set_options(self):
    assert 0 <= self.threshold <= 1, "Threshold must be in [0, 1] range."

  def _execute(self):
    self.binary_stack = self.input_stack > self.threshold


process_steps["ThresholdBinarise"] = ThresholdBinarise
