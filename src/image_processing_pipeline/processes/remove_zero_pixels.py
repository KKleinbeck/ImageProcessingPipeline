import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option
class RemoveZeroPixels(AbstractProcessStep):
  """Remove dead pixels from the image stack.
  
  Pixels with value 0 are considered dead pixels and replaced by either the minimum
  (bar 0 value pixels) or the maximum of the respective slice, depending on the options parameter.
  """

  input_stack: Input[np.ndarray]
  """A ndarray containing pixel values corresponding to a tiff image/stack """
  
  corrected_stack: Deliverable[np.ndarray]
  """A ndarray containing pixel values that have had outlers above a specified threshold removed. 
  These outliers have been set to the value of the threshold."""
  
  replace_by : Option[str]  = "min"
  """Defines what value replaces the 0 pixel values
  - "min" : 0 pixels are replaced by the minimum value (bar 0)
  - "max" : 0 pixels are replaced by the maximum value of the array""" 

  def _on_set_options(self):
    assert self.replace_by in ["min", "max"], "Option 'replace_by' must be either 'min' or 'max'."

  def _execute(self):
    replace_by = 0.0
    if self.replace_by == "min":
      replace_by = np.min(self.input_stack[self.input_stack > 0])
    elif self.replace_by == "max":
      replace_by = np.max(self.input_stack)

    self.corrected_stack = np.where(self.input_stack == 0, replace_by, self.input_stack)


process_steps["RemoveZeroPixels"] = RemoveZeroPixels
