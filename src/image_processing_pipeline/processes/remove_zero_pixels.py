import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option


class RemoveZeroPixels(AbstractProcessStep):
  """Remove dead pixels from the ndarray.

  Pixels with value 0 are considered dead pixels and replaced by either the minimum
  (bar 0 value pixels) or the maximum of the respective slice, depending on the options parameter.
  """

  input_stack: Input[np.ndarray]
  """A ndarray containing 0 values (which represent dead pixels with no data)"""

  corrected_stack: Deliverable[np.ndarray]
  """A ndarray containing no 0 values. The 0 values have been replaced in accordance with the options parameter."""

  replace_by: Option[str] = "min"
  """Defines what value is used to replace the 0 values
  -min: The 0 values are replaced by the minimum value (bar 0) of the frame
  -max: The 0 values are replaced by the maximum value of the frame"""

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
