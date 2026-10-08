import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable


class StarFill(AbstractProcessStep):
  """Fill the interior of a mask like stack.
  
      Pixels are considered to be in the interior, if they are surrounded by mask pixels on the same x and y axis.
  
      Assumptions: input_mask is either binary or contains only 0 and 1s. This is not checked however. Using this process
      for other inputs is considered undefined behaviour.
      """
  
  input_mask: Input[np.ndarray]
  """A ndarray containing a 0/1 mask, which may or may not have large regions of 0s surrounded by 1s"""
      
  output_mask: Deliverable[np.ndarray]
  """An ndarray containing a 0/1 mask, where any 0 that was surrounded by 1s has been converted to a 1"""
 
  def _execute(self):
    cs1 = np.cumsum(self.input_mask, axis=1)
    inner1 = cs1[:, -1, :][:, None, :] * cs1 - cs1**2
    cs2 = np.cumsum(self.input_mask, axis=2)
    inner2 = cs2[:, :, -1][:, :, None] * cs2 - cs2**2
    self.output_mask = (inner1 * inner2 >= 1).astype(np.int32)


process_steps["StarFill"] = StarFill
