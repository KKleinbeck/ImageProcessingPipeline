import numpy as np
from scipy.ndimage import median_filter

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option

class MedianFilter(AbstractProcessStep):
   """Apply a median filter to the image stack.
  
      For this the scipy.ndimage.median_filter function is used. The filter is applied
      `iterations` times with a filter size of `size`.
      """
   input_stack: Input[np.ndarray]
   """Stack of 0/1 masks"""

   filtered_stack: Deliverable[np.ndarray]

   iterations: Option[int] =1
   size: Option[int] = 3
 

  def _execute(self):
    self.filtered_stack = np.copy(self.input_stack)
    for _ in range(self.iterations):
      self.filtered_stack = median_filter(self.filtered_stack, size=self.size, axes=(1, 2))


process_steps["MedianFilter"] = MedianFilter
