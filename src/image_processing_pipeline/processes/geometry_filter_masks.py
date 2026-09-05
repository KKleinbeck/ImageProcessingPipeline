import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)
from image_processing_pipeline._types import Input, Deliverable, Option

class GeometryFilterMasks(AbstractProcessStep):
  """Applies geometric filtering to connected components in the input binary mask stack.
  
      For each connected component in each slice of the input stack, the following criteria are checked:
      - Aspect Ratio: The ratio of width to height (dx/dy) and height to width (dy/dx) must be above specified minimums.
      - Area: The area (width * height) must be within specified minimum and maximum bounds.
      - Size: The width (dx) and height (dy) must be within specified minimum and maximum bounds.
   """
  
  input_stack: Input[np.ndarray]
  """A 0/1 mask stack which contains a variety of regions of interest (1s)"""
  
  filtered_mask_stack: Deliverable[np.ndarray]  
  """A 0/1 mask stack which contains only regions of interest that meet the inputted requirements"""
  
  min_aspect_dx_dy: Option[float] = 0.0
  """Minimum aspect ratio value of width to height (dx/dy)"""
  min_aspect_dy_dx: Option[float] = 0.0
  """Minimum aspect ratio value of height to width (dy/dx)"""
  min_area: Option[float] = 0.0
  """Minimum value for area in pixels"""
  max_area: Option[float] = np.inf
  """Maximum value for area in pixels"""
  min_size_dx: Option[float] = 0.0
  """Minimum value in pixels for the width along the x-axis"""
  max_size_dx: Option[float] = np.inf
  """Maximum value in pixels for the width along the x-axis"""
  min_size_dy: Option[float] = 0.0
  """Minimum value in pixels for the height along the y-axis"""
  max_size_dy: Option[float] = np.inf
  """Maximum value in pixels for the height along the y-axis"""


  def _execute(self):
    for n in range(self.input_stack.shape[0]):
      labelled, _ = nd.label(self.input_stack[n, :, :])
      for indices in nd.value_indices(labelled).values():
        dX = np.max(indices[0]) - np.min(indices[0])
        dY = np.max(indices[1]) - np.min(indices[1])
        area = len(indices[0])
        if area < self.min_area or area > self.max_area:
          self.input_stack[n, *indices] = 0
          continue
        if dX < self.min_size_dx or dX > self.max_size_dx or dY < self.min_size_dy or dY > self.max_size_dy:
          self.input_stack[n, *indices] = 0
          continue
        if dX / (dY + 1e-6) < self.min_aspect_dx_dy or dY / (dX + 1e-6) < self.min_aspect_dy_dx:
          self.input_stack[n, *indices] = 0
    self.filtered_mask_stack = self.input_stack


process_steps["GeometryFilterMasks"] = GeometryFilterMasks
