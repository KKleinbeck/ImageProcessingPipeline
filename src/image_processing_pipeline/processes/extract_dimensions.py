import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable

class ExtractDimensions(AbstractProcessStep):
  """Extract the shape of a tiff stack; 
  
  Will extract the depth, height and width of a stack of tiff files.
  2d images will have a depth of 0."""
  
  input_stack: Input[np.ndarray]
  '''A single/stack of tiff file/s'''
  
  depth: Deliverable[int]
  '''Depth of tiff stack in frames'''
  width: Deliverable[int]
  '''Width of image in tiff stack in pixels'''
  height: Deliverable[int]
  '''Height of image in tiff stack in pixels'''
  
  def _on_set_inputs(self):
    assert self.input_stack.ndim in [2, 3], f"Input stack must be 2D or 3D, got {self.input_stack.ndim}D."

  def _execute(self):
    if self.input_stack.ndim == 3:
      self.depth, self.height, self.width = self.input_stack.shape
    else:
      self.depth = 0
      self.height, self.width = self.input_stack.shape


process_steps["ExtractDimensions"] = ExtractDimensions
