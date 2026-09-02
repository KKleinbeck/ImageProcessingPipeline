import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option

class ExtractFrames(AbstractProcessStep):
  """Extract a set of frames from the input stack."""
  
  input_stack: Input[np.ndarray]
  '''A stack of tiff file/s'''
  
  extracted_frames: Deliverable[np.ndarray]
  '''Tiff image/stack of frames that were extracted'''
  
  frames = Option[list] = [0]
  '''List of frames to be extracted'''


  def _on_set_options(self):
    n_frames = self.input_stack.shape[0]
    assert np.min(self.frames) >= -n_frames, (
      f"Frame range exceeded, tried to extract down to frame index {np.min(self.frames)}, "
      + f"but only {n_frames} frames available."
    )
    assert np.max(self.frames) < n_frames, (
      f"Frame range exceeded, tried to extract up to frame index {np.min(self.frames)}, "
      + f"but only {n_frames} frames available."
    )

  def _execute(self):
    self.extracted_frames = self.input_stack[self.frames, :, :]


process_steps["ExtractFrames"] = ExtractFrames
