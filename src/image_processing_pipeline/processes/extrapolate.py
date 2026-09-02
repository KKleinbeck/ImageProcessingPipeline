import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable


class Extrapolate(AbstractProcessStep):
  """Extrapolate input stack a the beginning and end.

  Scans the input stack for frames that have no generated mask (e.g. every pixel has a 0 value)
  at the beginning or end and extrapolates these frames according to the closest
  frame.
  """

  input_stack: Input[np.ndarray]
  """Stack of 0/1 masks, with potentially missing masks at the beginning or end."""

  extrapolated_stack: Deliverable[np.ndarray]
  """Processed 0/1 mask stack. If frames at the beginning/end were initially blank, they now contain an extrapolated 0/1 masks."""
  extrapolated_frames: Deliverable[list]
  """List of booleans, indicating which frame have been extrapolated."""

  def _execute(self):
    # Find missing frames
    missing_frames = np.any(self.input_stack, axis=(1, 2))
    first_valid_frame = np.argmax(missing_frames)
    last_valid_frame = missing_frames.size - np.argmax(missing_frames[::-1])

    # Transform into prediction for interpolated frames, removing missing frames at the start / end
    extrapolated_frames = ~missing_frames
    extrapolated_frames[first_valid_frame:last_valid_frame] = False

    # Extrapolate
    self.extrapolated_stack = self.input_stack
    self.extrapolated_stack[:first_valid_frame, :, :] = self.input_stack[first_valid_frame, :, :]
    self.extrapolated_stack[last_valid_frame:, :, :] = self.input_stack[last_valid_frame - 1, :, :]
    self.extrapolated_frames = extrapolated_frames.tolist()  # To support serialisation


process_steps["Extrapolate"] = Extrapolate
