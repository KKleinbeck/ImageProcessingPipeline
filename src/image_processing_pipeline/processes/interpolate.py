import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)
from image_processing_pipeline._types import Input, Deliverable, Option

class Interpolate(AbstractProcessStep):
  """Generate interpolated result at given index.
  
      Fills the interpolated stack at index `i` by interpolating the with values of the
      input stack at index `s` (start) and `e` (end), depending on the mode.
      - `mode == "interpolate"`: Takes the values of the input stack at index `s` and `e` and weights them according to
        their distance to `i` for the interpolation.
      - `mode == "previous"`: Fills the interpolated stack with the value of the input stack at index `s`.
      - `mode == "next"`: Fills the interpolated stack with the value of the input stack at index `e`.
      - `mode == "common_footprint"`: Only for mask inputs. Fills the interpolated stack with overlap of the masks at
        index `s` and `e`.
      """
 
  input_stack: Input[np.ndarray]
  """Stack of 0/1 masks, with potentially missing masks in the middle"""
  
  interpolated_stack: Deliverable[np.ndarray]
  """Processed 0/1 mask stack. If frames in the middle were initially blank, they now contain interpolated 0/1 masks."""
  interpolated_frames: Deliverable[list]
  """List of booleans, indicating which frames have been interpolated."""

  mode: Option[str] = "common_footprint"
  """Different modes for interpolation.
      - "interpolate"`: Takes the values of the input stack at index `s` and `e` and weights them according to
        their distance to `i` for the interpolation.
      -"previous"`: Fills the interpolated stack with the value of the input stack at index `s`.
      -"next"`: Fills the interpolated stack with the value of the input stack at index `e`.
      - "common_footprint"`: Only for mask inputs. Fills the interpolated stack with overlap of the masks at
        index `s` and `e`."""
  

  def _on_set_options(self):
    if self.mode not in {"interpolate", "common_footprint", "previous", "next"}:
      raise ValueError(f"Unknown mode '{self.mode}'. Supported: interpolate, common_footprint, previous, next")

    if self.mode == "common_footprint":
      assert np.isin(1.0 * self.input_stack, [0.0, 1.0]).all(), (
        "Mode 'common_footprint' requires input_stack to have only 0 & 1 or binary values."
      )

  def _interpolate(self, i: int, s: int, e: int) -> None:
    
    match self.mode:
      case "interpolate":
        w1 = (i - s + 1) / (e - s + 2)
        w2 = (e + 1 - i) / (e - s + 2)
        self.interpolated_stack[i, :] = w1 * self.input_stack[s - 1, :] + w2 * self.input_stack[e + 1, :]
      case "previous":
        self.interpolated_stack[i, :] = self.input_stack[s - 1, :]
      case "next":
        self.interpolated_stack[i, :] = self.input_stack[e + 1, :]
      case "common_footprint":
        self.interpolated_stack[i, :] = self.input_stack[s - 1, :] * self.input_stack[e + 1, :]

  def _execute(self):
    # Find missing frames
    missing_frames = np.any(self.input_stack, axis=(1, 2))
    first_valid_frame = np.argmax(missing_frames)
    last_valid_frame = missing_frames.size - np.argmax(missing_frames[::-1])

    # Transform into prediction for interpolated frames, removing missing frames at the start / end
    self.interpolated_frames = ~missing_frames
    self.interpolated_frames[:first_valid_frame] = False
    self.interpolated_frames[last_valid_frame:] = False

    # Find where the value changes (from False to True or True to False)
    diffs = np.diff(self.interpolated_frames.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]

    # Interpolate
    self.interpolated_stack = self.input_stack
    if self.mode == "interpolate" and np.any(self.interpolated_frames):
      self.interpolated_stack = self.interpolated_stack.astype("float32")
    for s, e in zip(starts, ends):
      for i in range(s, e + 1):
        self._interpolate(i, s, e)
    self.interpolated_frames = self.interpolated_frames.tolist()  # To support serialisation


process_steps["Interpolate"] = Interpolate
