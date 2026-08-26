"""Functionality for input stacks with an associated mask stack."""

import numpy as np

from image_processing_pipeline._types import Input, Option


class MaskedInputMixin():
  input_stack: Input[np.ndarray]
  mask_stack: Input[np.ndarray]

  mode: Option[str] = "interpolate"

  def _on_set_inputs(self):
    if self.input_stack.shape[0] <= self.mask_stack.shape[0]:
      raise IndexError("Input stack must have equal or greater depth than mask stack.")


  def _on_set_options(self):
    if self.input_stack.shape[0] == self.mask_stack.shape[0]:
      self.mode = "previous" # No interpolation needed
    if self.mode not in {"interpolate", "common_footprint", "previous", "next"}:
      raise ValueError(f"Unknown mode '{self.mode}'. Supported: interpolate, common_footprint, previous, next")

  
  def _get_mask_at_frame(self, frame_idx: int) -> np.ndarray:
    mask_idx = (
      0 if self.input_stack.shape[0] == 1 else
      frame_idx * (self.mask_stack.shape[0] - 1) / (self.input_stack.shape[0] - 1)
    )
    lower_idx = int(np.floor(mask_idx))
    upper_idx = int(np.ceil(mask_idx))
    if self.mode == "previous":
      weight_upper = 0
    elif self.mode == "next":
      weight_upper = 1
    else:
      weight_upper = mask_idx - lower_idx
    weight_lower = 1 - weight_upper

    return weight_lower * self.mask_stack[lower_idx] + weight_upper * self.mask_stack[upper_idx]