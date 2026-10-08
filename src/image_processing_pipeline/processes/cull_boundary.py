import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option

class CullBoundary(AbstractProcessStep):
  """ Crops the ndarray to specified dimensions that define a region of interest.

  The values fro the cropping are set via the options. For width and height, you can set up to 2 of the 3 parameters.
  The width/height value will be followed first, followed by removing data from the exterior edges. 
   - Width: Specified by left, right and width. 
   - Height: pecified by top, bottom and height.
   - Offset:  
  """ 

  input_stack: Input[np.ndarray]
  """A ndarray containing pixel values corresponding to a full tiff image/stack """
  culled_stack: Deliverable[np.ndarray]
  """A ndarray containing pixel values corresponding to a specified region of a tiff image/stack """

  former_image_shape: Deliverable[tuple]
  """The original shape of the ndarray prior to cropping"""
  culled_image_offset: Deliverable[tuple]
  """Values which correspond to the cropping parameters."""
  
  top: Option[int,float] = 0
  """Number of rows to remove from the top"""
  bottom: Option[int,float] = 0
  """Number of rows to remove from the bottom"""
  left: Option[int,float] = 0
  """Number of columns to remove from the left"""
  right: Option[int,float] = 0
  """Number of columns to remove from the right"""
  width: Option[int,float] = 0
  """Limit to number of columns wide, from left to right"""
  height: Option[int | float] = 0 
  """Limit of number of rows long, from top to bottom"""
  offset: Option[tuple | None, None]
  """NEEDS TO BE COMMENTED"""

  def _on_set_inputs(self):
    self.former_image_shape = self.input_stack.shape[1:]

  def _on_set_options(self):
    # Guarantee: Left, right, top, bottom parameters are in the correct format.
    for option in ["left", "right", "top", "bottom"]:
      value = getattr(self, option)
      max_pixel_size = self.former_image_shape[1 if option in ["left", "right"] else 0]

      if isinstance(value, float):
        assert 0 <= value < 1.0, f"Float option {option} must be in [0, 1.0) range."
        pixel_value = int(value * max_pixel_size)
        setattr(self, option, pixel_value)
      else:
        assert isinstance(value, int) and 0 <= value, f"Integer option {option} must be a non-negative integer."

    # Guarantee: If Offset is set, this does not clash with other parameters
    if self.offset is not None:
      assert self.top == 0 and self.left == 0, (
        f"Got option `offset` ({self.offset}), and options `left` ({self.left}) and `top` ({self.top}).\n"
        + "Only one set must be provided"
      )
      self.top = self.offset[0]
      self.left = self.offset[1]

    # Guarantee: If width and height is set, this does not clash with the other parameters
    if self.width != 0:
      if self.right != 0 and self.left != 0:
        raise ValueError(
          "Attempting to set option 'width', 'left', and 'right' simulatneously."
          + "Only two of these parameters can be specified in one instance"
        )

      if isinstance(self.width, float):
        assert 0 <= value < 1.0, "Float option width must be in [0, 1.0) range."
        self.width = int(self.width * self.former_image_shape[1])

      if self.left != 0:
        self.right = self.former_image_shape[1] - self.left - self.width
      else:
        self.left = self.former_image_shape[1] - self.right - self.width

    if self.height != 0:
      if self.top != 0 and self.bottom != 0:
        raise ValueError(
          "Attempting to set option 'height', 'top', and 'bottom' simulatneously."
          + "Only two of these parameters can be specified in one instance"
        )

      if isinstance(self.height, float):
        assert 0 <= value < 1.0, "Float option height must be in [0, 1.0) range."
        self.height = int(self.height * self.former_image_shape[0])

      if self.top != 0:
        self.bottom = self.former_image_shape[0] - self.top - self.height
      else:
        self.top = self.former_image_shape[0] - self.bottom - self.height

    # Guarantee: Culling parameters lie within the image
    if self.top + self.bottom >= self.former_image_shape[0] or self.left + self.right >= self.former_image_shape[1]:
      raise ValueError(
        f"Culling options too large for image size {self.former_image_shape}: "
        f"top {self.top}, bottom {self.bottom}, left {self.left}, right {self.right}."
      )

  def _execute(self):
    top, bottom = self.top, self.bottom
    left, right = self.left, self.right

    self.culled_stack = self.input_stack[:, top:-bottom, left:-right]
    self.culled_image_offset = (top, left)


process_steps["CullBoundary"] = CullBoundary
