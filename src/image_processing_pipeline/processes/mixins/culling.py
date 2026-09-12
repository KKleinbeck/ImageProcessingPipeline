from image_processing_pipeline._types import Option, Deliverable


class CullingMixin:
  top: Option[int | float] = 0
  "Cull frame from the top. Incompatible with options `height` and `offset`."
  bottom: Option[int | float | None] = None
  "Cull image from the bottom. Incompatible with option `height`"
  left: Option[int | float] = 0
  "Cull frame from the left. Incompatible with options `width` and `offset`."
  right: Option[int | float | None] = None
  "Cull frame from the right. Incompatible with option `width`."
  width: Option[int | float] = 0
  "Cull frame to given width. Incompatible with setting both `left` and `right`."
  height: Option[int | float] = 0
  "Cull frame to given height. Incompatible with setting both `top` and `bottom`."
  offset: Option[tuple | None] = None
  "Cull image at given offset (i.e., tuple of top and left cull). Incompatible with setting options `left` and `top`."

  former_image_shape: Deliverable[tuple]
  "Shape of the frames prior to culling."

  def _on_set_options(self):
    # Guarantee: Left, right, top, bottom parameters are in the correct format.
    for option in ["left", "right", "top", "bottom"]:
      value = getattr(self, option)

      if isinstance(value, float):
        assert 0 <= value < 1.0, f"Float option {option} must be in [0, 1.0) range."
        max_pixel_size = self.former_image_shape[1 if option in ["left", "right"] else 0]
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
    if (
      self.top + (self.bottom or 0) >= self.former_image_shape[0]
      or self.left + (self.right or 0) >= self.former_image_shape[1]
    ):
      raise ValueError(
        f"Culling options too large for image size {self.former_image_shape}: "
        f"top {self.top}, bottom {self.bottom}, left {self.left}, right {self.right}."
      )
