from image_processing_pipeline.processes.mixins.culling import CullingMixin
from pathlib import Path

import numpy as np
import tifffile as tiff

from image_processing_pipeline.framework.process_step import AbstractProcessStep, process_steps

from image_processing_pipeline._types import Input, Deliverable


class LoadStack(AbstractProcessStep, CullingMixin):
  input_path: Input[Path]

  loaded_stack: Deliverable[np.ndarray]
  former_image_shape: Deliverable[tuple]
  culled_image_offset: Deliverable[tuple]

  def _on_set_inputs(self):
    with tiff.TiffFile(self.input_path) as tif:
      assert len(tif.series) == 1, f"Can only load tif files with a single series, got {tif.series} instead."
      self.former_image_shape = tif.pages[0].shape

  def _execute(self):
    """Load a stack from a multipage tiff file."""
    top, bottom = self.top, self.bottom
    left, right = self.left, self.right

    crop_input = top != 0 or (bottom != 0 and bottom is not None) or left != 0 or (right != 0 and right is not None)
    if crop_input:
      with tiff.TiffFile(self.input_path) as tif:
        self.loaded_stack = np.array(
          [
            page.asarray()[top : (bottom if bottom is None else -bottom), left : (right if right is None else -right)]
            for page in tif.pages
          ],
          dtype=tif.pages[0].dtype,
        )

      self.culled_image_offset = (top, left)
    else:
      with tiff.TiffFile(self.input_path) as tif:
        self.loaded_stack = np.array([page.asarray() for page in tif.pages], dtype=tif.pages[0].dtype)

      self.culled_image_offset = (0, 0)


process_steps["LoadStack"] = LoadStack
