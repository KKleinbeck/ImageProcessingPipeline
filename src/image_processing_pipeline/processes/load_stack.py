from pathlib import Path

import numpy as np
import tifffile as tiff

from image_processing_pipeline.framework.process_step import process_steps
from image_processing_pipeline.processes.cull_boundary import CullBoundary

from image_processing_pipeline._types import Input, Deliverable

class LoadStack(CullBoundary):  # Inherit from CullBoundary to reuse its options
  """Load a stack from a multipage tiff file.
  
  Opens an image from a tiff file and crops the image to a region of interest (defined by CullBoundary).
  If full_image_processing is True, no croppping will occur"""

  input_path: Input[Path]
  """Input path to tiff file"""
  full_image_processing: Input[bool] = False
  """A boolean true/fale input.
     -True: No cropping or culling of any region of the image will occur. Accordingly, no chamber recognition occurs. 
     -False: The images in the tiff stack will be cropped to a region of interest."""
  
  loaded_stack: Deliverable[np.ndarray]
  """Tiff stack that is cropped to a region of interest (defined by CullBoundary)"""
  former_image_shape: Deliverable[tuple]
  """Original shape of frame 0 of the initial tiff file"""
  culled_image_offset: Deliverable[tuple]
"""Image offset""" #I DO NOT KNOW WHAT THIS DOES
  # Options and option verification inherited from CullBoundary

  def _on_set_inputs(self):
    with tiff.TiffFile(self.input_path) as tif:
      assert len(tif.series) == 1, f"Can only load tif files with a single series, got {tif.series} instead."
      self.former_image_shape = tif.pages[0].shape

  def _execute(self):
    top, bottom = self.top, self.bottom
    left, right = self.left, self.right

    if not self.full_image_processing:
      with tiff.TiffFile(self.input_path) as tif:
        self.loaded_stack = np.array(
          [page.asarray()[top:-bottom, left:-right] for page in tif.pages], dtype=tif.pages[0].dtype
        )

      self.culled_image_offset = (top, left)
    else:
      with tiff.TiffFile(self.input_path) as tif:
        self.loaded_stack = np.array([page.asarray() for page in tif.pages], dtype=tif.pages[0].dtype)

      self.culled_image_offset = (top, left)


process_steps["LoadStack"] = LoadStack
