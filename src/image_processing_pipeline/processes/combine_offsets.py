import re

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import RegexInput, Option, Deliverable

class CombineOffsets(AbstractProcessStep):
  """Combines multiple offsets through element-wise addition to provide a specific region of interest (ROI).
  
  Takes the offsets that are generated in previous steps and combines them to generate a 'region of interest' (ROI) around the structure. 
  The extra_horizontal and extra_vertical provide a buffer around the ROI to ensure the cropped region contains the entire ROI in case of imaging drift.
  """
  offset_: RegexInput[tuple, r"q\d+"] # I DO NOT KNOW HOW TO DO THIS ONE  HELP
  inputs = {r"offset_\d+": tuple}
  
  combined_offset : Deliverable[tuple]
  '''Offset value which gives the co-ordinates to crop the image'''
  
  extra_horizontal: Option[int] = 0
  '''Extra amount of offset to add to the width of the region (in pixels)'''
  extra_vertical = Option[int] = 0
  '''Extra amount of offset to add to the height of the region (in pixels)'''


  def _execute(self):
    v_offset = self.extra_vertical
    h_offset = self.extra_horizontal

    for field_name in self.inputs_actual:
      if not re.match(r"offset_\d+", field_name):
        continue

      offset = getattr(self, field_name)
      v_offset += offset[0]
      h_offset += offset[1]

    self.combined_offset = (v_offset, h_offset)


process_steps["CombineOffsets"] = CombineOffsets
