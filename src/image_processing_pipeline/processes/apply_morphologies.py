import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, Deliverable

class ApplyMorphologies(AbstractProcessStep):
  """Apply morphological operations to the input stack according to the specified strategy."""  

  input_stack: Input[np.ndarray]
  """A ndarray containing a 0/1 mask"""

  morphed_stack : Deliverable[np.ndarray] 
  """A ndarray containing a modified 0/1 mask that has undergone a morpholigcal operation."""
 
  strategy : Option[dict, "binary_erosion": {"iterations": 1}]
  """The following options are available for morphological operartions. For further details please see SciPy documentation
    - binary_erosion
    - binary_dilation
    - binary_opening
    - binary_closing
  """

  def _execute(self):
    for name, params in self.strategy.items():
      if name == "binary_erosion":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_erosion(self.input_stack, iterations=iterations, axes=(1, 2)).astype(
          self.input_stack.dtype
        )
      elif name == "binary_dilation":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_dilation(self.input_stack, iterations=iterations, axes=(1, 2)).astype(
          self.input_stack.dtype
        )
      elif name == "binary_opening":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_opening(self.input_stack, iterations=iterations, axes=(1, 2)).astype(
          self.input_stack.dtype
        )
      elif name == "binary_closing":
        iterations = params.get("iterations", 1)
        self.input_stack = nd.binary_closing(self.input_stack, iterations=iterations, axes=(1, 2)).astype(
          self.input_stack.dtype
        )
    self.morphed_stack = self.input_stack


process_steps["ApplyMorphologies"] = ApplyMorphologies
