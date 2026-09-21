import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option


class ApplyMorphologies(AbstractProcessStep):
  input_stack: Input[np.ndarray]
  "A stack of arbitrary images."

  morphed_stack: Deliverable[np.ndarray]
  "Stack after transformation"

  strategy: Option[dict] = {"binary_erosion": {"iterations": 1}}
  """Strategies to apply. The step executes all provided strategies in order.

  Supported strategies are binary_erosion, binary_dilation, binary_opnening, and binary_closing;
  see `scipy.ndimage`.
  """

  def _on_set_options(self):
    supported_strategies = ["binary_erosion", "binary_dilation", "binary_opnening", "binary_closing"]
    for name in self.strategy.keys():
      if name not in supported_strategies:
        raise KeyError(
          f"Cannot apply strategy '{name}', supported is {','.join(str(s) for s in supported_strategies)}."
        )

  def _execute(self):
    """Apply morphological operations to the input stack according to the specified strategy."""
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
