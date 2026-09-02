import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Deliverable, Option

class GenerateEdgeMask(AbstractProcessStep):
  """Generates an edge mask.
  
  The masks contains True where an edge is detected, False otherwise.
  Parameter sigma defines the kernel width used for the Gaussian Laplace filter.
  """
  input_stack: Input[np.ndarray]
  "A single/stack of tiff image/s"
  
  edge_mask: Deliverable[np.ndarray]
  "A 0/1 mask. The mask contains 1 when an edge is detected and 0 otherwise"
  
  sigma: Option[float] = 10.0
  """Kernel width used for the Gaussian Laplace filter"""
  
  def _on_set_options(self):
    assert self.sigma > 0, "Sigma must be positive."

  def _execute(self):
    res = nd.gaussian_laplace(self.input_stack, self.sigma, axes=(1, 2))
    self.edge_mask = res < 0


process_steps["GenerateEdgeMask"] = GenerateEdgeMask
