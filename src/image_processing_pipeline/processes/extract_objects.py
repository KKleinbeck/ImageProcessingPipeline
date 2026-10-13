import numpy as np
import scipy.ndimage as nd

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, RegexDeliverable

class ExtractObjects(AbstractProcessStep):
  """Extracts pixel-separated objects from the stack.

  Individual objects are defined by regions that are separated from one another by more than one pixel within each
  frame, but overlap by at least one pixel along the stack direction.
  The amount of expected objects must be specified via the number of `object_stack_` and `offset_` deliverables
  in the pipeline config and must match the actual number of objects found.
  """

  input_stack: Input[np.ndarray]
  """A single/stack of tiff file/s"""

  object_stack_: RegexDeliverable[np.ndarray, r"object_stack_\d+"]
  """Stack of individual objects."""
  offset_: RegexDeliverable[tuple, r"offset_\d+"]
  """Offset of each object in the input_stack."""


  def _on_verify_deliverables(self):
    self.stacks = sorted({k for k in self.deliverables_actual if "stack" in k})
    self.offsets = sorted({k for k in self.deliverables_actual if "offset" in k})
    assert len(self.stacks) == len(self.offsets), (
      f"Mismatch between the number of delivered stacks ({len(self.stacks)}) and "
      + f"number of delivered stacks ({len(self.offsets)})."
    )

    for s, o in zip(self.stacks, self.offsets):
      assert s.split("_")[-1] == o.split("_")[-1], f"Mismatch of paring index. Tried to pair deliverables {s} and {o}."

  def _execute(self):
    labelled, n_labels = nd.label(self.input_stack)
    n_expected = len(self.deliverables_actual) // 2
    assert n_labels == n_expected, f"Mismatch between objects found ({n_labels}) and expected number ({n_expected})"
    ranges = nd.find_objects(labelled)

    for n, (stack, offset) in enumerate(zip(self.stacks, self.offsets)):
      setattr(self, stack, self.input_stack[:, ranges[n][1], ranges[n][2]])
      setattr(self, offset, (ranges[n][1].start, ranges[n][2].start))


process_steps["ExtractObjects"] = ExtractObjects
