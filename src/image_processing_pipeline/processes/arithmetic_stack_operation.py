import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, Deliverable


class ArithmeticStackOperation(AbstractProcessStep):
  stack_a: Input[np.ndarray]
  stack_b: Input[np.ndarray]

  result_stack: Deliverable[np.ndarray]

  operation: Option[str] = ""

  def _on_set_inputs(self):
    assert self.stack_a.shape == self.stack_b.shape, "Input stacks must have the same shape"

  def _on_set_options(self):
    assert self.operation in {"add", "subtract", "multiply", "divide"}, f"Unknown operation '{self.operation}'"

  def _execute(self):
    """Apply arithmetic operation between two stacks."""
    if self.operation == "add":
      self.result_stack = self.stack_a + self.stack_b
    elif self.operation == "subtract":
      self.result_stack = self.stack_a - self.stack_b
    elif self.operation == "multiply":
      self.result_stack = self.stack_a * self.stack_b
    elif self.operation == "divide":
      self.result_stack = self.stack_a / self.stack_b


process_steps["ArithmeticStackOperation"] = ArithmeticStackOperation
