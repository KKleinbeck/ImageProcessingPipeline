import numpy as np

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import Input, Option, Deliverable

class ArithmeticStackOperation(AbstractProcessStep):
  """Apply arithmetic operation between two stacks. The two input stacks must have the same shape"""
 
  stack_a : Input[np.ndarray]
  """The first of two ndarrays"""
  stack_b : Input[np.ndarray]
  """The second of two ndarrays"""

  results_stack : Deliverable[np.ndarray]
  """ A ndarray which contains the result of the arimthetic operation performed on the two inputted ndarrays"""

  operation : Option[str] = ""
  """ The following arithemtic operations can be performed
    - add: addition (a + b)
    - subtract: subtraction (a - b)
    - multiply: multiplication ( a * b)
    - divide: division ( a / b) """

  result_stack: Deliverable[np.ndarray]

  operation: Option[str] = ""

  def _on_set_inputs(self):
    assert self.stack_a.shape == self.stack_b.shape, "Input stacks must have the same shape"

  def _on_set_options(self):
    assert self.operation in {"add", "subtract", "multiply", "divide"}, f"Unknown operation '{self.operation}'"

  def _execute(self):
    if self.operation == "add":
      self.result_stack = self.stack_a + self.stack_b
    elif self.operation == "subtract":
      self.result_stack = self.stack_a - self.stack_b
    elif self.operation == "multiply":
      self.result_stack = self.stack_a * self.stack_b
    elif self.operation == "divide":
      self.result_stack = self.stack_a / self.stack_b


process_steps["ArithmeticStackOperation"] = ArithmeticStackOperation
