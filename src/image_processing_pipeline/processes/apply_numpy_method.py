import numpy as np

from typing import cast

from image_processing_pipeline.framework.process_step import (
  AbstractProcessStep,
  process_steps,
)

from image_processing_pipeline._types import RegexInput, Option, RegexDeliverable


class ApplyNumpyMethod(AbstractProcessStep):
  """Apply a numpy function on the input stacks.

  This step allows the application of numpy's "universal functions" (`ufunc`s) on the input stacks.
  Depending on the expected arguments of the numpy function, the step applies the function in
  following fashion:
  - If the numpy ufunc requires one input argument, then the ufunc is applied on each input stack
    individually. For this each input stack must have one associated deliverable stack, with a
    matching index.
  - If the numpy ufunc requires more than one argument (e.g. `np.add`, `np.multiply`) then this
    step expects exactly this number of inputs. In this case the process step must also have
    exactly one deliverable, labelled `output_stack_1`.
  - Universal functions that produce more than one output stack are not supported currently.
  """

  input_stack_: RegexInput[np.ndarray, r"input_stack_\d+"]
  """Input stacks, which will be inputs to the numpy method."""

  output_stack_: RegexDeliverable[np.ndarray, r"output_stack_\d+"]
  """Results of the numpy method."""

  method: Option[str] = ""
  """The numpy method that shall be applied."""
  kwargs: Option[dict] = {}
  """Optional arguments that will be passed to the numpy method."""

  def _on_set_options(self):
    self.ufunc = getattr(np, self.method, None)
    if self.ufunc is None:
      raise TypeError(
        f"Method '{self.method}' does not exist in numpy's global scope. Make sure that 'numpy.{self.method}' exists."
      )
    if not isinstance(self.ufunc, np.ufunc):
      raise TypeError(
        f"'{self.method}' exists in numpy's global scope, but isn't a numpy.ufunc. This is a hard requirement."
      )

  def _on_verify_deliverables(self):
    self.ufunc = cast(np.ufunc, self.ufunc)
    if self.ufunc.nin == 1:
      if len(self.inputs_actual) != len(self.deliverables_actual):
        raise ValueError(
          f"Number of inputs ({len(self.inputs_actual)}) does not match number of deliverables "
          f"({len(self.deliverables_actual)})."
        )
      for input_key in self.inputs_actual:
        id = input_key.rsplit("_", 1)[1]
        expected_key = "output_stack_" + id
        if expected_key not in self.deliverables_actual:
          raise ValueError(f"Input stack {input_key} has no match output stack ({expected_key}).")
    elif self.ufunc.nout != 1:
      raise ValueError(
        f"Numpy method {self.ufunc} produces {self.ufunc.nout} outputs, current support is just for a single output."
      )
    else:
      if self.ufunc.nin != len(self.inputs_actual):
        raise ValueError(
          f"Numpy method {self.method} expects {self.ufunc.nin} inputs. Got {len(self.inputs_actual)} instead."
        )
      if len(self.deliverables_actual) != 1 or "output_stack_1" not in self.deliverables_actual:
        raise ValueError("Multi-input numpy methods require a single output stack with id 'output_stack_1'.")
    pass

  def _execute(self):
    self.ufunc = cast(np.ufunc, self.ufunc)

    if self.ufunc.nin == 1:
      self._generate_single_argument_results()
    else:
      self._generate_multi_argument_result()

  def _generate_single_argument_results(self):
    self.ufunc = cast(np.ufunc, self.ufunc)

    for input_key in self.inputs_actual:
      id = input_key.rsplit("_", 1)[1]
      output_key = "output_stack_" + id

      input = getattr(self, input_key)
      output = self.ufunc(input, **self.kwargs)
      setattr(self, output_key, output)

  def _generate_multi_argument_result(self):
    self.ufunc = cast(np.ufunc, self.ufunc)

    args = (getattr(self, input_key) for input_key in self.inputs_actual)
    self.output_stack_1 = self.ufunc(*args, **self.kwargs)


process_steps["ApplyNumpyMethod"] = ApplyNumpyMethod
