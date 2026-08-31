"""Unit tests for the AbstractProcessStep base class in image_processing_pipeline.framework.process_step."""

import pytest

from image_processing_pipeline.framework.process_step import AbstractProcessStep

from image_processing_pipeline._types import Input, Option, Deliverable, RegexInput, RegexDeliverable


class EmptyProcessStep(AbstractProcessStep):
  def _execute(self):
    pass


@pytest.mark.unit
def test_input_errors():
  class TestProcessStep(EmptyProcessStep):
    test_input: Input[int]

  with pytest.raises(ValueError, match="Missing Input: test_input"):
    TestProcessStep()

  with pytest.raises(TypeError, match="Input .* must be of type int, got str."):
    TestProcessStep(inputs={"test_input": "str_instead_of_int"})

  with pytest.raises(ValueError, match="Unexpected Input: unknown_input"):
    EmptyProcessStep(inputs={"unknown_input": 0})


@pytest.mark.unit
def test_option():
  class TestProcessStep(EmptyProcessStep):
    test_option: Option[int] = 0

  with pytest.raises(TypeError, match="Option .* must be of type int, got str."):
    TestProcessStep(options={"test_option": "str_instead_of_int"})

  with pytest.raises(ValueError, match="Unexpected Option: test_extra"):
    TestProcessStep(options={"test_extra": 1})

  step = TestProcessStep()
  assert getattr(step, "test_option") == 0

  step = TestProcessStep(options={"test_option": 1})
  assert getattr(step, "test_option") == 1


@pytest.mark.unit
def test_deliverable_missing_raises_attributeerror():
  class TestProcessStep(EmptyProcessStep):
    test_deliverable: Deliverable[int]

  with pytest.raises(ValueError, match="Missing Deliverable: test_deliverable"):
    TestProcessStep()


@pytest.mark.unit
def test_deliverable_type_mismatch_raises_typeerror():
  class TestProcessStep(EmptyProcessStep):
    test_deliverable: Deliverable[int]

    def _execute(self):
      self.test_deliverable = "str_instead_of_int"  # ty: ignore[invalid-assignment]

  with pytest.raises(TypeError, match="Deliverable .* must be of type int, got str"):
    step = TestProcessStep(delivers_id_map={"test_deliverable": "_"})
    step.execute()


@pytest.mark.unit
def test_deliverable_delivers_correctly():
  class TestProcessStep(EmptyProcessStep):
    test_deliverable: Deliverable[int]

    def _execute(self):
      self.test_deliverable = 1

  step = TestProcessStep(delivers_id_map={"test_deliverable": "result_deliverable"})
  assert step.execute() == {"result_deliverable": 1}


@pytest.mark.unit
def test_regex_input():
  class TestProcessStep(EmptyProcessStep):
    test_regex_input: RegexInput[int, r"regex_input_\d+"]

  with pytest.raises(ValueError, match="Missing Input: test_regex_input"):
    TestProcessStep(inputs={"bad_input": 0})

  step = TestProcessStep(inputs={"regex_input_1": 0})
  assert "regex_input_1" in step.inputs_actual


@pytest.mark.unit
def test_regex_deliverable_pattern_match():
  class TestProcessStep(EmptyProcessStep):
    test_deliverable: RegexDeliverable[int, r"regex_deliverable_\d+"]

  step = TestProcessStep(delivers_id_map={"regex_deliverable_1": "regex_deliverable_1"})
  assert "regex_deliverable_1" in step.deliverables_actual


@pytest.mark.unit
def test_hooks_called_during_initialisation():
  class TestProcessStep(EmptyProcessStep):
    def __init__(self, *args, **kwargs):
      self.on_set_inputs_called = 0
      self.on_set_options_called = 0
      self.on_verify_deliverables_called = 0
      super().__init__(*args, **kwargs)

    def _on_set_inputs(self):
      self.on_set_inputs_called += 1

    def _on_set_options(self):
      self.on_set_options_called += 1

    def _on_verify_deliverables(self):
      self.on_verify_deliverables_called += 1

  step = TestProcessStep()

  assert step.on_set_inputs_called == 1
  assert step.on_set_options_called == 1
  assert step.on_verify_deliverables_called == 1
