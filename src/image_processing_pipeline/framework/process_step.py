"""Implementation of the abstract ProcessStep interface class."""

import re
from abc import ABC, abstractmethod
from typing import get_args
from image_processing_pipeline._types import AttributeType, attributes_match

process_steps = {}


class AbstractProcessStep(ABC):
  def __init__(
    self, inputs: dict[str, object] = {}, options: dict[str, object] = {}, delivers_id_map: dict[str, str] = {}
  ):
    """Base class for every ProcessStep.

    Validates the types of the provided inputs and options, and checks that the keys of `delivers_id_map` matches the
    defined deliverables. For regex based inputs, this generates the actually defined deliverables.
    """
    # Create copies to avoid modification of class variables
    self.inputs_actual = self._get_defined_attribute(AttributeType.Input)
    self.deliverables_actual = self._get_defined_attribute(AttributeType.Deliverable)
    self.options_actual = self._get_defined_attribute(AttributeType.Option)

    self._validate_options()

    self.delivers_id_map = delivers_id_map or {}
    self._verify_and_add(self.inputs_actual, inputs or {}, source=AttributeType.Input)
    self._on_set_inputs()
    self._verify_and_add(self.options_actual, options or {}, source=AttributeType.Option)
    self._on_set_options()
    self._verify_ids(self.deliverables_actual, set(self.delivers_id_map), source=AttributeType.Deliverable)
    self._on_verify_deliverables()

  def _on_set_inputs(self):
    """Hook for subclasses to react to inputs being set."""

  def _on_set_options(self):
    """Hook for subclasses to react to options being set."""

  def _on_verify_deliverables(self):
    """Hook for subclasses to react to deliverables being verified."""

  def execute(self) -> dict[str, object]:
    """Run the process step.

    This calls the internal `_execute` method of, then valides the produced deliverables.

    Returns
    -------
    Dictionary mapping the deliverable keys, as defined in the pipeline config, to the result.

    """
    self._execute()
    self._validate_deliverables()
    return {self.delivers_id_map[d]: getattr(self, d) for d in self.deliverables_actual if d in self.delivers_id_map}

  @abstractmethod
  def _execute(self):
    pass

  def _get_defined_attribute(self, attribute_id: AttributeType) -> dict[str, type]:
    result: dict[str, type] = {}
    for attribute, attribute_type in self._annotated_attributes.items():
      type_args = get_args(attribute_type)
      if len(type_args) >= 2 and attributes_match(type_args[1]["type"], attribute_id):
        result[attribute] = type_args[0]
    return result

  @property
  def _annotated_attributes(self) -> dict[str, object]:
    annotated_attributes = type(self).__annotations__
    for base in type(self).__bases__:
      annotated_attributes |= base.__annotations__
    return annotated_attributes

  # ============================================================
  # MARK: Validators
  def _validate_options(self) -> None:
    for option, expected_type in self.options_actual.items():
      if not hasattr(self, option):
        raise ValueError(
          f"Process {type(self).__name__} defined optional parameter {option} but did not provide a default value."
        )
      value = getattr(self, option)
      if not isinstance(value, expected_type):
        raise TypeError(
          f"Process {type(self).__name__} - parameter {option} expected type {expected_type}, but got {type(value)}."
        )

  def _verify_and_add(
    self, reference: dict[str, type], data: dict[str, object], source: AttributeType, extra_okay: bool = False
  ) -> dict[str, object]:
    """Check provided inputs against required schema and set them as attributes."""
    self._verify_ids(reference, set(data), source=source, extra_okay=extra_okay)

    # Check types and assign attributes
    for key, expected_type in reference.items():
      obj = data.get(key, getattr(self, key)) if source == AttributeType.Option else data[key]
      if not isinstance(obj, expected_type):
        raise TypeError(
          f"{type(self).__name__} argument validation failed. "
          f"{source.name} '{key}' must be of type {expected_type.__name__}, got {type(obj).__name__}."
        )
      setattr(self, key, obj)

    return {k: data[k] for k in data if k not in reference} if extra_okay else {}

  def _verify_ids(
    self, reference: dict[str, type], data_keys: set[str], source: AttributeType, extra_okay: bool = False
  ) -> None:
    """Check the keys in the reference against the data."""
    required_keys = set(reference.keys())

    missing = set[str]() if source == AttributeType.Option else required_keys - data_keys
    extra = data_keys - required_keys

    for required_key in missing.copy():
      expected_type, type_metadata = get_args(self._annotated_attributes[required_key])
      if "pattern" not in type_metadata:
        continue  # Regex Matching not possible, error is thrown later
      attribute_pattern = type_metadata["pattern"]

      for provided_key in extra.copy():
        if re.match(attribute_pattern, provided_key):
          # Register matches and update the dict / sets
          reference.pop(required_key, None)
          reference[provided_key] = expected_type
          missing.discard(required_key)  # `Discard to prevent errors
          extra.remove(provided_key)  # Explicitly catch double deletion

    # Check exact match
    msg = []
    if missing and source != AttributeType.Deliverable:  # ignore missing ids for Deliverables
      msg.append(f"Missing {source.name}: {', '.join(missing)}")
    if extra and not extra_okay:
      msg.append(f"Unexpected {source.name}: {', '.join(extra)}")
    if msg:
      raise ValueError(f"{type(self).__name__} argument validation failed. " + "; ".join(msg))

  def _validate_deliverables(self) -> None:
    """Check that deliverables exist as attributes and match expected types."""
    for key, expected_type in self.deliverables_actual.items():
      if not hasattr(self, key):
        raise AttributeError(f"Deliverable '{key}' is missing as an attribute.")
      val = getattr(self, key)
      if not isinstance(val, expected_type):
        raise TypeError(f"Deliverable '{key}' must be of type {expected_type.__name__}, got {type(val).__name__}")
