"""Unit tests for validation steps in ProcessPipeline."""
import pytest
import yaml
from pathlib import Path

# Import the class to test
from image_processing_pipeline.framework.process_pipeline import ProcessPipeline


def _write_temp_yaml(content: str, tmp_path: Path) -> Path:
  """Helper to write a temporary yaml file."""
  p = tmp_path / "pipeline.yaml"
  p.write_text(content)
  return p


# ----------------------------------------------------------------------
# Helper to build minimal config needed for ProcessPipeline construction
# ----------------------------------------------------------------------
def _make_minimal_config(tmp_path: Path) -> dict:
  """Return a minimal pipeline configuration dict."""
  # Minimal valid structure that satisfies _validate_config (requires Inputs, PipelineSteps, Serialisations)
  return {
    "Inputs": [],           # can be empty list
    "PipelineSteps": [
      {
        "DisplayId": "init",
        "ProcessStep": "dummy",   # a placeholder; step existence will error later if not registered
        "Deliverables": {}      # must be dict (will be validated later)
        # No Inputs or Options needed for this minimal example
      }
    ],
    "Serialisations": [
      {
        "Data": [],           # list of data ids to serialise
        "RelativeOutputPath": "results.yaml"
      }
    ]
  }


# ----------------------------------------------------------------------
# 1. Config path validation
# ----------------------------------------------------------------------
def test_process_pipeline_raises_file_not_found_on_missing_config_path(tmp_path):
  """The validator for config_path must raise FileNotFoundError if file does not exist."""
  missing_path = tmp_path / "nonexistent.yaml"
  # inputs and output_dir are required but we can give dummy valid paths
  with pytest.raises(FileNotFoundError):
    ProcessPipeline(
      config_path=missing_path,
      output_dir=tmp_path / "output",
      inputs={},
      framework_config=None  # will be default; not needed for this test as we bypass validation via exception earlier?
    )


# ----------------------------------------------------------------------
# 2. Invalid YAML syntax raises ValueError from yaml loading
# ----------------------------------------------------------------------
def test_invalid_yaml_raises_value_error(tmp_path):
  """If the config file contains malformed YAML, a clear error is raised."""
  bad_yaml = tmp_path / "bad.yaml"
  bad_yaml.write_text("!!! This is not valid YAML")
  
  # Use a dummy inputs dict and output_dir; they are not validated yet
  pytest.raises(ValueError, lambda: ProcessPipeline(
    config_path=bad_yaml,
    output_dir=tmp_path / "output",
    inputs={},
    framework_config=None
  ))


# ----------------------------------------------------------------------
# 3. Missing required top‑level keys raises ValueError listing missing keys
# ----------------------------------------------------------------------
def test_missing_required_config_keys_raises_value_error(tmp_path):
  """If any of the three required sections are omitted, a helpful error is raised."""
  cfg = _make_minimal_config(tmp_path)
  # Remove one required key to trigger the validation error
  del cfg["Inputs"]  # This makes Inputs missing
  
  yaml_path = tmp_path / "pipeline.yaml"
  yaml_path.write_text(yaml.safe_dump(cfg))

  # Expect a ValueError mentioning missing keys
  pytest.raises(ValueError, lambda: ProcessPipeline(
    config_path=yaml_path,
    output_dir=tmp_path / "output",
    inputs={},
    framework_config=None
  ))


# ----------------------------------------------------------------------
# 4. 'Inputs' not a list raises TypeError
# ----------------------------------------------------------------------
def test_inputs_must_be_list_raises_type_error(tmp_path):
  """The config field 'Inputs' must be a list; otherwise raise TypeError."""
  cfg = _make_minimal_config(tmp_path)
  # Make Inputs a string instead of a list
  cfg["Inputs"] = "not-a-list"

  yaml_path = tmp_path / "pipeline.yaml"
  yaml_path.write_text(yaml.safe_dump(cfg))

  with pytest.raises(TypeError):
    ProcessPipeline(
      config_path=yaml_path,
      output_dir=tmp_path / "output",
      inputs={},
    )


# ----------------------------------------------------------------------
# 5. Declared input not present in data manager raises ValueError
# ----------------------------------------------------------------------
def test_declared_input_not_registered_raises_value_error(tmp_path):
  """If an Input entry cannot be resolved by the DataManager, a clear error is raised."""
  # Create a config that declares an input which will never be registered.
  cfg = _make_minimal_config(tmp_path)
  cfg["Inputs"] = ["missing_input"]

  yaml_path = tmp_path / "pipeline.yaml"
  yaml_path.write_text(yaml.safe_dump(cfg))

  # No special preparation; the validation expects the declared input to exist in the DataManager.
  pytest.raises(ValueError, lambda: ProcessPipeline(
    config_path=yaml_path,
    output_dir=tmp_path / "output",
    inputs={},
    framework_config=None
  ))


# ----------------------------------------------------------------------
# 6. Inputs format invalid (not dict) raises ValueError
# ----------------------------------------------------------------------
def test_step_inputs_must_be_dict_raises_value_error(tmp_path):
  """Step's 'Inputs' must be a dictionary; any other type raises ValueError."""
  cfg = _make_minimal_config(tmp_path)
  # Add a step with an invalid Inputs format (e.g., a string)
  cfg["PipelineSteps"][0]["Inputs"] = "invalid_input_format"

  yaml_path = tmp_path / "pipeline.yaml"
  yaml_path.write_text(yaml.safe_dump(cfg))

  pytest.raises(ValueError, lambda: ProcessPipeline(
    config_path=yaml_path,
    output_dir=tmp_path / "output",
    inputs={},
    framework_config=None
  ))


# ----------------------------------------------------------------------
# 7. Step's Deliverables format invalid raises ValueError
# ----------------------------------------------------------------------
def test_step_deliverables_must_be_dict_raises_value_error(tmp_path):
  """Step's 'Deliverables' must be a dict; otherwise raise ValueError."""
  cfg = _make_minimal_config(tmp_path)
  # Make Deliverables something other than dict, e.g., a string
  cfg["PipelineSteps"][0]["Deliverables"] = "not-a-dict"

  yaml_path = tmp_path / "pipeline.yaml"
  yaml_path.write_text(yaml.safe_dump(cfg))

  pytest.raises(ValueError, lambda: ProcessPipeline(
    config_path=yaml_path,
    output_dir=tmp_path / "output",
    inputs={},
    framework_config=None
  ))


# ----------------------------------------------------------------------
# 8. Duplicate deliverable registration raises ValueError (conflict detection)
# ----------------------------------------------------------------------
def test_duplicate_deliverable_raises_value_error(tmp_path):
  """If two steps try to register the same deliverable id, an error is raised."""
  cfg = _make_minimal_config(tmp_path)

  # Create two steps that attempt to register the same key in Deliverables
  cfg["PipelineSteps"] = [
    {
      "DisplayId": "step1",
      "ProcessStep": "dummy1",
      "Deliverables": {"out1": "some_value"}
    },
    {
      "DisplayId": "step2",
      "ProcessStep": "dummy2",
      "Deliverables": {"out1": "another_value"}  # duplicate key
    }
  ]

  yaml_path = tmp_path / "pipeline.yaml"
  yaml_path.write_text(yaml.safe_dump(cfg))

  pytest.raises(ValueError, lambda: ProcessPipeline(
    config_path=yaml_path,
    output_dir=tmp_path / "output",
    inputs={},
    framework_config=None
  ))


# ----------------------------------------------------------------------
# 9. Serialisations missing required fields raises ValueError
# ----------------------------------------------------------------------
def test_serialisation_missing_data_or_relative_output_path_raises_value_error(tmp_path):
  """Each serialisation must define `Data` and `RelativeOutputPath`."""
  cfg = _make_minimal_config(tmp_path)
  # Remove the required field from one of the serialisations
  del cfg["Serialisations"][0]["Data"]  # now missing

  yaml_path = tmp_path / "pipeline.yaml"
  yaml_path.write_text(yaml.safe_dump(cfg))

  pytest.raises(ValueError, lambda: ProcessPipeline(
    config_path=yaml_path,
    output_dir=tmp_path / "output",
    inputs={},
    framework_config=None
  ))