import pytest
import yaml

from image_processing_pipeline import ProcessPipeline, FrameworkSettings


BASIC_EXAMPLE_CONFIG: dict = {
  "Inputs": ["MyNumber"],
  "PipelineSteps": [
    {
      "DisplayId": "Offset by 3",
      "ProcessStep": "NumberAdder",
      "Inputs": {"number_1": "MyNumber"},
      "Options": {"extra_summand": 3},
      "Deliverables": {"sum": "MyNumberPlus3"},
    }
  ],
  "Serialisations": [{"RelativeOutputPath": ".", "Data": ["MyNumberPlus3"]}],
}


@pytest.fixture(scope="function")
def working_directory(tmp_path):
  """Setup a working directory with a prefilled config."""
  config_path = tmp_path / "config.yaml"
  with open(config_path, "w") as fp:
    yaml.dump(BASIC_EXAMPLE_CONFIG, fp)

  return tmp_path


@pytest.mark.e2e
def test_basic_example(working_directory):
  config_path = working_directory / "config.yaml"
  output_path = working_directory / "outputs"

  pp = ProcessPipeline(inputs={"MyNumber": 1}, config_path=config_path, output_dir=output_path)
  pp.run()

  output_contents = set(file.name for file in output_path.iterdir())
  assert output_contents == set(("pipeline_settings.yaml", "MyNumberPlus3.yaml"))

  with open(output_path / "MyNumberPlus3.yaml") as fp:
    my_number_result = yaml.safe_load(fp)

  assert "data" in my_number_result
  assert my_number_result["data"] == 4


@pytest.mark.e2e
def test_custom_settings(working_directory):
  config_path = working_directory / "config.yaml"
  output_path = working_directory / "outputs"

  framework_settings = FrameworkSettings(pipeline_settings_name="custom_name.yaml")
  pp = ProcessPipeline(
    inputs={"MyNumber": 1}, config_path=config_path, output_dir=output_path, framework_settings=framework_settings
  )
  pp.run()

  output_contents = set(file.name for file in output_path.iterdir())
  assert output_contents == set(("custom_name.yaml", "MyNumberPlus3.yaml"))

  with open(output_path / "custom_name.yaml") as fp:
    serialised_framework_settings = yaml.safe_load(fp)

  assert serialised_framework_settings["inputs"] == {"MyNumber": 1}
  assert serialised_framework_settings["framework_settings"] == framework_settings.model_dump()


@pytest.mark.e2e
def test_override_protection(working_directory):
  config_path = working_directory / "config.yaml"
  output_path = working_directory / "outputs"

  # Run 1
  pp = ProcessPipeline(
    inputs={"MyNumber": 1},
    config_path=config_path,
    output_dir=output_path,
  )
  pp.run()

  # Run 2 - protected
  with pytest.raises(FileExistsError):
    ProcessPipeline(
      inputs={"MyNumber": 1},
      config_path=config_path,
      output_dir=output_path,
    )

  # Run 2 - unprotected
  framework_settings = FrameworkSettings(prevent_override=False)
  ProcessPipeline(
    inputs={"MyNumber": 1}, config_path=config_path, output_dir=output_path, framework_settings=framework_settings
  )
