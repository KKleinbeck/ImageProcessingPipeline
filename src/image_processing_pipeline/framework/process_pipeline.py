"""Central entry point of the framework."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from image_processing_pipeline.framework.framework_settings import FrameworkSettings
from image_processing_pipeline.framework.data_manager import data_managers
from image_processing_pipeline.framework.process_data import ProcessDataSerialiser
from image_processing_pipeline.framework.process_step import process_steps
import image_processing_pipeline.processes as _  # Ensure all processes are registered  # noqa: F401


class ProcessPipeline(BaseModel):
  """Responsible for running the pipeline steps, validating the config and populating the data manager.

  The `ProcessPipeline` requires a config, provided through `config_path`, that denotes all relevant inputs, defines the
  order of the executed `ProcessStep`s and the defines the serialised outputs. In addition `ProcessPipeline` must
  receive the dictonary `inputs`, which defines all inputs required by the config, and `output_dir`, which defines
  the serialisation location for the config's results.

  Examples
  --------
  >>> from pathlib import Path
  >>> from image_processing_pipeline import ProcessPipeline
  >>>
  >>> pp = ProcessPipeline(
  >>>   inputs={"MyInput": 1},
  >>>   config_path=Path.cwd() / "config.yaml",
  >>>   output_dir=Path.cwd() / "outputs",
  >>> )
  >>>
  >>> pp.run()

  """

  model_config = ConfigDict(extra="allow")

  config_path: Path
  "Path to the pipeline config file (yaml)."
  output_dir: Path
  "Location of the output directory. Must not contain results of a previously successful run."
  inputs: dict[str, Any]
  "Dictionary of the required inputs for the config."

  framework_settings: FrameworkSettings = FrameworkSettings()
  "Configuration of the process pipeline."

  def __init__(self, *args, **kwargs) -> None:
    """Initialises and thoroughly checks the provided parameters and config file."""
    super().__init__(*args, **kwargs)

    if not self.output_dir.exists():
      self.output_dir.mkdir(parents=True, exist_ok=True)
    self._validate_clean_outputs()

    self.data_manager = data_managers[self.framework_settings.data_manager_type]()
    self.data_manager.register(self.inputs)

    # Load config and validate state
    self.config = self._load_config()
    self._validate_config()
    self._validate_inputs()
    self._validate_pipeline_steps()
    self.pipeline_steps = self.config["PipelineSteps"]

  def _load_config(self) -> dict:
    """Load YAML configuration file.

    Raises
    ------
    ValueError
      If the config misses a "PipelineSteps", or "Serialisations" section.
    TypeError
      If the config contains an 'Inputs' section, that is not a list.
    ValueError
      If the config contains an unknown ProcessStep.

    """
    with open(self.config_path, "r", encoding="utf-8") as f:
      try:
        config = yaml.safe_load(f)
      except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {self.config_path}: {e}")
    return config

  def run(self):
    """Run the pipeline steps in the provided pipeline config file.

    This runs all pipeline steps in the config sequentially. Upon any execution errors
    all so far available results will be stored. If the pipeline finishes successfully,
    it will serialise it's input state to the output directory, thereby freezing this
    directory and preventing the pipeline from overriding the contained data.
    """
    total_steps = len(self.pipeline_steps)
    width = self.framework_settings.execution_settings.counter_width or math.floor(math.log10(total_steps) + 1)

    try:
      for idx, step_config in enumerate(self.pipeline_steps, start=1):
        display_id = step_config["DisplayId"]
        process_name = step_config["ProcessStep"]

        # Print formatted step info
        print(f"[{idx:>{width}}/{total_steps:{width}}] Executing: {display_id}")

        # Prepare kwargs for instantiation
        kwargs = {"delivers_id_map": step_config["Deliverables"]}
        kwargs["inputs"] = {k: self.data_manager.get(v) for k, v in step_config["Inputs"].items()}
        if "Options" in step_config:
          kwargs["options"] = {
            id: self.data_manager.get(val) if isinstance(val, str) and self.data_manager.contains(val) else val
            for id, val in step_config["Options"].items()  # Read from data manager if id is present
          }

        # Instantiate and execute
        process_class = process_steps[process_name]
        current_process = process_class(**kwargs)
        deliverables = current_process.execute()
        self.data_manager.register(deliverables)
    finally:
      print("[" + (2 * width + 1) * "=" + "] Saving results")

      pds = ProcessDataSerialiser()
      serialisation_targets = self.config["Serialisations"]
      for target in serialisation_targets:
        data = {key: self.data_manager.get(key) for key in target["Data"] if self.data_manager.contains(key)}
        pds.save(data, target, self.output_dir)

    # Only on success serialise it's own state
    self.serialise()

  def serialise(self) -> None:
    """Write itself to yaml or json file at `serialisation_path`.

    Raises
    ------
    NameError
      If `serialisation_path` (defined through Frameworksettings(pipeline_settings_name=...)) does not target a json or
      yaml file.

    """
    standard_fields = set(type(self).model_fields.keys())

    self_representation = self.model_dump(include=standard_fields, mode="json")

    with open(self.serialisation_path, "w") as file:
      if self.serialisation_path.suffix == ".json":
        json.dump(self_representation, file, indent=2)
      elif self.serialisation_path.suffix == ".yaml":
        yaml.dump(self_representation, file)
      else:
        raise NameError(
          f"Don't know how to serialise ProcessPipeline to '{self.serialisation_path.suffix}'.\n"
          "\tCurrently supported is json and yaml."
        )

  @property
  def serialisation_path(self) -> Path:
    """Universal path for the serialisation result.

    Returns
    -------
    Path to the serialiased pipeline settings.

    """
    return self.output_dir / self.framework_settings.pipeline_settings_name

  # ============================================================
  # MARK: Validators
  @field_validator("config_path", mode="after")
  @staticmethod
  def _validate_config_path(config_path: Path) -> Path:
    if not config_path.exists():
      raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path

  def _validate_clean_outputs(self):
    if self.serialisation_path.exists() and self.framework_settings.prevent_override:
      raise FileExistsError(
        f"There are already valid results in {self.output_dir}.\n\t"
        "Either provide a new `output_dir` or set `framework_config.prevent_override = False`"
      )

  def _validate_config(self):
    """Validate the content of the config.

    Raises
    ------
    ValueError
      If the config misses a "PipelineSteps", or "Serialisations" section.
    TypeError
      If the config contains an 'Inputs' section, that is not a list.
    ValueError
      If the config contains an unknown ProcessStep.

    """
    required_keys = {"PipelineSteps", "Serialisations"}
    missing = required_keys - self.config.keys()
    if missing:
      raise ValueError(f"Config file {self.config_path} is missing required keys: {', '.join(missing)}")

    if not isinstance(self.config.get("Inputs", []), list):
      raise TypeError("Config field 'Inputs' must be a list of strings.")

    pipeline_steps = self.config["PipelineSteps"]
    for step_config in pipeline_steps:
      process_step_name = step_config["ProcessStep"]
      if process_step_name not in process_steps:
        raise ValueError(f"Unknown ProcessStep '{process_step_name}' in step '{step_config['DisplayId']}'")

  def _validate_inputs(self):
    """Check consistency between declared config inputs and data manager contents.

    Raises
    ------
    ValueError
      If an input listed in the config's 'Input' section was not provided.
    ValueError
      If extra inputs were provided, that aren't listed in the configs 'Inputs' section,
      This can be suppressed via the `pedantic_input_checking = False` in the `FrameworkConfig`.

    """
    declared_inputs = self.config.get("Inputs", [])
    for inp in declared_inputs:
      if not self.data_manager.contains(inp):
        raise ValueError(f"Config requires input '{inp}' which is not registered in data manager.")

    registered = set(self.data_manager.registered_results())
    declared = set(declared_inputs)
    extra = registered - declared
    if extra:
      msg = f"Data manager has inputs not declared in config: {', '.join(extra)}"
      if self.framework_settings.pedantic_input_checking:
        raise ValueError(msg)
      else:
        import warnings

        warnings.warn(msg, UserWarning)

  def _validate_pipeline_steps(self) -> None:
    """Validate the structure of the PipelineSteps entry."""

    # Helper methods
    def _validate_step(step, i: int):
      if not isinstance(step, dict):
        raise TypeError(f"Pipeline config error.\n\tStep {i} is not a dictionary.")
      missing = required_keys - step.keys()
      if missing:
        raise ValueError(f"Pipeline config error.\n\tStep {i} is missing required keys: {', '.join(missing)}")

    def _validate_inputs(inputs, display_id: str, i: int):
      if isinstance(inputs, dict) is False:
        raise ValueError(
          f"Pipeline config error.\n\tStep '{display_id}' (#{i}) has invalid 'Inputs' format. Must be a dictionary."
        )

      for input in inputs.values():
        if not dm_copy.contains(input):
          raise ValueError(
            "Pipeline config error.\n\t"
            f"Step '{display_id}' (#{i}) requires input '{input}', "
            f"which is not available in data manager."
          )

    def _validate_deliverables(deliverables, display_id: str, i: int):
      if isinstance(deliverables, dict) is False:
        raise ValueError(
          f"Pipeline config error.\n\tStep '{display_id}' (#{i}) has invalid 'Deliverables' format. Must be a dict."
        )

      try:
        dm_copy.register({k: None for k in deliverables.values()})
      except TypeError as e:
        raise ValueError(
          "Pipeline config error.\n\t"
          f"Step '{display_id}' (#{i}) tried to register a deliverable "
          f"that was already defined earlier. Details: {e}"
        )

    def _validate_pipeline_serialisation(dm_copy):
      serialisations = self.config["Serialisations"]
      if not isinstance(serialisations, list):
        raise TypeError("Pipeline config error.\n\tSerialisations must be given as a list")

      serialisation_targets = set()
      for serialisation in serialisations:
        required_keys = {"Data", "RelativeOutputPath"}
        missing_keys = required_keys - serialisation.keys()
        if missing_keys:
          raise ValueError(
            "Pipeline config error.\n\tNot every serialisation defines the fields `Data` and `RelativeOutputPath`."
          )
        serialisation_targets.update(set(serialisation["Data"]))

      for target in serialisation_targets.copy():
        if dm_copy.contains(target):
          serialisation_targets.remove(target)

      assert len(serialisation_targets) == 0, (
        f"Config tries to serialise\n\t{serialisation_targets},\nwhich aren't provided by any step."
      )

    # Start of validation
    steps = self.config["PipelineSteps"]
    if not isinstance(steps, list):
      raise TypeError("'PipelineSteps' must be a list.")

    required_keys = {"DisplayId", "ProcessStep", "Deliverables"}

    # Work on a copy of the data manager for validation purposes
    dm_copy = copy.deepcopy(self.data_manager)

    for i, step in enumerate(steps, start=1):
      _validate_step(step, i)
      display_id = step["DisplayId"]
      if "Inputs" in step:
        _validate_inputs(step["Inputs"], display_id, i)
      _validate_deliverables(step["Deliverables"], display_id, i)

    if "Serialisations" in self.config:
      _validate_pipeline_serialisation(dm_copy)
