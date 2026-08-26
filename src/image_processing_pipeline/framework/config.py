"""Config class for the runtime and validation behaviour of `ProcessPipeline`."""

from pydantic import BaseModel, field_validator

from .data_manager import data_managers


class ExecutionSettings(BaseModel):
  counter_width: int | None = None
  "Display width for the process step counter. `None` enables automatic scaling"


class FrameworkConfig(BaseModel):
  data_manager_type: str = "native"
  "Any data manager registered under `image_processing_pipeline.data_manager.data_managers`"
  execution_settings: ExecutionSettings = ExecutionSettings()
  pedantic_input_checking: bool = True
  "Check pipeline config inputs thorough. If `True` do not allow extra inputs."
  pipeline_settings_name: str = "pipeline_settings.yaml"
  "File name for serialised state of the pipeline and it's config."
  prevent_override: bool = True
  "Prevent overriding previously successful results."


  @field_validator('data_manager_type', mode='after')  
  @staticmethod
  def _validate_data_manager_type(data_manager_type: str) -> str:
    if data_manager_type not in data_managers:
      raise KeyError(
        f"Unknown data manager '{data_manager_type}'. Available: {', '.join(data_managers.keys())}"
      )
    return data_manager_type
  

  @field_validator('pipeline_settings_name', mode='after')  
  @staticmethod
  def validate_pipeline_settings_name(pipeline_settings_name: str) -> str:
    splits = pipeline_settings_name.rsplit(".", 1)
    if len(splits) == 1:
      return pipeline_settings_name + ".yaml"
    elif splits[1] not in ["json", "yaml"]:
      raise ValueError(
        "Framework Config Error.\n\t"
        f"Cannot serialise pipeline to {splits[1]}. Supported is 'json', 'yaml'."
      )
    return pipeline_settings_name
