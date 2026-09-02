"""Image Processing Pipeline.

Run a linear data processing pipeline, defined through a yaml-config.

Start the pipeline by instantiating `ProcessPipeline` and calling `run` on it.
Fine tune its behaviour by passing `FrameworkSettings` to it.
Visualise image results via the `Visualiser` class.
"""

from importlib.metadata import PackageNotFoundError, version

from image_processing_pipeline.framework.framework_settings import FrameworkSettings
from image_processing_pipeline.framework.process_pipeline import ProcessPipeline
from image_processing_pipeline.framework.visualiser import Visualiser

__all__ = ["FrameworkSettings", "ProcessPipeline", "Visualiser"]

try:
  __version__ = version("image_process_pipeline")
except PackageNotFoundError:
  __version__ = "unknown"
