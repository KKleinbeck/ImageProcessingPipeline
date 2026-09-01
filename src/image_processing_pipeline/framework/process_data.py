"""Implementation of the `AbstractProcessData` and `ProcessData` classes.

This module also contains the `ProcessDataSerialiser`, relevant for data serialisation at the end of the Pipeline run.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml


class AbstractProcessData(ABC):
  def __init__(self, data, name: str):
    """Create a new serialisable representation of the provided data."""
    self.data: object = data
    self.name = name

  @abstractmethod
  def _serialise(self, dir: Path) -> object:
    """Serialise the data.

    This method should either:
    - return `self.data` directly if it is already serialisable, or
    - implement a custom serialisation routine (e.g. save to a file) and return the path to the serialised data.

    Parameter
    ---------
    dir:
      Target directory for the resulting files.

    Returns
    -------
    A serialisable object or a string containing the path to the seralised data.

    """
    raise NotImplementedError("Subclasses must implement serialise method")

  def _to_yaml(self, dir: Path) -> None:
    """Write yaml file with the result of self._serialise.

    The yaml file contains:
      - data: the serialised data or path
      - type: the fully qualified type name of the original data

    Parameter
    ---------
    dir:
      Target directory for the resulting files.
    """
    serialised_data = self._serialise(dir)
    yaml_path = dir / f"{self.name}.yaml"
    with yaml_path.open("w") as f:
      yaml.safe_dump(
        {"data": serialised_data, "type": f"{type(self.data).__module__}.{type(self.data).__qualname__}"}, f
      )

  def serialise(self, dir: Path) -> None:
    """Serialise itself.

    If provided directory does not exists it will be created. Then create a yaml file that either contains the
    serialised data or a path to the actual serialised object, e.g. a tiff files when this represents an image.

    Parameter
    ---------
    dir:
      Target directory, which will contain the resulting yaml.

    Raises
    ------
    NotADirectoryError
      Provided path is not a directory.

    """
    dir.mkdir(parents=True, exist_ok=True)
    if not dir.is_dir():
      raise NotADirectoryError(f"{dir} is not a directory")

    self._to_yaml(dir)

  @staticmethod
  @abstractmethod
  def load(yaml_file: Path):
    """Load serialised data from the provided path.

    Parameter
    ---------
    yaml_file: Path to the yaml representation of the data.
    """
    raise NotImplementedError("Subclasses must implement load method")


class CollectableProcessData(AbstractProcessData):
  pass


class ProcessData(CollectableProcessData):
  def _serialise(self, dir: Path) -> object:
    return self.data

  @staticmethod
  def load(yaml_file: Path) -> object:
    """Load data from a yaml file, casting it to the stored type."""
    with yaml_file.open("r") as f:
      meta = yaml.safe_load(f)

    type_str = meta["type"]
    data = meta["data"]

    # Dynamically import type
    module_name, _, class_name = type_str.rpartition(".")
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    return cls(data)


class ProcessTiffData(AbstractProcessData):
  data: np.ndarray

  def __init__(self, data: np.ndarray, name: str):
    """Create a new serialisable representation of the provided data."""
    if not isinstance(data, np.ndarray):
      raise TypeError("ProcessTiffData expects a numpy.ndarray")
    if data.ndim not in (2, 3):
      raise ValueError("ProcessTiffData only supports 2D or 3D numpy arrays")
    super().__init__(data, name)

  def _serialise(self, dir: Path) -> str:
    """Save the numpy array as a TIFF file."""
    tif_path = dir / f"{self.name}.tif"
    if "int" in str(self.data.dtype):
      int_type = "uint8" if np.max(self.data) < 256 else "uint16"
      tiff.imwrite(tif_path, self.data.astype(int_type), photometric="minisblack")
    elif "float" in str(self.data.dtype):
      tiff.imwrite(tif_path, self.data.astype("float32"), photometric="minisblack")
    else:
      raise TypeError(
        f"Cannot serialise result {self.name} of type {self.data.dtype}. " + "Supported are float and int types."
      )
    return str(tif_path)

  @staticmethod
  def load(yaml_file: Path) -> np.ndarray:
    """Load TIFF file back into numpy array."""
    with yaml_file.open("r") as f:
      meta = yaml.safe_load(f)

    tif_path = Path(meta["data"])
    return tiff.imread(tif_path)


# --- Registry System ---


class ProcessDataSerialiser:
  _instance = None
  _registry: dict[type, type[AbstractProcessData]]

  def __new__(cls):
    """Provide the singleton instance of `ProcessDataSerialiser`."""
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._registry = {}
    return cls._instance

  def register(self, py_type: type, data_cls: type[AbstractProcessData]):
    """Register a `ProcessData` realisation for the type `py_type`."""
    self._registry[py_type] = data_cls

  def get_data_cls(self, py_type: type):
    """Get the `ProcessData` realisation registered for `py_type`."""
    return self._registry.get(py_type, ProcessData)

  def save(self, data: dict, details: dict, output_dir: Path):
    """Save entries of `data` with a suitable AbstractProcessData wrapper."""
    target_dir = output_dir / details["RelativeOutputPath"]
    target_dir.mkdir(exist_ok=True, parents=True)

    if "CollectTo" in details:
      collection = {}
      for k, v in data.items():
        wrapper_cls = self.get_data_cls(type(v))
        if issubclass(wrapper_cls, CollectableProcessData):
          collection[k] = v
        else:
          wrapper = wrapper_cls(v, k)
          wrapper.serialise(target_dir)
      collectionWrapper = ProcessData(collection, details["CollectTo"])
      collectionWrapper.serialise(target_dir)
    else:
      for k, v in data.items():
        wrapper_cls = self.get_data_cls(type(v))
        wrapper = wrapper_cls(v, k)
        wrapper.serialise(target_dir)

  def load(self, yaml_file: Path) -> AbstractProcessData:
    """Load using the ProcessData subclass stored in the yaml."""
    with yaml_file.open("r") as f:
      meta = yaml.safe_load(f)

    type_str = meta["type"]
    module_name, _, class_name = type_str.rpartition(".")
    module = __import__(module_name, fromlist=[class_name])
    data_cls = getattr(module, class_name)
    wrapper_cls = self.get_data_cls(data_cls)
    return wrapper_cls.load(yaml_file)


# --- Register standard mappings ---
process_data_serialiser = ProcessDataSerialiser()
process_data_serialiser.register(np.ndarray, ProcessTiffData)
