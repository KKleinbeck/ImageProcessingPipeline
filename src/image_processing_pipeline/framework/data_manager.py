"""Store and provide data that is generated and consumed during the pipeline run."""

import copy

data_managers = {}


class DataManager:
  """Native Data Manager.

  This is the template for all new `DataManger` realisations as well as the native data manager, that is used
  by default. Data manager store and provide the data that is produced and consumed during a pipeline run.
  At the each new `ProcessStep` the `ProcessPipeline` will request all relevant input data from the data manager
  and after the execution will register, i.e., store, the deliverables of the `ProcessStep`.

  This data manager stores all results in RAM, therefore data aquisition is generally quick but the amount of data
  that can stored is hardware limited.
  """

  def __init__(self):
    """Initialise an empty DataManager."""
    self._results = {}

  def contains(self, id: str) -> bool:
    """Return whether provided id exisits in registry.

    Parameters
    ----------
    id:
      ID / name that is searched in the registry.

    Returns
    -------
    True if id is in registry, False otherwise.

    """
    return id in self._results

  def get(self, id: str) -> object:
    """Return data stored under the provided ID.

    Parameters
    ----------
    id:
      ID / name that is searched in the registry.

    Returns
    -------
    The object stored under said ID.

    Raises
    ------
    KeyError:
      If there is no object in the regirsty with the provided ID.

    """
    if not self.contains(id):
      raise KeyError(f"Data with id {id} not found.")
    return copy.deepcopy(self._results[id])

  def registered_results(self) -> list[str]:
    """Provide the IDs of all registered objects.

    Returns
    -------
    A list of all IDs in the registry.

    """
    return list(self._results.keys())

  def register(self, id: str | dict[str, object], data: object | None = None) -> None:
    """Add new data to registry.

    This has to operational mods, depending whether `id` is a single string or a dictionary
    of string values. If `id` is a string, then store `data` under the provided string ID.
    If `id` is a dictionary, store every value under the provided key.

    Parameters
    ----------
    id:
      ID / name under which the data is stored or a dictonary of IDs and data.
    data:
      Object to be stored.

    Raises
    ------
    ValueError
      If `id` is a dict and the data is not None.
    TypeError
      If `id` is neither a string or dict.

    """
    if isinstance(id, str):
      self._register_individual(id, data)
    elif isinstance(id, dict):
      if data is not None:
        raise ValueError(
          "Tried to register multiple objects by passing a dict to `ids`, but also provided some additional data."
        )
      self._register_bulk(id)
    else:
      raise TypeError(f"id must be a string or a dict. Instead got {type(id)}.")

  def _register_bulk(self, data_dict: dict[str, object]) -> None:
    """Register a bulk of data in dictionary form all at once.

    Registeres the values under their string keys.

    Parameters
    ----------
    data_dict:
      Dictionary of the data being registered.

    Raises
    ------
    TypeError:
      If dictionary keys are not strings.

    """
    for id in data_dict:
      if isinstance(id, str) is False:
        raise TypeError("All keys in data_dict must be strings.")
    for id, data in data_dict.items():
      self._register_individual(id, data)

  def _register_individual(self, id: str, data: object) -> None:
    """Register a new object under the provided ID.

    Placeholder IDs (underscore strings "_") will be silently ignored.

    New realisations of `DataManager` that do not which to store the data in RAM shall
    override this method.

    Parameters
    ----------
    id:
      ID / name under which the data is stored or a dictonary of IDs and data
    data:
      Object to be stored.

    Raises
    ------
    KeyError
      If the registry already contains the an entry under the provided ID.

    """
    if id == "_":
      return  # Ignore placeholder
    if self.contains(id):
      raise KeyError(f"Data with id {id} already exists.")
    self._results[id] = data


data_managers["native"] = DataManager
