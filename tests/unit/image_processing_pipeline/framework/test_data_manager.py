"""Unit tests for the DataManager class in image_processing_pipeline.framework."""

import pytest

# Import the class directly from the installed package / source tree.
from image_processing_pipeline.framework.data_manager import DataManager


@pytest.fixture
def empty_manager():
  """Provide a fresh DataManager instance for each test."""
  return DataManager()


@pytest.mark.unit
def test_init_creates_empty_registry(empty_manager):
  """A new DataManager should start with no registered results."""
  assert not empty_manager.registered_results()
  # All look-ups should miss initially.
  assert not empty_manager.contains("any-id")


@pytest.mark.unit
def test_register_single_id_success(empty_manager):
  """Register a single piece of data under a string key - should succeed."""
  test_data = {"value": 42, "list": [1, 2, 3]}
  empty_manager.register("my-key", test_data)

  # The ID should now be present.
  assert empty_manager.contains("my-key")
  # Retrieval via get() should return a deepcopy of the original data.
  retrieved = empty_manager.get("my-key")
  assert retrieved == test_data
  # The returned copy must be independent – mutating it does not affect stored data.
  result_copy = retrieved  # keep reference for clarity
  result_copy["new_key"] = "added"
  # Access again – original should remain unchanged.
  still_stored = empty_manager.get("my-key")
  assert "new_key" not in still_stored


@pytest.mark.unit
def test_register_bulk_success(empty_manager):
  """Register multiple items via a dictionary using the bulk mode."""
  bulk_data = {
    "first": 1,
    "second": {"inner": "value"},
    "third": [0],
  }
  # Using dict + no extra data argument should register each entry.
  empty_manager.register(bulk_data, None)

  for key in bulk_data:
    assert empty_manager.contains(key)
    assert empty_manager.get(key) == bulk_data[key]


@pytest.mark.unit
def test_bulk_registration_with_explicit_data_raises(empty_manager):
  """Providing both a dict and extra data should raise ValueError."""
  with pytest.raises(ValueError, match="Tried to register multiple objects"):
    empty_manager.register({"a": 1}, "extra")


@pytest.mark.unit
def test_invalid_id_type_raises_typeerror(empty_manager):
  """register() must reject non‑string / non‑dict id arguments."""
  with pytest.raises(TypeError, match="id must be a string or a dict"):
    empty_manager.register(123, "data")

  with pytest.raises(TypeError, match="id must be a string or a dict"):
    empty_manager.register(None, {})


@pytest.mark.unit
def test_register_duplicate_key_raises_keyerror(empty_manager):
  """Attempting to store data under an already‑used ID should raise KeyError."""
  empty_manager.register("dup-key", "initial")
  with pytest.raises(KeyError, match="already exists"):
    empty_manager.register("dup-key", "second")


@pytest.mark.unit
def test_register_placeholder_id_is_ignored(empty_manager):
  """IDs that are exactly '_' must be silently ignored."""
  # Registering with a placeholder should not add anything.
  empty_manager.register("_", {"ignore_me": 1})
  # The placeholder must still not appear in the registry.
  assert "_ " not in ("_",) and not empty_manager.contains("_")


@pytest.mark.unit
def test_get_raises_keyerror_for_missing_id(empty_manager):
  """Calling get() on an unknown ID should raise KeyError with a clear message."""
  with pytest.raises(KeyError, match="Data with id missing-id not found."):
    empty_manager.get("missing-id")


@pytest.mark.unit
def test_registered_results_returns_all_keys(empty_manager):
  """registered_results() must list all IDs currently stored."""
  empty_manager.register("a", 1)
  empty_manager.register("b_c", {"x": "y"})
  # Order is not guaranteed, but the set of IDs must match.
  assert set(empty_manager.registered_results()) == {"a", "b_c"}

  # Adding a new entry should increase the list length.
  empty_manager.register("d", [5])
  assert len(empty_manager.registered_results()) == 3


@pytest.mark.unit
def test_get_returns_deepcopy_not_same_object(empty_manager):
  """The object returned by get must be a deep copy, not the original stored reference."""
  original_list = [1, 2, {"inner": "obj"}]
  empty_manager.register("deep", original_list)

  retrieved = empty_manager.get("deep")
  # Mutate the mutable elements of the retrieved copy.
  retrieved[0] = 99
  retrieved[-1]["inner"] = "changed"

  # The stored data must remain unchanged when accessed again.
  still_stored = empty_manager.get("deep")
  assert still_stored == [1, 2, {"inner": "obj"}]
  assert original_list[0] == 1


@pytest.mark.unit
def test_bulk_registration_keys_must_be_strings(empty_manager):
  """All keys in the bulk dict must be strings; otherwise a TypeError is raised."""
  mixed_dict = {
    "valid_key": 1,
    42: "numeric_key",  # <-- invalid
    ("tuple",): "also_invalid",
  }
  with pytest.raises(TypeError, match="must be strings"):
    empty_manager.register(mixed_dict, None)
