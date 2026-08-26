# Project Layout

.
├── src
│   └── image_processing_pipeline
│       ├── __init__.py
│       ├── _types.py
│       ├── framework
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── data_manager.py
│       │   ├── process_data.py
│       │   ├── process_pipeline.py
│       │   ├── process_step.py
│       │   ├── typed_data_interface.py
│       │   └── visualiser.py
│       └── processes
│           ├── __init__.py
│           ├── analyse_statistics.py
│           ├── apply_mask.py
│           ├── apply_morphologies.py
│           ├── arithmetic_stack_operation.py
│           ├── combine_offsets.py
│           ├── cull_boundary.py
│           ├── extract_dimensions.py
│           ├── extract_frames.py
│           ├── extract_objects.py
│           ├── extrapolate.py
│           ├── fourier_denoise.py
│           ├── generate_blob_mask.py
│           ├── generate_edge_mask.py
│           ├── generate_full_mask.py
│           ├── geometry_filter_masks.py
│           ├── interpolate.py
│           ├── invert.py
│           ├── load_stack.py
│           ├── median_filter.py
│           ├── normalise.py
│           ├── number_adder.py
│           ├── remove_outliers.py
│           ├── remove_zero_pixels.py
│           ├── shrink_to_content.py
│           ├── star_fill.py
│           ├── threshold_binarise.py
│           └── visualise_blob_mask.py
├── tests
│   ├── data
│   ├── e2e
│   │   └── framework
│   └── unit
│       ├── framework
│       └── processes

# Package Details
"src/image_processing_pipeline/framework/process_pipeline.py" is the heart and soul of the package. It defines a pipeline that runs
one `ProcessStep` after another. "src/image_processing_pipeline/processes" contains all available realisations of `ProcessStep`. They
are independent of one another and the framework as a whole.

# Test Design
Tests use pytests and shall follow common pytest usage patterns.
Each unit test must pass in under one second. They are stored in tests/unit. Each test here is decorated with `pytest.mark.unit`.
End to end tests can take as long as they want. They are stored in tests/e2e. Each test here is decorated with `pytest.mark.e2e`/
The test data layout follows the layout of the files in the src directory, i.e., each src file shall have one test file in unit and potentially e2e.
If the a test requires reference data, they shall be sourced from tests/data. Here we follow the same layout, e.g. "tests/data/unit/framework/config" for data used in "tests/unit/framework/test_config.py". 