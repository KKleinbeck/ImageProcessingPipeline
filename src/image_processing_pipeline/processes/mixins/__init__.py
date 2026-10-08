"""Module for shared functionalities across multiple process steps."""

from image_processing_pipeline.processes.mixins.culling import CullingMixin
from image_processing_pipeline.processes.mixins.masked_input import MaskedInputMixin

__all__ = ["CullingMixin", "MaskedInputMixin"]
