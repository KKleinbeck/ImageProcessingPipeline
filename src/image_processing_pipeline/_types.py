from typing import Annotated, TypeAlias, TypeVar, cast
from enum import Enum

T = TypeVar("T")

class AttributeType(Enum):
  Input            = 0b0001
  Option           = 0b0010
  Deliverable      = 0b0100

  RegexInput       = 0b1001
  RegexDeliverable = 0b1100

def attributes_match(lhs: AttributeType, rhs: AttributeType) -> bool:
  return lhs.value >= rhs.value and bool(lhs.value & rhs.value)

Input: TypeAlias = Annotated[T, {"type": AttributeType.Input}]
Option: TypeAlias = Annotated[T, {"type": AttributeType.Option}]
Deliverable: TypeAlias = Annotated[T, {"type": AttributeType.Deliverable}]

class RegexInput[T, U]:
  def __class_getitem__(cls, params: tuple[type, str]) -> T:
    if len(params) != 2:
      raise IndexError(
        "RegexInput types must be called with two arguments: a type and a regex pattern.\n\t"
        f"Instead {len(params)} arguments were passed."
      )
    t, u = params
    if not isinstance(t, type):
      TypeError("RegexInput's first argument must be a valid type.")
    if not isinstance(u, str):
      TypeError("RegexInput's second argument must be a regex pattern.")
    return cast(T, Annotated[t, {"type": AttributeType.RegexInput, "pattern": u}])  # ty: ignore[invalid-type-form]

class RegexDeliverable[T, U]:
  def __class_getitem__(cls, params: tuple[type, str]) -> T:
    if len(params) != 2:
      raise IndexError(
        "RegexDeliverable types must be called with two arguments: a type and a regex pattern.\n\t"
        f"Instead {len(params)} arguments were passed."
      )
    t, u = params
    if not isinstance(t, type):
      TypeError("RegexDeliverable's first argument must be a valid type.")
    if not isinstance(u, str):
      TypeError("RegexDeliverable's second argument must be a regex pattern.")
    return cast(T, Annotated[t, {"type": AttributeType.RegexDeliverable, "pattern": u}])  # ty: ignore[invalid-type-form]
