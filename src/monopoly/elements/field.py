from dataclasses import dataclass
from typing import Iterator

from .family import Family


@dataclass
class Field:
    id: int
    name: str
    family: Family
    placement: str
    index: int


@dataclass
class PricedField(Field):
    price: float


@dataclass
class RentableField(PricedField):
    rent: list[float]
    mortgage: float


def create_field(**kwargs) -> Field:
    if "rent" in kwargs:
        return RentableField(**kwargs)
    if "price" in kwargs:
        return PricedField(**kwargs)
    return Field(**kwargs)


def deserialize_fields(
    serialized_fields: dict, families: dict[str, Family]
) -> Iterator[Field]:
    for field_placement, field_data in serialized_fields["corners"].items():
        yield Field(
            id=field_data["id"],
            name=field_data["name"],
            family=families["corner"],
            placement=field_placement,
            index=0,
        )
    for section in ["top", "bottom", "left", "right"]:
        for field_index, field_data in enumerate(serialized_fields[section]):
            field_family = field_data["family"]
            yield create_field(
                **{
                    **field_data,
                    "family": families[field_family],
                    "placement": section,
                    "index": field_index,
                }
            )


class FieldIndex:
    def __init__(self, fields: list[Field]) -> None:
        self.fields = sorted(fields, key=lambda field: field.id)
        self.fields_by_place = {
            (field.placement, field.index): field for field in self.fields
        }

    def get_by_id(self, id: int) -> Field | None:
        if id < 0 or id >= len(self.fields):
            return None
        return self.fields[id]

    def get_by_place(self, placement: str, index: int = 0) -> Field | None:
        return self.fields_by_place.get((placement, index))
