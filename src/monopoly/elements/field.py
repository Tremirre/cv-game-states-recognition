from dataclasses import dataclass, field
from typing import Iterator

from .family import Family


@dataclass
class Field:
    id: int
    name: str
    family: Family
    placement: str
    index: int
    price: float = -1
    rent: list[float] = field(default_factory=list)
    current_level: int = 0
    mortgage: float = -1

    @property
    def is_payable(self) -> bool:
        return self.price > 0

    @property
    def is_rentable(self) -> bool:
        return self.mortgage > 0

    @property
    def upgrade_cost(self) -> float:
        if not self.family.is_upgradeable:
            return -1
        if self.current_level == 4:
            return self.family.hotel_price
        return self.family.house_price

    @property
    def get_rent_price(self) -> float:
        return self.rent[self.current_level]


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
            yield Field(
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

    def get_by_name(self, name: str) -> Field | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    @property
    def size(self) -> int:
        return len(self.fields)
