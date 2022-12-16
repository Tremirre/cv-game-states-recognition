from dataclasses import dataclass
from typing import Iterator


@dataclass
class Family:
    name: str


@dataclass
class PropertyFamily(Family):
    house_price: float
    hotel_price: float


def deserialize_families(serialized_families: dict) -> Iterator[Family]:
    for name, data in serialized_families.items():
        if "hotel_price" in data:
            yield PropertyFamily(name=name, **data)
        yield Family(name=name)
