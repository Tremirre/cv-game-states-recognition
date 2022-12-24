from dataclasses import dataclass
from typing import Iterator


@dataclass
class Family:
    name: str
    house_price: float = -1
    hotel_price: float = -1

    @property
    def is_upgradeable(self) -> bool:
        return self.house_price > 0 and self.hotel_price > 0


def deserialize_families(serialized_families: dict) -> Iterator[Family]:
    for name, data in serialized_families.items():
        yield Family(name=name, **data)
