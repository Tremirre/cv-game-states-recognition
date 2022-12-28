from dataclasses import dataclass, field


@dataclass
class Player:
    name: str
    money: float = 15
    position: int = 0
    in_jail: bool = False
    owned_properties: set[int] = field(default_factory=set)
