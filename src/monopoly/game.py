from .elements import Player, FieldIndex, Field


class GameState:
    def __init__(self) -> None:
        self.turn = 0
        self.players: dict[str, Player] = {}
        self.mortgaged_properties: set[int] = set()

    def register_player(self, player: Player) -> None:
        self.players[player.name] = player

    def next_turn(self) -> None:
        self.turn += 1

    def mortgage_change(self, property_id: int) -> bool:
        is_mortgaged = property_id in self.mortgaged_properties
        if is_mortgaged:
            self.mortgaged_properties.remove(property_id)
        else:
            self.mortgaged_properties.add(property_id)
        return not is_mortgaged

    def get_field_owner(self, field_id: int) -> Player | None:
        for player in self.players.values():
            if field_id in player.owned_properties:
                return player
        return None


class Game:
    def __init__(self, game_state: GameState, field_index: FieldIndex) -> None:
        self.field_index = field_index
        self.state = game_state
        self.latest_events = []

    def safe_get_field_by_id(
        self,
        field_id: int,
        validate_rentable: bool = False,
        validate_payable: bool = False,
    ) -> Field:
        field = self.field_index.get_by_id(field_id)
        if field is None:
            raise ValueError(f"Field {field_id} does not exist")
        if validate_payable and not field.is_payable:
            raise ValueError(f"Field {field_id} is not payable")
        if validate_rentable and not field.is_rentable:
            raise ValueError(f"Field {field_id} is not rentable")
        return field

    def process_player_move(self, player_name: str, steps: int) -> None:
        player = self.state.players[player_name]
        player.position += steps
        if player.position >= self.field_index.size:
            self.latest_events.append(f"{player_name} passed GO, collected 2M")
            player.position -= self.field_index.size
            player.money += 2

        field = self.safe_get_field_by_id(player.position, validate_rentable=True)

        self.latest_events.append(f"{player_name} moved to {field.name}")
        owner = self.state.get_field_owner(player.position)
        if field.is_payable and not field.is_rentable and owner is None:
            player.money -= field.price
            self.latest_events.append(f"{player_name} paid {field.price} in taxes")
        elif field.is_rentable and owner is not None:
            self.latest_events.append(
                f"{player_name} paid {field.get_rent_price} to {owner.name} in rent"
            )
            player.money -= field.get_rent_price
            owner.money += field.get_rent_price

    def process_mortgage_change(self, property_id: int) -> None:
        field = self.safe_get_field_by_id(property_id, validate_rentable=True)
        player = self.state.get_field_owner(property_id)
        if player is None:
            raise ValueError(
                f"Property {property_id} is not owned so it cannot be mortgaged"
            )
        changed_to_mortgage = self.state.mortgage_change(property_id)
        money_change = field.mortgage * (1 - (not changed_to_mortgage) * 2)
        self.latest_events.append(
            f"{player.name} {'mortgaged' if changed_to_mortgage else 'unmortgaged'} {field.name} for {money_change}"
        )
        player.money += money_change

    def process_upgrade_property(self, property_id: int) -> None:
        field = self.safe_get_field_by_id(
            property_id, validate_payable=True, validate_rentable=True
        )
        player = self.state.get_field_owner(property_id)
        if player is None:
            raise ValueError(
                f"Property {property_id} is not owned so it cannot be upgraded"
            )
        if field.current_level == 5:
            raise ValueError(f"Property {property_id} is already fully upgraded")
        cost = field.upgrade_cost
        player.money -= cost
        self.latest_events.append(
            f"{player.name} upgraded {field.name} for {cost} to level {field.current_level + 1}"
        )
        field.current_level += 1

    def process_buy_property(self, player_name: str, property_id: int) -> None:
        player = self.state.players[player_name]
        field = self.safe_get_field_by_id(
            property_id, validate_payable=True, validate_rentable=True
        )
        if self.state.get_field_owner(property_id) is not None:
            raise ValueError(f"Property {property_id} is already owned")
        player.money -= field.price
        self.latest_events.append(f"{player_name} bought {field.name}")
        player.owned_properties.add(property_id)

    def send_to_jail(self, player_name: str) -> None:
        self.latest_events.append(f"{player_name} was sent to jail")
        player = self.state.players[player_name]
        player.in_jail = True
        player.position = 10

    def free_from_jail(self, player_name: str) -> None:
        self.latest_events.append(f"{player_name} was freed from jail")
        player = self.state.players[player_name]
        player.in_jail = False

    def reset_events(self) -> list[str]:
        latest = self.latest_events.copy()
        self.latest_events.clear()
        return latest
