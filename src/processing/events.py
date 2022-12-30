import abc

import numpy as np

from .context import ContextReader


@abc.abstractmethod
class EventDetector(abc.ABC):
    def __init__(self, tracked_readers: list[ContextReader]):
        self.tracked_readers = {reader.name: reader for reader in tracked_readers}

    def detect(self) -> bool:
        ...

    def get_message(self) -> str:
        ...


class DiceEventDetector(EventDetector):
    def __init__(self, tracked_readers: list[ContextReader]):
        super().__init__(tracked_readers)
        self.last_throw = None
        self.last_pos = None
        self.last_non_zero_pos = None
        self.cooldown = -1

    def detect(self) -> bool:
        self.cooldown -= 1
        dice_dots_reader = self.tracked_readers["dice_dots"]
        dice_pos_x_reader = self.tracked_readers["dice_pos_x"]
        last_pos = self.last_pos
        cur_pos = dice_pos_x_reader.get_value()
        if (
            last_pos == 0
            and cur_pos != 0
            and (
                self.last_non_zero_pos is None
                or np.abs(cur_pos - self.last_non_zero_pos) > 1
            )
        ):
            self.cooldown = 25
        if cur_pos != 0:
            self.last_non_zero_pos = cur_pos
        self.last_pos = cur_pos

        self.last_throw = dice_dots_reader.get_value()
        return self.cooldown == 0

    def get_message(self) -> str:
        if self.last_throw is None or self.last_throw < 2:
            self.cooldown = 3
            return ""
        print("DICE EVENT")
        return f"Dice throw detected - rolled {self.last_throw}"


class CardEventDetector(EventDetector):
    def __init__(self, tracked_readers: list[ContextReader], player: str):
        super().__init__(tracked_readers)
        self.cards = 0
        self.player = player
        self.property = ""

    def detect(self) -> bool:
        detected_cards = self.tracked_readers[f"{self.player}_cards"].get_value()
        if detected_cards > self.cards:
            self.cards = detected_cards
            self.property = self.tracked_readers[f"{self.player}_pos"].get_value()
            return True
        return False

    def get_message(self) -> str:
        print("PROPERTY EVENT")
        return f"{self.player.capitalize()} bought new property ({self.property})"


class MoveEventDetector(EventDetector):
    def __init__(self, tracked_readers: list[ContextReader], player: str):
        super().__init__(tracked_readers)
        self.player = player
        self.last_position = None

    def detect(self) -> bool:
        position = self.tracked_readers[f"{self.player}_pos"].get_value()
        if self.last_position is None:
            self.last_position = position
            return False
        if position != self.last_position:
            self.last_position = position
            return True
        return False

    def get_message(self) -> str:
        print("MOVE EVENT")
        msg = f"{self.player.capitalize()} moved to {self.last_position}"
        if self.last_position == "Go to Prison":
            msg += f" - {self.player.capitalize()} goes to prison"
        elif self.last_position == "Community Chest":
            msg += f" - {self.player.capitalize()} draws a community chest card"
        elif self.last_position == "Chance":
            msg += f" - {self.player.capitalize()} draws a chance card"
        elif self.last_position == "Income Tax":
            msg += f" - {self.player.capitalize()} pays income tax"
        elif self.last_position == "Super Tax":
            msg += f" - {self.player.capitalize()} pays super tax"
        return msg
