from typing import Protocol

from rl_algos.pg.extra import Episode


class Trainer(Protocol):
    def update(self, episode: Episode) -> dict[str, float]: ...
