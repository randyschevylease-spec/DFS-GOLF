"""
base_bot.py — Abstract base class all bots inherit from.

Defines the interface every bot must implement:
  - name: str property identifying the bot
  - build_lineup(player_pool, salary_cap) -> list of players
  - Each lineup must respect salary cap and roster constraints

Bots receive a player pool (from cache) and return a valid DK lineup.
They do NOT have access to actual results — only pre-contest data.
"""

from abc import ABC, abstractmethod


class BaseBot(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Bot's display name for leaderboards."""
        ...

    @abstractmethod
    def build_lineup(self, player_pool: list[dict], salary_cap: int) -> list[dict]:
        """
        Given a player pool and salary cap, return a 6-player lineup.
        Each player dict has: dg_id, player_name, salary, ownership, projected_pts, etc.
        Returns list of 6 player dicts.
        """
        ...
