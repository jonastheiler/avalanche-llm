"""What every model must output: the avalanche problems a bulletin describes.

The same Pydantic model validates model outputs and gives the LLMs their JSON schema.
"""

from typing import Literal

from pydantic import BaseModel, Field

ProblemType = Literal[
    "new_snow",
    "wind_slab",
    "persistent_weak_layers",
    "wet_snow",
    "gliding_snow",
    "no_distinct_avalanche_problem",
]
Aspect = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


class Problem(BaseModel):
    type: ProblemType
    danger_level: int = Field(ge=1, le=5, description="1 low, 2 moderate, ... 5 very high")
    danger_modifier: Literal["+", "-"] | None = Field(
        None, description="Swiss subdivision of the level, e.g. 2+ or 3-; null if none"
    )
    aspects: list[Aspect] = Field([], description="Slope aspects where the problem occurs")
    elevation_lower: int | None = Field(None, description="Occurs above this elevation (m)")
    elevation_upper: int | None = Field(None, description="Occurs below this elevation (m)")


class Bulletin(BaseModel):
    problems: list[Problem]
