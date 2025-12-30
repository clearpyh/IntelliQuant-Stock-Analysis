from dataclasses import dataclass
from typing import List

@dataclass
class IndicatorsFrame:
    index: List[str]
    columns: List[str]
    data: List[List[float]]

@dataclass
class ADXSeries:
    index: List[str]
    values: List[float]
