from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AnalysisResult:
    timestamp: str
    data: Dict[str, Any]
