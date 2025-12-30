from dataclasses import dataclass
from typing import Optional

@dataclass
class ModuleStatus:
    status: str
    time: Optional[str]
    fresh: Optional[bool]
