from dataclasses import dataclass

@dataclass
class Config():
    epochs :int = 5
    max_lap_k: int = 64
    threshold_on_lap_pe: float = 0.95