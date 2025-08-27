from dataclasses import dataclass

@dataclass
class Config:
    grid_size: tuple[int, int] = (720, 1080)
    update_frequency: float = 20

    sigma: float = 0.08
    mu: float = 0.11