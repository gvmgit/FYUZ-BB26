"""The hidden faults a teammate can plant in a blind test."""
from dataclasses import dataclass

from .topology import Settings


@dataclass(frozen=True)
class Fault:
    kind: str    # "tilt" | "azimuth" | "cell_off"
    cell: int
    value: float = 0.0  # tilt degrees, or azimuth degrees; unused for cell_off

    def apply(self, settings: Settings) -> Settings:
        out = {c: dict(s) for c, s in settings.items()}
        if self.kind == "tilt":
            out[self.cell]["tilt"] = self.value
        elif self.kind == "azimuth":
            out[self.cell]["azimuth"] = self.value % 360
        elif self.kind == "cell_off":
            out[self.cell]["enabled"] = False
        else:
            raise ValueError(f"unknown fault kind {self.kind}")
        return out
