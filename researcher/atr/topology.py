"""Network inventory and sample-data helpers. No MAVERIC/torch imports, so the agent runtime stays light."""
from typing import Dict

import pandas as pd

from . import config

Settings = Dict[int, dict]  # cell_id -> {"tilt": deg, "azimuth": deg, "enabled": bool}


def _sim_dirs() -> Dict[int, "config.Path"]:
    out = {}
    for d in sorted(config.SIM_DATA.glob("sim_*")):
        sc = pd.read_csv(d / "site_config.csv")
        out[int(sc.cell_el_deg.iloc[0])] = d
    return out


def load_topology() -> pd.DataFrame:
    d = next(iter(_sim_dirs().values()))
    sc = pd.read_csv(d / "site_config.csv")
    return sc[["cell_id", "cell_lat", "cell_lon", "cell_az_deg", "cell_carrier_freq_mhz"]].reset_index(drop=True)


def default_inventory(tilt: float = 10.0) -> Settings:
    """The documented network: every cell at `tilt`, azimuth as in the topology, all on. Needs no model files."""
    topo = load_topology().set_index("cell_id")
    return {int(c): {"tilt": float(tilt), "azimuth": float(topo.loc[c].cell_az_deg), "enabled": True}
            for c in topo.index}


def load_grid() -> pd.DataFrame:
    d = next(iter(_sim_dirs().values()))
    f = pd.read_csv(d / "full_data.csv", usecols=["loc_x", "loc_y", "cell_id"])
    cid = f.cell_id.min()
    return f[f.cell_id == cid][["loc_x", "loc_y"]].reset_index(drop=True)
