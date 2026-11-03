"""Serving cell, SINR and coverage proxies computed from per-cell RSRP."""
import numpy as np

from . import config


def link_state(rsrp_dbm: np.ndarray):
    """rsrp_dbm: (n_ue, n_cells). Full-load SINR proxy: serving vs. all other cells + noise."""
    serving = rsrp_dbm.argmax(1)
    srv_rsrp = rsrp_dbm.max(1)
    lin = 10 ** (rsrp_dbm / 10)
    s = lin[np.arange(len(lin)), serving]
    interference = lin.sum(1) - s
    sinr = 10 * np.log10(s / (interference + 10 ** (config.NOISE_DBM / 10)))
    return serving, srv_rsrp, sinr


def coverage_pct(srv_rsrp: np.ndarray) -> float:
    return float(100 * np.mean(srv_rsrp >= config.COVERAGE_RSRP_DBM))


def rlf_risk_pct(srv_rsrp: np.ndarray, sinr: np.ndarray) -> float:
    return float(100 * np.mean((srv_rsrp < config.RLF_RSRP_DBM) | (sinr < config.RLF_SINR_DB)))
