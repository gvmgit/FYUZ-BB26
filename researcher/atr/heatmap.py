"""Best-server RSRP heatmaps (the image the multimodal model reads alongside KPI tables)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_filmstrip(twin, panels, path, suptitle: str = ""):
    """panels: [(title, mean_rsrp[n_points, n_cells])] drawn side by side with one shared colour scale."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4), dpi=110, sharey=True)
    for ax, (title, mean) in zip(np.atleast_1d(axes), panels):
        sc = ax.scatter(twin.grid.loc_x, twin.grid.loc_y, c=mean.max(1), s=4, cmap="RdYlGn", vmin=-125, vmax=-85)
        for cid in twin.cells:
            t = twin.topology.loc[cid]
            ax.plot(t.cell_lon, t.cell_lat, "k^", ms=5)
            ax.annotate(str(cid), (t.cell_lon, t.cell_lat), textcoords="offset points", xytext=(3, 3), fontsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
    fig.colorbar(sc, ax=axes, label="best-server RSRP (dBm)", shrink=0.8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_best_server(twin, mean_rsrp: np.ndarray, path, title: str):
    best = mean_rsrp.max(1)
    serving = mean_rsrp.argmax(1)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    sc = ax.scatter(twin.grid.loc_x, twin.grid.loc_y, c=best, s=7, cmap="RdYlGn", vmin=-125, vmax=-85)
    fig.colorbar(sc, ax=ax, label="best-server RSRP (dBm)")
    for j, cid in enumerate(twin.cells):
        t = twin.topology.loc[cid]
        ax.plot(t.cell_lon, t.cell_lat, "k^", ms=8)
        ax.annotate(f"cell {cid}\n{int((serving == j).sum())} px", (t.cell_lon, t.cell_lat),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
