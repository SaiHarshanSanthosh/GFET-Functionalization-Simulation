from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path(
    "analysis/pyrene_lateral_5ns.csv"
)

OUTPUT_DIR = Path(
    "analysis/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(INPUT)

time_ns = (
    df["time_ps"].to_numpy()
    - df["time_ps"].iloc[0]
) / 1000.0

x_nm = df["lateral_x_nm"].to_numpy()
y_nm = df["lateral_y_nm"].to_numpy()

disp_nm = (
    df["lateral_displacement_nm"]
    .to_numpy()
)

path_nm = (
    df["path_length_nm"]
    .to_numpy()
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.5),
)


# ------------------------------------------------------------
# A. XY surface trajectory
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    x_nm,
    y_nm,
    linewidth=1.0,
)

ax.scatter(
    x_nm[0],
    y_nm[0],
    s=80,
    label="Start",
    zorder=5,
)

ax.scatter(
    x_nm[-1],
    y_nm[-1],
    s=90,
    marker="X",
    label="5 ns",
    zorder=5,
)

ax.set_xlabel(
    "x displacement (nm)"
)

ax.set_ylabel(
    "y displacement (nm)"
)

ax.set_title(
    "A. Pyrene lateral trajectory on graphene"
)

ax.axis("equal")

ax.grid(alpha=0.2)

ax.legend(
    frameon=False
)


# ------------------------------------------------------------
# B. Net displacement vs time
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    time_ns,
    disp_nm,
    linewidth=1.4,
)

ax.set_xlabel(
    "Production time (ns)"
)

ax.set_ylabel(
    "Net lateral displacement (nm)"
)

ax.set_title(
    "B. Lateral displacement from starting position"
)

ax.grid(alpha=0.2)


summary = (
    f"Final displacement = "
    f"{disp_nm[-1]:.2f} nm\n"
    f"Maximum displacement = "
    f"{disp_nm.max():.2f} nm\n"
    f"2 ps-sampled path = "
    f"{path_nm[-1]:.1f} nm"
)

ax.text(
    0.04,
    0.96,
    summary,
    transform=ax.transAxes,
    va="top",
    bbox={
        "boxstyle": "round",
        "alpha": 0.15,
    },
)


fig.suptitle(
    "Pyrene Remains Adsorbed While Diffusing Laterally Across Graphene",
    fontsize=17,
)

fig.tight_layout()

OUTPUT = (
    OUTPUT_DIR
    /
    "13_pyrene_5ns_lateral_motion.png"
)

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("PASS: lateral figure created")
print(OUTPUT)
