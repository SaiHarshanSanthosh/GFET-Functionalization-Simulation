from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUTS
# ============================================================

CSV_FILE = Path(
    "analysis/pyrene_motion_110to500ps.csv"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "equilibrated_300K_500ps_positions_nm.npy"
)

OUTPUT_DIR = Path(
    "analysis/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SYSTEM INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250
LIGAND_START = 3750

# Exact 16 aromatic pyrene atoms identified previously.
PYRENE_LOCAL = np.arange(
    2,
    18,
    dtype=int,
)

PYRENE_GLOBAL = (
    LIGAND_START
    +
    PYRENE_LOCAL
)


# ============================================================
# LOAD TRAJECTORY ANALYSIS
# ============================================================

df = pd.read_csv(
    CSV_FILE
)

time_ps = (
    df["time_ps"]
    .to_numpy()
)

separation_A = (
    df["pyrene_height_nm"]
    .to_numpy()
    * 10.0
)


mean_sep = float(
    separation_A.mean()
)

std_sep = float(
    separation_A.std()
)


# ============================================================
# LOAD FINAL 500 ps COORDINATES
# ============================================================

positions_nm = np.load(
    POSITIONS_FILE
)

graphene_z_A = (
    positions_nm[
        :N_GRAPHENE_CARBONS,
        2
    ]
    * 10.0
)

pyrene_z_A = (
    positions_nm[
        PYRENE_GLOBAL,
        2
    ]
    * 10.0
)


graphene_mean_A = float(
    graphene_z_A.mean()
)

pyrene_mean_A = float(
    pyrene_z_A.mean()
)

final_sep_A = (
    pyrene_mean_A
    -
    graphene_mean_A
)


# Coordinates relative to mean graphene plane.
graphene_relative_A = (
    graphene_z_A
    -
    graphene_mean_A
)

pyrene_relative_A = (
    pyrene_z_A
    -
    graphene_mean_A
)


# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


fig = plt.figure(
    figsize=(13, 9)
)

gs = fig.add_gridspec(
    2,
    2,
    hspace=0.35,
    wspace=0.28,
)


# ============================================================
# PANEL A: DEFINITION
# ============================================================

ax = fig.add_subplot(
    gs[0, 0]
)

ax.set_xlim(
    0,
    10,
)

ax.set_ylim(
    -0.8,
    5.0,
)


# Graphene plane
ax.plot(
    [1, 9],
    [0, 0],
    linewidth=5,
)

ax.text(
    5,
    -0.35,
    "Mean graphene carbon plane",
    ha="center",
    va="top",
)


# Pyrene aromatic plane
ax.plot(
    [3, 7],
    [mean_sep, mean_sep],
    linewidth=5,
)

ax.text(
    5,
    mean_sep + 0.28,
    "Mean of 16 pyrene aromatic atoms",
    ha="center",
    va="bottom",
)


# Distance arrow
ax.annotate(
    "",
    xy=(2.0, mean_sep),
    xytext=(2.0, 0),
    arrowprops={
        "arrowstyle": "<->",
        "linewidth": 2,
    },
)

ax.text(
    2.25,
    mean_sep / 2,
    (
        f"d = {mean_sep:.2f} "
        r"$\AA$"
    ),
    va="center",
)


formula = (
    r"$d = "
    r"\langle z_{\mathrm{pyrene}}\rangle"
    r" - "
    r"\langle z_{\mathrm{graphene}}\rangle$"
    "\n\n"
    "16 aromatic pyrene atoms\n"
    "1250 graphene carbon atoms"
)

ax.text(
    9.3,
    1.8,
    formula,
    ha="right",
    va="center",
)

ax.set_title(
    "A. Definition of the separation coordinate"
)

ax.axis(
    "off"
)


# ============================================================
# PANEL B: ACTUAL ATOMS AT 500 ps
# ============================================================

ax = fig.add_subplot(
    gs[0, 1]
)

rng = np.random.default_rng(
    20260819
)

x_graphene = (
    rng.normal(
        0.0,
        0.06,
        len(graphene_relative_A),
    )
)

x_pyrene = (
    rng.normal(
        1.0,
        0.035,
        len(pyrene_relative_A),
    )
)


ax.scatter(
    x_graphene,
    graphene_relative_A,
    s=8,
    alpha=0.22,
)

ax.scatter(
    x_pyrene,
    pyrene_relative_A,
    s=35,
    alpha=0.85,
)


ax.hlines(
    0.0,
    -0.28,
    0.28,
    linewidth=2,
)

ax.hlines(
    final_sep_A,
    0.72,
    1.28,
    linewidth=2,
)


ax.annotate(
    "",
    xy=(0.55, final_sep_A),
    xytext=(0.55, 0),
    arrowprops={
        "arrowstyle": "<->",
        "linewidth": 1.8,
    },
)


ax.text(
    0.59,
    final_sep_A / 2,
    (
        f"{final_sep_A:.2f} "
        r"$\AA$"
        "\n(at 500 ps)"
    ),
    va="center",
)


ax.set_xticks(
    [0, 1],
    [
        "Graphene\n1250 C atoms",
        "Pyrene\n16 aromatic atoms",
    ],
)

ax.set_ylabel(
    r"z relative to mean graphene plane ($\AA$)"
)

ax.set_title(
    "B. Atomic coordinates at 500 ps"
)

ax.grid(
    axis="y",
    alpha=0.2,
)


# ============================================================
# PANEL C: FULL DISTRIBUTION
# ============================================================

ax = fig.add_subplot(
    gs[1, 0]
)

ax.hist(
    separation_A,
    bins=28,
    alpha=0.82,
)


ax.axvline(
    mean_sep,
    linestyle="--",
    linewidth=2.5,
    label=(
        f"MD mean = "
        f"{mean_sep:.3f} "
        r"$\AA$"
    ),
)


ax.axvline(
    3.5,
    linestyle=":",
    linewidth=2.5,
    label=(
        "3.5 "
        r"$\AA$"
        " reference"
    ),
)


ax.axvspan(
    mean_sep - std_sep,
    mean_sep + std_sep,
    alpha=0.14,
    label=(
        r"$\pm 1$ SD = "
        f"{std_sep:.3f} "
        r"$\AA$"
    ),
)


ax.set_xlabel(
    r"Pyrene-graphene separation ($\AA$)"
)

ax.set_ylabel(
    "Trajectory frames"
)

ax.set_title(
    "C. Separation sampled over 110-500 ps"
)

ax.legend(
    frameon=False,
)

ax.grid(
    alpha=0.2,
)


# ============================================================
# PANEL D: SEPARATION VS TIME
# ============================================================

ax = fig.add_subplot(
    gs[1, 1]
)


ax.plot(
    time_ps,
    separation_A,
    linewidth=1.1,
)


ax.axhline(
    mean_sep,
    linestyle="--",
    linewidth=2,
    label=(
        f"Mean = "
        f"{mean_sep:.3f} "
        r"$\AA$"
    ),
)


ax.axhline(
    3.5,
    linestyle=":",
    linewidth=2,
    label=(
        "3.5 "
        r"$\AA$"
        " reference"
    ),
)


ax.fill_between(
    time_ps,
    mean_sep - std_sep,
    mean_sep + std_sep,
    alpha=0.14,
)


ax.set_xlabel(
    "Simulation time (ps)"
)

ax.set_ylabel(
    r"Separation ($\AA$)"
)

ax.set_title(
    "D. Stable interfacial distance over time"
)

ax.legend(
    frameon=False,
)

ax.grid(
    alpha=0.2,
)


# ============================================================
# TITLE + SUMMARY
# ============================================================

fig.suptitle(
    "How the Pyrene-Graphene Adsorption Distance Is Quantified",
    fontsize=18,
    y=0.98,
)


fig.text(
    0.5,
    0.015,
    (
        f"Trajectory result: "
        f"{mean_sep:.3f} +/- "
        f"{std_sep:.3f} Angstrom   |   "
        f"500 ps frame: "
        f"{final_sep_A:.3f} Angstrom"
    ),
    ha="center",
    fontsize=12,
)


OUTPUT = (
    OUTPUT_DIR
    /
    "06_how_pyrene_graphene_distance_is_measured.png"
)

fig.savefig(
    OUTPUT,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("=" * 68)
print("PYRENE-GRAPHENE DISTANCE DERIVATION")
print("=" * 68)

print(
    f"Mean graphene z at 500 ps: "
    f"{graphene_mean_A:.3f} A"
)

print(
    f"Mean pyrene z at 500 ps:   "
    f"{pyrene_mean_A:.3f} A"
)

print(
    f"500 ps separation:         "
    f"{final_sep_A:.3f} A"
)

print()

print(
    f"Trajectory mean:            "
    f"{mean_sep:.3f} +/- "
    f"{std_sep:.3f} A"
)

print()
print("PASS: figure created")
print(OUTPUT)
