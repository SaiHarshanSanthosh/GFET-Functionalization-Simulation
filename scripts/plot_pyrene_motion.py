from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = Path(
    "analysis/pyrene_motion_110to500ps.csv"
)

OUTPUT_DIR = Path(
    "analysis/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if not INPUT_CSV.exists():
    raise RuntimeError(
        f"Missing analysis CSV: {INPUT_CSV}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

time_ps = df["time_ps"].to_numpy()

height_A = (
    df["pyrene_height_nm"].to_numpy()
    * 10.0
)

lateral_nm = (
    df["lateral_displacement_nm"]
    .to_numpy()
)

x_nm = (
    df["lateral_x_nm"]
    .to_numpy()
)

y_nm = (
    df["lateral_y_nm"]
    .to_numpy()
)

path_nm = (
    df["path_length_nm"]
    .to_numpy()
)

tilt_deg = (
    df["tilt_deg"]
    .to_numpy()
)


# ============================================================
# SUMMARY
# ============================================================

mean_height = float(
    height_A.mean()
)

std_height = float(
    height_A.std()
)

min_height = float(
    height_A.min()
)

max_height = float(
    height_A.max()
)

mean_tilt = float(
    tilt_deg.mean()
)

std_tilt = float(
    tilt_deg.std()
)

max_tilt = float(
    tilt_deg.max()
)

final_displacement = float(
    lateral_nm[-1]
)

max_displacement = float(
    lateral_nm.max()
)

total_path = float(
    path_nm[-1]
)


print()
print("=" * 70)
print("PYRENE-GRAPHENE QUANTITATIVE SUMMARY")
print("=" * 70)

print(
    f"Mean separation:      "
    f"{mean_height:.3f} A"
)

print(
    f"Separation SD:        "
    f"{std_height:.3f} A"
)

print(
    f"Separation range:     "
    f"{min_height:.3f} - "
    f"{max_height:.3f} A"
)

print(
    f"Mean tilt:            "
    f"{mean_tilt:.2f} degrees"
)

print(
    f"Maximum tilt:         "
    f"{max_tilt:.2f} degrees"
)

print(
    f"Final lateral shift:  "
    f"{final_displacement:.3f} nm"
)

print(
    f"Maximum lateral shift:"
    f" {max_displacement:.3f} nm"
)

print(
    f"Cumulative path:      "
    f"{total_path:.3f} nm"
)

print("=" * 70)
print()


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


# ============================================================
# 1. SEPARATION VS TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5.5)
)

ax.plot(
    time_ps,
    height_A,
    linewidth=1.3,
    label="Instantaneous separation",
)

ax.axhline(
    mean_height,
    linestyle="--",
    linewidth=2,
    label=(
        f"MD mean = "
        f"{mean_height:.2f} Å"
    ),
)

ax.axhline(
    3.5,
    linestyle=":",
    linewidth=2,
    label="3.5 Å reference",
)

ax.fill_between(
    time_ps,
    mean_height - std_height,
    mean_height + std_height,
    alpha=0.15,
    label=(
        f"±1 SD = "
        f"{std_height:.2f} Å"
    ),
)

ax.set_xlabel(
    "Simulation time (ps)"
)

ax.set_ylabel(
    "Pyrene–graphene separation (Å)"
)

ax.set_title(
    "Unrestrained Pyrene Maintains Stable Contact with Graphene"
)

ax.set_xlim(
    time_ps.min(),
    time_ps.max(),
)

ax.legend(
    frameon=False,
)

ax.grid(
    alpha=0.2,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "01_pyrene_graphene_separation_vs_time.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 2. SEPARATION DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

ax.hist(
    height_A,
    bins=28,
    alpha=0.8,
)

ax.axvline(
    mean_height,
    linestyle="--",
    linewidth=2,
    label=(
        f"MD mean = "
        f"{mean_height:.2f} Å"
    ),
)

ax.axvline(
    3.5,
    linestyle=":",
    linewidth=2,
    label="3.5 Å reference",
)

ax.set_xlabel(
    "Pyrene–graphene separation (Å)"
)

ax.set_ylabel(
    "Number of trajectory frames"
)

ax.set_title(
    "Pyrene–Graphene Separation Distribution"
)

ax.legend(
    frameon=False,
)

ax.grid(
    alpha=0.2,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "02_pyrene_graphene_separation_distribution.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 3. LATERAL DISPLACEMENT VS TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5.5)
)

ax.plot(
    time_ps,
    lateral_nm,
    linewidth=1.5,
)

ax.set_xlabel(
    "Simulation time (ps)"
)

ax.set_ylabel(
    "Net lateral displacement (nm)"
)

ax.set_title(
    "Pyrene Slides Laterally While Remaining Surface-Bound"
)

ax.set_xlim(
    time_ps.min(),
    time_ps.max(),
)

summary = (
    f"Final displacement = "
    f"{final_displacement:.28f} nm\n"
    f"Maximum displacement = "
    f"{max_displacement:.2f} nm"
)

# Fix formatting cleanly.
summary = (
    f"Final displacement = "
    f"{final_displacement:.2f} nm\n"
    f"Maximum displacement = "
    f"{max_displacement:.2f} nm"
)

ax.text(
    0.02,
    0.96,
    summary,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox={
        "boxstyle": "round",
        "alpha": 0.15,
    },
)

ax.grid(
    alpha=0.2,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "03_pyrene_lateral_displacement_vs_time.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 4. XY SURFACE TRAJECTORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    x_nm,
    y_nm,
    linewidth=1.2,
)

ax.scatter(
    x_nm[0],
    y_nm[0],
    s=80,
    label="110 ps",
    zorder=3,
)

ax.scatter(
    x_nm[-1],
    y_nm[-1],
    s=90,
    marker="X",
    label="500 ps",
    zorder=3,
)

ax.set_xlabel(
    "x displacement on graphene (nm)"
)

ax.set_ylabel(
    "y displacement on graphene (nm)"
)

ax.set_title(
    "Lateral Path of Pyrene Across the Graphene Surface"
)

ax.axis(
    "equal"
)

ax.legend(
    frameon=False,
)

ax.grid(
    alpha=0.2,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "04_pyrene_xy_surface_trajectory.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 5. TILT VS TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5.5)
)

ax.plot(
    time_ps,
    tilt_deg,
    linewidth=1.4,
)

ax.axhline(
    mean_tilt,
    linestyle="--",
    linewidth=2,
    label=(
        f"Mean tilt = "
        f"{mean_tilt:.1f}°"
    ),
)

ax.set_xlabel(
    "Simulation time (ps)"
)

ax.set_ylabel(
    "Tilt angle (degrees)"
)

ax.set_title(
    "Pyrene Remains Nearly Parallel to the Graphene Plane"
)

ax.set_xlim(
    time_ps.min(),
    time_ps.max(),
)

ax.legend(
    frameon=False,
)

ax.grid(
    alpha=0.2,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "05_pyrene_tilt_vs_time.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 6. PI SUMMARY FIGURE
# ============================================================

fig = plt.figure(
    figsize=(13, 9)
)

gs = fig.add_gridspec(
    2,
    2,
    hspace=0.32,
    wspace=0.28,
)


# Panel A
ax1 = fig.add_subplot(
    gs[0, 0]
)

ax1.plot(
    time_ps,
    height_A,
    linewidth=1.1,
)

ax1.axhline(
    mean_height,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_height:.2f} Å",
)

ax1.axhline(
    3.5,
    linestyle=":",
    linewidth=1.8,
    label="3.5 Å reference",
)

ax1.set_xlabel(
    "Time (ps)"
)

ax1.set_ylabel(
    "Separation (Å)"
)

ax1.set_title(
    "A. Pyrene–graphene separation"
)

ax1.legend(
    frameon=False,
    fontsize=9,
)

ax1.grid(
    alpha=0.2,
)


# Panel B
ax2 = fig.add_subplot(
    gs[0, 1]
)

ax2.plot(
    time_ps,
    tilt_deg,
    linewidth=1.2,
)

ax2.axhline(
    mean_tilt,
    linestyle="--",
    linewidth=2,
)

ax2.set_xlabel(
    "Time (ps)"
)

ax2.set_ylabel(
    "Tilt (°)"
)

ax2.set_title(
    "B. Pyrene orientation"
)

ax2.grid(
    alpha=0.2,
)


# Panel C
ax3 = fig.add_subplot(
    gs[1, 0]
)

ax3.plot(
    x_nm,
    y_nm,
    linewidth=1.1,
)

ax3.scatter(
    x_nm[0],
    y_nm[0],
    s=65,
    label="Start",
)

ax3.scatter(
    x_nm[-1],
    y_nm[-1],
    s=75,
    marker="X",
    label="500 ps",
)

ax3.set_xlabel(
    "x displacement (nm)"
)

ax3.set_ylabel(
    "y displacement (nm)"
)

ax3.set_title(
    "C. Lateral surface motion"
)

ax3.axis(
    "equal"
)

ax3.legend(
    frameon=False,
    fontsize=9,
)

ax3.grid(
    alpha=0.2,
)


# Panel D
ax4 = fig.add_subplot(
    gs[1, 1]
)

ax4.axis(
    "off"
)

summary_text = (
    "500 ps MD summary\n\n"
    f"Mean separation\n"
    f"{mean_height:.2f} ± "
    f"{std_height:.2f} Å\n\n"
    f"Separation range\n"
    f"{min_height:.2f}–"
    f"{max_height:.2f} Å\n\n"
    f"Mean tilt\n"
    f"{mean_tilt:.1f}°\n\n"
    f"Maximum tilt\n"
    f"{max_tilt:.1f}°\n\n"
    f"Maximum lateral displacement\n"
    f"{max_displacement:.2f} nm\n\n"
    "Pyrene remained unrestrained."
)

ax4.text(
    0.08,
    0.92,
    summary_text,
    transform=ax4.transAxes,
    verticalalignment="top",
    fontsize=14,
)


fig.suptitle(
    "Pyrene–PEG5 Adsorption Dynamics on Supported Graphene",
    fontsize=18,
    y=0.98,
)

fig.savefig(
    OUTPUT_DIR
    / "00_PI_summary_pyrene_graphene.png",
    bbox_inches="tight",
)

plt.close(fig)


print(
    "PASS: all figures created"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print()
print(
    "Most useful PI figure:"
)

print(
    OUTPUT_DIR
    / "00_PI_summary_pyrene_graphene.png"
)

