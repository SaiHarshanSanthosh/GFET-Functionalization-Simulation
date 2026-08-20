from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT / OUTPUT
# ============================================================

PRODUCTION_CSV = Path(
    "analysis/production_300K_500to5500ps_log.csv"
)

ENERGY_SCAN_CSV = Path(
    "analysis/interaction_energy_scan/"
    "pyrene_graphene_interaction_energy_scan.csv"
)

FIGURE_DIR = Path(
    "analysis/figures"
)

OUTPUT_SUMMARY_CSV = Path(
    "analysis/pyrene_production_5ns_block_summary.csv"
)

OUTPUT_SUMMARY_TXT = Path(
    "analysis/pyrene_production_5ns_summary.txt"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK FILES
# ============================================================

if not PRODUCTION_CSV.exists():
    raise FileNotFoundError(
        f"Missing production log:\n{PRODUCTION_CSV}"
    )

if not ENERGY_SCAN_CSV.exists():
    raise FileNotFoundError(
        f"Missing interaction scan:\n{ENERGY_SCAN_CSV}"
    )


# ============================================================
# LOAD PRODUCTION DATA
# ============================================================

df = pd.read_csv(
    PRODUCTION_CSV
)

required_columns = [
    "time_ps",
    "temperature_K",
    "pyrene_height_nm",
    "pyrene_tilt_deg",
    "water_O_min_nm",
    "water_O_max_nm",
    "waters_below_graphene",
    "upper_wall_violation_nm",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


time_ps = (
    df["time_ps"]
    .to_numpy(dtype=float)
)

time_ns = (
    (
        time_ps
        -
        time_ps[0]
    )
    /
    1000.0
)

temperature_K = (
    df["temperature_K"]
    .to_numpy(dtype=float)
)

height_A = (
    df["pyrene_height_nm"]
    .to_numpy(dtype=float)
    *
    10.0
)

tilt_deg = (
    df["pyrene_tilt_deg"]
    .to_numpy(dtype=float)
)

water_min_nm = (
    df["water_O_min_nm"]
    .to_numpy(dtype=float)
)

water_max_nm = (
    df["water_O_max_nm"]
    .to_numpy(dtype=float)
)

waters_below = (
    df["waters_below_graphene"]
    .to_numpy(dtype=int)
)

wall_violation_nm = (
    df["upper_wall_violation_nm"]
    .to_numpy(dtype=float)
)


# ============================================================
# VALIDATE TIME RANGE
# ============================================================

print()
print("=" * 74)
print("5 NS PYRENE PRODUCTION ANALYSIS")
print("=" * 74)

print(
    f"Frames in log:          {len(df)}"
)

print(
    f"Start time:             {time_ps[0]:.3f} ps"
)

print(
    f"End time:               {time_ps[-1]:.3f} ps"
)

print(
    f"Production duration:    "
    f"{(time_ps[-1] - time_ps[0]) / 1000.0:.3f} ns"
)


if abs(
    time_ps[0] - 500.0
) > 1.0e-6:

    raise RuntimeError(
        "Expected production log to start at 500 ps."
    )


if abs(
    time_ps[-1] - 5500.0
) > 1.0e-6:

    raise RuntimeError(
        "Expected production log to end at 5500 ps."
    )


# ============================================================
# LOAD ACTUAL RIGID-SCAN MINIMUM
# ============================================================

scan = pd.read_csv(
    ENERGY_SCAN_CSV
)

scan_distance_A = (
    scan["distance_A"]
    .to_numpy(dtype=float)
)

scan_full_energy = (
    scan[
        "full_ligand_graphene_kJ_mol"
    ]
    .to_numpy(dtype=float)
)

scan_pyrene_energy = (
    scan[
        "pyrene_aromatic_graphene_kJ_mol"
    ]
    .to_numpy(dtype=float)
)


full_scan_index = int(
    np.argmin(
        scan_full_energy
    )
)

pyrene_scan_index = int(
    np.argmin(
        scan_pyrene_energy
    )
)


full_scan_min_A = float(
    scan_distance_A[
        full_scan_index
    ]
)

pyrene_scan_min_A = float(
    scan_distance_A[
        pyrene_scan_index
    ]
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

height_mean_A = float(
    height_A.mean()
)

height_std_A = float(
    height_A.std()
)

height_min_A = float(
    height_A.min()
)

height_max_A = float(
    height_A.max()
)


tilt_mean_deg = float(
    tilt_deg.mean()
)

tilt_std_deg = float(
    tilt_deg.std()
)

tilt_max_deg = float(
    tilt_deg.max()
)


temperature_mean_K = float(
    temperature_K.mean()
)

temperature_std_K = float(
    temperature_K.std()
)


max_waters_below = int(
    waters_below.max()
)

max_wall_violation_nm = float(
    wall_violation_nm.max()
)


# Descriptive metric only.
# This is NOT being defined as a desorption criterion.
frames_above_4A = int(
    np.count_nonzero(
        height_A > 4.0
    )
)

fraction_above_4A = float(
    frames_above_4A
    /
    len(height_A)
)


# ============================================================
# RUNNING MEAN
#
# Production log spacing = 2 ps.
# Use a 100 ps running window.
# ============================================================

dt_ps = float(
    np.median(
        np.diff(
            time_ps
        )
    )
)

RUNNING_WINDOW_PS = 100.0

running_points = max(
    1,
    int(
        round(
            RUNNING_WINDOW_PS
            /
            dt_ps
        )
    ),
)


height_running_A = (
    pd.Series(
        height_A
    )
    .rolling(
        window=running_points,
        center=True,
        min_periods=1,
    )
    .mean()
    .to_numpy()
)


tilt_running_deg = (
    pd.Series(
        tilt_deg
    )
    .rolling(
        window=running_points,
        center=True,
        min_periods=1,
    )
    .mean()
    .to_numpy()
)


# ============================================================
# 500 ps BLOCK ANALYSIS
#
# This is important:
# if adsorption is stable, block means should not show
# systematic vertical drift over the 5 ns trajectory.
# ============================================================

BLOCK_PS = 500.0

production_start_ps = float(
    time_ps[0]
)

production_end_ps = float(
    time_ps[-1]
)


block_rows = []


block_start = (
    production_start_ps
)


while block_start < production_end_ps - 1.0e-6:

    block_end = min(
        block_start
        +
        BLOCK_PS,
        production_end_ps,
    )

    # Avoid double-counting boundaries.
    if block_end < production_end_ps:

        mask = (
            (time_ps >= block_start)
            &
            (time_ps < block_end)
        )

    else:

        mask = (
            (time_ps >= block_start)
            &
            (time_ps <= block_end)
        )


    block_height = (
        height_A[
            mask
        ]
    )

    block_tilt = (
        tilt_deg[
            mask
        ]
    )

    block_temperature = (
        temperature_K[
            mask
        ]
    )


    if len(
        block_height
    ) == 0:

        raise RuntimeError(
            "Empty block detected."
        )


    block_rows.append(
        {
            "block_start_ps": block_start,
            "block_end_ps": block_end,

            "block_center_ns": (
                (
                    0.5
                    *
                    (
                        block_start
                        +
                        block_end
                    )
                    -
                    production_start_ps
                )
                /
                1000.0
            ),

            "frames": len(
                block_height
            ),

            "height_mean_A": float(
                block_height.mean()
            ),

            "height_std_A": float(
                block_height.std()
            ),

            "height_min_A": float(
                block_height.min()
            ),

            "height_max_A": float(
                block_height.max()
            ),

            "tilt_mean_deg": float(
                block_tilt.mean()
            ),

            "tilt_max_deg": float(
                block_tilt.max()
            ),

            "temperature_mean_K": float(
                block_temperature.mean()
            ),

            "temperature_std_K": float(
                block_temperature.std()
            ),
        }
    )


    block_start = (
        block_end
    )


blocks = pd.DataFrame(
    block_rows
)

blocks.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False,
)


# ============================================================
# SIMPLE LINEAR DRIFT ESTIMATE
#
# This is descriptive only.
# It tells us whether the average height trends upward/downward.
# ============================================================

slope_A_per_ns, intercept = np.polyfit(
    time_ns,
    height_A,
    1,
)

slope_A_per_ns = float(
    slope_A_per_ns
)


# ============================================================
# PRINT SUMMARY
# ============================================================

summary_lines = []

summary_lines.append(
    "=" * 74
)

summary_lines.append(
    "PYRENE / GRAPHENE 5 NS PRODUCTION SUMMARY"
)

summary_lines.append(
    "=" * 74
)

summary_lines.append(
    ""
)

summary_lines.append(
    f"Production interval: "
    f"{time_ps[0]:.1f} -> "
    f"{time_ps[-1]:.1f} ps "
    f"({time_ns[-1]:.3f} ns)"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "PYRENE HEIGHT"
)

summary_lines.append(
    f"  Mean:          "
    f"{height_mean_A:.4f} A"
)

summary_lines.append(
    f"  SD:            "
    f"{height_std_A:.4f} A"
)

summary_lines.append(
    f"  Range:         "
    f"{height_min_A:.4f} -> "
    f"{height_max_A:.4f} A"
)

summary_lines.append(
    f"  Linear slope:  "
    f"{slope_A_per_ns:+.5f} A/ns"
)

summary_lines.append(
    f"  Frames >4 A:   "
    f"{frames_above_4A} / "
    f"{len(height_A)} "
    f"({100.0 * fraction_above_4A:.3f}%)"
)

summary_lines.append(
    "  (>4 A is descriptive only, not a desorption definition.)"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "ORIENTATION"
)

summary_lines.append(
    f"  Mean tilt:     "
    f"{tilt_mean_deg:.3f} deg"
)

summary_lines.append(
    f"  Tilt SD:       "
    f"{tilt_std_deg:.3f} deg"
)

summary_lines.append(
    f"  Maximum tilt:  "
    f"{tilt_max_deg:.3f} deg"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "THERMOSTAT / WATER"
)

summary_lines.append(
    f"  Temperature:   "
    f"{temperature_mean_K:.3f} +/- "
    f"{temperature_std_K:.3f} K"
)

summary_lines.append(
    f"  Max waters below graphene: "
    f"{max_waters_below}"
)

summary_lines.append(
    f"  Max upper-wall violation:  "
    f"{max_wall_violation_nm:.6f} nm"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "RIGID ENERGY-SCAN REFERENCE"
)

summary_lines.append(
    f"  Full linker minimum: "
    f"{full_scan_min_A:.3f} A"
)

summary_lines.append(
    f"  Pyrene minimum:      "
    f"{pyrene_scan_min_A:.3f} A"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "500 ps BLOCK MEANS"
)


for _, row in blocks.iterrows():

    summary_lines.append(
        (
            f"  "
            f"{row['block_start_ps']:6.0f}"
            f"-"
            f"{row['block_end_ps']:6.0f} ps:  "
            f"{row['height_mean_A']:.4f} "
            f"+/- "
            f"{row['height_std_A']:.4f} A"
        )
    )


summary_lines.append(
    ""
)

summary_lines.append(
    "=" * 74
)


summary_text = "\n".join(
    summary_lines
)

print()
print(
    summary_text
)

OUTPUT_SUMMARY_TXT.write_text(
    summary_text,
    encoding="utf-8",
)


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


# ============================================================
# FIGURE 1
# FULL HEIGHT TRAJECTORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)


ax.plot(
    time_ns,
    height_A,
    linewidth=0.8,
    alpha=0.65,
    label="Instantaneous separation",
)


ax.plot(
    time_ns,
    height_running_A,
    linewidth=2.2,
    label="100 ps running mean",
)


ax.axhline(
    height_mean_A,
    linestyle="--",
    linewidth=2,
    label=(
        f"5 ns mean = "
        f"{height_mean_A:.3f} A"
    ),
)


ax.axhline(
    full_scan_min_A,
    linestyle=":",
    linewidth=2,
    label=(
        f"Rigid-scan minimum = "
        f"{full_scan_min_A:.2f} A"
    ),
)


ax.fill_between(
    time_ns,
    height_mean_A - height_std_A,
    height_mean_A + height_std_A,
    alpha=0.12,
    label=(
        f"5 ns mean +/- 1 SD"
    ),
)


ax.set_xlabel(
    "Production time (ns)"
)

ax.set_ylabel(
    "Pyrene-graphene separation (A)"
)

ax.set_title(
    "Pyrene Remains Surface-Associated Throughout 5 ns Production MD"
)

ax.set_xlim(
    0,
    5,
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    frameon=False,
)


fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    /
    "09_pyrene_5ns_height_vs_time.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# 500 ps BLOCK STABILITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5.8)
)


ax.errorbar(
    blocks["block_center_ns"],
    blocks["height_mean_A"],
    yerr=blocks["height_std_A"],
    marker="o",
    capsize=4,
    linewidth=1.8,
)


ax.axhline(
    height_mean_A,
    linestyle="--",
    linewidth=2,
    label=(
        f"Overall mean = "
        f"{height_mean_A:.3f} A"
    ),
)


ax.axhline(
    full_scan_min_A,
    linestyle=":",
    linewidth=2,
    label=(
        f"Rigid-scan minimum = "
        f"{full_scan_min_A:.2f} A"
    ),
)


ax.set_xlabel(
    "Production time (ns)"
)

ax.set_ylabel(
    "500 ps block mean separation (A)"
)

ax.set_title(
    "Adsorption Distance Is Stable Across Consecutive 500 ps Blocks"
)

ax.set_xlim(
    0,
    5,
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    frameon=False,
)


fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    /
    "10_pyrene_5ns_block_stability.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)


ax.hist(
    height_A,
    bins=35,
    alpha=0.82,
)


ax.axvline(
    height_mean_A,
    linestyle="--",
    linewidth=2.2,
    label=(
        f"5 ns mean = "
        f"{height_mean_A:.3f} A"
    ),
)


ax.axvline(
    full_scan_min_A,
    linestyle=":",
    linewidth=2.2,
    label=(
        f"Rigid-scan minimum = "
        f"{full_scan_min_A:.2f} A"
    ),
)


ax.set_xlabel(
    "Pyrene-graphene separation (A)"
)

ax.set_ylabel(
    "Production frames"
)

ax.set_title(
    "5 ns Distribution of Pyrene-Graphene Separation"
)

ax.grid(
    alpha=0.22,
)

ax.legend(
    frameon=False,
)


fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    /
    "11_pyrene_5ns_height_distribution.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 4
# PI SUMMARY
# ============================================================

fig = plt.figure(
    figsize=(13.5, 9.5)
)


gs = fig.add_gridspec(
    2,
    2,
    hspace=0.33,
    wspace=0.27,
)


# ------------------------------------------------------------
# A. HEIGHT
# ------------------------------------------------------------

ax1 = fig.add_subplot(
    gs[0, 0]
)


ax1.plot(
    time_ns,
    height_A,
    linewidth=0.7,
    alpha=0.55,
)


ax1.plot(
    time_ns,
    height_running_A,
    linewidth=2,
    label="100 ps running mean",
)


ax1.axhline(
    full_scan_min_A,
    linestyle=":",
    linewidth=1.8,
    label=(
        f"Energy minimum "
        f"{full_scan_min_A:.2f} A"
    ),
)


ax1.set_xlabel(
    "Production time (ns)"
)

ax1.set_ylabel(
    "Separation (A)"
)

ax1.set_title(
    "A. Pyrene-graphene separation"
)

ax1.set_xlim(
    0,
    5,
)

ax1.grid(
    alpha=0.2,
)

ax1.legend(
    frameon=False,
    fontsize=8,
)


# ------------------------------------------------------------
# B. BLOCK MEANS
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    gs[0, 1]
)


ax2.errorbar(
    blocks["block_center_ns"],
    blocks["height_mean_A"],
    yerr=blocks["height_std_A"],
    marker="o",
    capsize=3,
    linewidth=1.6,
)


ax2.axhline(
    height_mean_A,
    linestyle="--",
    linewidth=1.8,
)


ax2.set_xlabel(
    "Production time (ns)"
)

ax2.set_ylabel(
    "500 ps mean separation (A)"
)

ax2.set_title(
    "B. Block stability"
)

ax2.set_xlim(
    0,
    5,
)

ax2.grid(
    alpha=0.2,
)


# ------------------------------------------------------------
# C. TILT
# ------------------------------------------------------------

ax3 = fig.add_subplot(
    gs[1, 0]
)


ax3.plot(
    time_ns,
    tilt_deg,
    linewidth=0.7,
    alpha=0.55,
)


ax3.plot(
    time_ns,
    tilt_running_deg,
    linewidth=2,
    label="100 ps running mean",
)


ax3.axhline(
    tilt_mean_deg,
    linestyle="--",
    linewidth=1.8,
    label=(
        f"Mean = "
        f"{tilt_mean_deg:.2f} deg"
    ),
)


ax3.set_xlabel(
    "Production time (ns)"
)

ax3.set_ylabel(
    "Pyrene tilt (degrees)"
)

ax3.set_title(
    "C. Pyrene orientation"
)

ax3.set_xlim(
    0,
    5,
)

ax3.grid(
    alpha=0.2,
)

ax3.legend(
    frameon=False,
    fontsize=8,
)


# ------------------------------------------------------------
# D. QUANTITATIVE SUMMARY
# ------------------------------------------------------------

ax4 = fig.add_subplot(
    gs[1, 1]
)

ax4.axis(
    "off"
)


summary_box = (
    "5 ns production result\n\n"

    f"Mean separation\n"
    f"{height_mean_A:.3f} +/- "
    f"{height_std_A:.3f} A\n\n"

    f"Observed range\n"
    f"{height_min_A:.3f} - "
    f"{height_max_A:.3f} A\n\n"

    f"Rigid-scan minimum\n"
    f"{full_scan_min_A:.2f} A\n\n"

    f"Mean tilt\n"
    f"{tilt_mean_deg:.2f} deg\n\n"

    f"Maximum tilt\n"
    f"{tilt_max_deg:.2f} deg\n\n"

    f"Temperature\n"
    f"{temperature_mean_K:.2f} +/- "
    f"{temperature_std_K:.2f} K\n\n"

    f"Height drift slope\n"
    f"{slope_A_per_ns:+.4f} A/ns\n\n"

    f"Waters below graphene\n"
    f"{max_waters_below}\n\n"

    "Pyrene unrestrained"
)


ax4.text(
    0.08,
    0.94,
    summary_box,
    transform=ax4.transAxes,
    verticalalignment="top",
    fontsize=13,
)


fig.suptitle(
    "5 ns Stability of Pyrene-PEG5 Adsorption on Supported Graphene",
    fontsize=18,
    y=0.98,
)


fig.savefig(
    FIGURE_DIR
    /
    "12_PI_summary_pyrene_5ns.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# COMPLETE
# ============================================================

print()
print(
    "PASS: 5 ns analysis complete"
)

print()

print(
    "Figures:"
)

print(
    "  analysis/figures/"
    "09_pyrene_5ns_height_vs_time.png"
)

print(
    "  analysis/figures/"
    "10_pyrene_5ns_block_stability.png"
)

print(
    "  analysis/figures/"
    "11_pyrene_5ns_height_distribution.png"
)

print(
    "  analysis/figures/"
    "12_PI_summary_pyrene_5ns.png"
)

print()

print(
    f"Block CSV: {OUTPUT_SUMMARY_CSV}"
)

print(
    f"Summary:   {OUTPUT_SUMMARY_TXT}"
)

print()

print(
    "PYRENE_5NS_ANALYSIS_PASS"
)