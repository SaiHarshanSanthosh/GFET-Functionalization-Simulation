from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

ANGLE_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_contact_angle_5ns_timeblocks.json"
)

GAP_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_periodic_gap_500to5000ps.json"
)

RUN_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_yb_300K_500ps_to_5000ps.json"
)

OUTPUT_PNG = (
    ROOT
    / "analysis"
    / "figures"
    / "24_ff1b_graphene_opc_wetting_validation_5ns.png"
)

OUTPUT_PDF = (
    ROOT
    / "analysis"
    / "figures"
    / "24_ff1b_graphene_opc_wetting_validation_5ns.pdf"
)


with open(ANGLE_JSON, encoding="utf-8") as f:
    angle_data = json.load(f)

with open(GAP_JSON, encoding="utf-8") as f:
    gap_data = json.load(f)

with open(RUN_JSON, encoding="utf-8") as f:
    run_data = json.load(f)


angle_blocks = angle_data["blocks"]
gap_blocks = gap_data["blocks_500ps"]


times_ns = np.asarray(
    [
        row["midpoint_ps"] / 1000.0
        for row in angle_blocks
    ],
    dtype=float,
)

angles = np.asarray(
    [
        row["contact_angle_deg"]
        for row in angle_blocks
    ],
    dtype=float,
)

widths = np.asarray(
    [
        row["mean_occupied_width_nm"]
        for row in gap_blocks
    ],
    dtype=float,
)

dry_gaps = np.asarray(
    [
        row["mean_largest_dry_gap_nm"]
        for row in gap_blocks
    ],
    dtype=float,
)


mean_angle = float(np.mean(angles))
sd_angle = float(np.std(angles))

mean_width = float(np.mean(widths))
sd_width = float(np.std(widths))

mean_gap = float(np.mean(dry_gaps))
sd_gap = float(np.std(dry_gaps))

edge_occupancy = float(
    gap_data["mean_edge_occupancy_fraction"]
)

waters_below = int(
    run_data["max_waters_below_graphene"]
)


print("=" * 80)
print("FF-1B DEJI SUMMARY FIGURE")
print("=" * 80)

print(
    f"Contact angle:       "
    f"{mean_angle:.3f} +/- {sd_angle:.3f} deg"
)

print(
    f"Mean occupied width: "
    f"{mean_width:.4f} +/- {sd_width:.4f} nm"
)

print(
    f"Mean dry gap:        "
    f"{mean_gap:.4f} +/- {sd_gap:.4f} nm"
)

print(
    f"Edge occupancy:      "
    f"{edge_occupancy:.6f}"
)

print(
    f"Waters below:        "
    f"{waters_below}"
)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(8.0, 9.0),
    sharex=True,
)


# ============================================================
# PANEL A — CONTACT ANGLE
# ============================================================

ax = axes[0]

ax.plot(
    times_ns,
    angles,
    marker="o",
    linewidth=1.8,
)

ax.axhline(
    mean_angle,
    linestyle="-",
    linewidth=1.4,
    label=f"5 ns mean = {mean_angle:.1f} deg",
)

ax.axhspan(
    mean_angle - sd_angle,
    mean_angle + sd_angle,
    alpha=0.12,
    label=f"Block SD = {sd_angle:.1f} deg",
)

ax.axhline(
    91.970,
    linestyle="--",
    linewidth=1.2,
    label="Earlier 300–500 ps estimate = 92.0 deg",
)

ax.set_ylabel(
    "Contact angle ( deg)"
)

ax.set_title(
    "A   Cylindrical contact-angle stability"
)

ax.legend(
    fontsize=8,
    loc="best",
)

ax.grid(
    alpha=0.2
)


# ============================================================
# PANEL B — OCCUPIED WIDTH
# ============================================================

ax = axes[1]

ax.plot(
    times_ns,
    widths,
    marker="o",
    linewidth=1.8,
)

ax.axhline(
    mean_width,
    linestyle="--",
    linewidth=1.2,
    label=f"Mean = {mean_width:.3f} nm",
)

ax.set_ylabel(
    "Occupied width (nm)"
)

ax.set_title(
    "B   Droplet lateral extent"
)

ax.legend(
    fontsize=8,
    loc="best",
)

ax.grid(
    alpha=0.2
)


# ============================================================
# PANEL C — PERIODIC DRY GAP
# ============================================================

ax = axes[2]

ax.plot(
    times_ns,
    dry_gaps,
    marker="o",
    linewidth=1.8,
)

ax.axhline(
    mean_gap,
    linestyle="--",
    linewidth=1.2,
    label=f"Mean = {mean_gap:.3f} nm",
)

ax.set_xlabel(
    "Simulation time (ns)"
)

ax.set_ylabel(
    "Largest dry gap (nm)"
)

ax.set_title(
    "C   Separation from periodic image"
)

ax.legend(
    fontsize=8,
    loc="best",
)

ax.grid(
    alpha=0.2
)


fig.suptitle(
    "FF-1B: Supported Graphene / OPC Wetting Validation\n"
    f"5 ns | ? = {mean_angle:.1f} ± {sd_angle:.1f} deg block variability | "
    f"edge occupancy = {edge_occupancy:.3f} | "
    f"water crossings = {waters_below}",
    fontsize=12,
)


fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)


OUTPUT_PNG.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("PNG:")
print(f"  {OUTPUT_PNG}")

print("PDF:")
print(f"  {OUTPUT_PDF}")

print()
print("FF1B_DEJI_SUMMARY_FIGURE_COMPLETE")
print("=" * 80)
