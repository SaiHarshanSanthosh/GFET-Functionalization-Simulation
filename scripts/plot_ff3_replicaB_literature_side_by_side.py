from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

A = Path("analysis")

INPUT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "literature_density_profile_validation.json"
)

OUT_PNG = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "literature_side_by_side_validation.png"
)

OUT_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "literature_side_by_side_validation.csv"
)

# ============================================================
# LOAD ACTUAL RESULTS FROM OUR TRAJECTORY ANALYSIS
# ============================================================

if not INPUT_JSON.exists():
    raise FileNotFoundError(INPUT_JSON)

data = json.loads(
    INPUT_JSON.read_text()
)

comparisons = {
    x["metric"]: x
    for x in data["literature_comparison"]
}

# Use only quantities that are directly comparable as
# density-profile peak positions.
wanted = [
    "Water O peak 1",
    "Water O peak 2",
    "Cl- dominant peak",
]

labels = [
    "Water O layer 1",
    "Water O layer 2",
    "Cl- interfacial peak",
]

ours = np.array([
    comparisons[name]["our_position_nm"]
    for name in wanted
])

ours_sd = np.array([
    comparisons[name]["our_block_sd_nm"]
    for name in wanted
])

literature = np.array([
    comparisons[name]["literature_position_nm"]
    for name in wanted
])

difference = ours - literature

# ============================================================
# PRINT TABLE
# ============================================================

print("=" * 78)
print("OUR MD vs PUBLISHED GRAPHENE-ELECTROLYTE STRUCTURE")
print("=" * 78)
print()

print(
    f"{'feature':26s} "
    f"{'our MD':>10s} "
    f"{'SD':>10s} "
    f"{'literature':>12s} "
    f"{'difference':>12s}"
)

print("-" * 78)

for label, o, sd, lit, d in zip(
    labels,
    ours,
    ours_sd,
    literature,
    difference,
):
    print(
        f"{label:26s} "
        f"{o:10.3f} "
        f"{sd:10.3f} "
        f"{lit:12.3f} "
        f"{d:+12.3f}"
    )

# ============================================================
# SAVE COMPARISON CSV
# ============================================================

lines = [
    (
        "feature,"
        "our_MD_nm,"
        "our_block_SD_nm,"
        "literature_nm,"
        "difference_nm"
    )
]

for label, o, sd, lit, d in zip(
    labels,
    ours,
    ours_sd,
    literature,
    difference,
):
    lines.append(
        f"{label},"
        f"{o:.6f},"
        f"{sd:.6f},"
        f"{lit:.6f},"
        f"{d:.6f}"
    )

OUT_CSV.write_text(
    "\n".join(lines) + "\n"
)

# ============================================================
# SIDE-BY-SIDE FIGURE
# ============================================================

x = np.arange(
    len(labels)
)

width = 0.36

fig, ax = plt.subplots(
    figsize=(9.5, 5.8)
)

bars_ours = ax.bar(
    x - width / 2,
    ours,
    width,
    yerr=ours_sd,
    capsize=5,
    label="Our MD — Replica B, 10 ns",
)

bars_lit = ax.bar(
    x + width / 2,
    literature,
    width,
    label="2023 Nanoscale Advances",
)

# Numerical labels from actual loaded values
for bar, value in zip(
    bars_ours,
    ours
):
    ax.text(
        bar.get_x()
        +
        bar.get_width() / 2,
        value + 0.020,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

for bar, value in zip(
    bars_lit,
    literature
):
    ax.text(
        bar.get_x()
        +
        bar.get_width() / 2,
        value + 0.020,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.set_ylabel(
    "Distance above graphene carbon plane (nm)"
)

ax.set_title(
    "Graphene–electrolyte interfacial structure:\n"
    "our explicit-DPBS MD vs published simulation"
)

ax.set_ylim(
    0.0,
    max(
        np.max(ours + ours_sd),
        np.max(literature)
    ) + 0.12
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.text(
    0.5,
    0.015,
    (
        "Literature: Hemanth et al., "
        "Nanoscale Advances (2023), "
        "DOI: 10.1039/D2NA00733A"
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        1
    ]
)

fig.savefig(
    OUT_PNG,
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)

print()
print(f"Source data: {INPUT_JSON}")
print(f"CSV:         {OUT_CSV}")
print(f"Figure:      {OUT_PNG}")
print()
print("SIDE-BY-SIDE VALIDATION FIGURE COMPLETE")
