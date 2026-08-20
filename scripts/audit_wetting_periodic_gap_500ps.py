from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

INPUT = (
    ROOT / "analysis"
    / "graphene_opc_wetting_yb_300K_500ps_oxygen_trajectory.npz"
)

OUTPUT_JSON = (
    ROOT / "analysis"
    / "graphene_opc_wetting_periodic_gap_300to500ps.json"
)

OUTPUT_FIGURE = (
    ROOT / "analysis" / "figures"
    / "18_graphene_opc_wetting_periodic_gap_300to500ps.png"
)

LY_NM = 5.32605623
START_PS = 300.0

# density map
DY_NM = 0.05
DZ_NM = 0.05
Z_MAX_NM = 3.0


print()
print("=" * 80)
print("GRAPHENE / OPC PERIODIC DRY-GAP AUDIT")
print("=" * 80)


data = np.load(INPUT)

times = np.asarray(
    data["times_ps"],
    dtype=float,
)

oxygen = np.asarray(
    data["oxygen_positions_nm"],
    dtype=float,
)

carbon_plane = np.asarray(
    data["carbon_plane_nm"],
    dtype=float,
)


mask = times >= START_PS

times = times[mask]
oxygen = oxygen[mask]
carbon_plane = carbon_plane[mask]


largest_gaps = []
occupied_widths = []

all_y_centered = []
all_z = []


for frame in range(len(times)):

    # Wrap every oxygen into one periodic y cell.
    y = np.mod(
        oxygen[frame, :, 1],
        LY_NM,
    )

    y_sorted = np.sort(y)

    # Circular gaps, including across the periodic boundary.
    gaps = np.diff(
        np.concatenate(
            [
                y_sorted,
                [y_sorted[0] + LY_NM],
            ]
        )
    )

    imax = int(
        np.argmax(gaps)
    )

    largest_gap = float(
        gaps[imax]
    )

    occupied_width = (
        LY_NM
        -
        largest_gap
    )

    largest_gaps.append(
        largest_gap
    )

    occupied_widths.append(
        occupied_width
    )


    # Put the largest dry gap at the periodic boundary so
    # the droplet is centered for the averaged density map.
    gap_start = float(
        y_sorted[imax]
    )

    shift_origin = (
        gap_start
        +
        0.5 * largest_gap
    ) % LY_NM


    y_centered = (
        y
        -
        shift_origin
    )

    y_centered -= (
        np.round(
            y_centered / LY_NM
        )
        *
        LY_NM
    )


    z = (
        oxygen[frame, :, 2]
        -
        carbon_plane[frame]
    )


    all_y_centered.append(
        y_centered
    )

    all_z.append(
        z
    )


largest_gaps = np.asarray(
    largest_gaps
)

occupied_widths = np.asarray(
    occupied_widths
)

all_y_centered = np.concatenate(
    all_y_centered
)

all_z = np.concatenate(
    all_z
)


print()
print(
    f"Analysis interval: "
    f"{times[0]:.1f} -> {times[-1]:.1f} ps"
)

print(
    f"Frames:            {len(times)}"
)

print(
    f"Periodic y width:  {LY_NM:.6f} nm"
)


print()
print("=" * 80)
print("PER-FRAME PERIODIC WIDTH")
print("=" * 80)

print(
    f"Mean occupied width:   "
    f"{np.mean(occupied_widths):.6f} nm"
)

print(
    f"Maximum occupied width:"
    f" {np.max(occupied_widths):.6f} nm"
)

print(
    f"Mean largest dry gap:  "
    f"{np.mean(largest_gaps):.6f} nm"
)

print(
    f"Minimum dry gap:       "
    f"{np.min(largest_gaps):.6f} nm"
)

print(
    f"Maximum dry gap:       "
    f"{np.max(largest_gaps):.6f} nm"
)


# ============================================================
# TIME-AVERAGED Y-Z OXYGEN DENSITY
# ============================================================

y_edges = np.arange(
    -0.5*LY_NM,
    0.5*LY_NM + DY_NM,
    DY_NM,
)

z_edges = np.arange(
    0.0,
    Z_MAX_NM + DZ_NM,
    DZ_NM,
)


counts, _, _ = np.histogram2d(
    all_y_centered,
    all_z,
    bins=[
        y_edges,
        z_edges,
    ],
)


# Number per y-z bin per frame.
density2d = (
    counts
    /
    len(times)
)


# Integrated y occupancy.
y_occupancy = np.sum(
    density2d,
    axis=1,
)


# Normalize to central maximum for an easy diagnostic.
if np.max(y_occupancy) > 0:
    y_occupancy_normalized = (
        y_occupancy
        /
        np.max(y_occupancy)
    )
else:
    y_occupancy_normalized = (
        y_occupancy.copy()
    )


y_centers = (
    0.5
    *
    (
        y_edges[:-1]
        +
        y_edges[1:]
    )
)

z_centers = (
    0.5
    *
    (
        z_edges[:-1]
        +
        z_edges[1:]
    )
)


edge_mask = (
    np.abs(y_centers)
    >
    (
        0.5*LY_NM
        -
        0.30
    )
)


mean_edge_occupancy_fraction = float(
    np.mean(
        y_occupancy_normalized[
            edge_mask
        ]
    )
)


print()
print("=" * 80)
print("TIME-AVERAGED DENSITY CHECK")
print("=" * 80)

print(
    f"Mean normalized water occupancy "
    f"within 0.30 nm of periodic y edges: "
    f"{mean_edge_occupancy_fraction:.6f}"
)

print()
print(
    "Interpretation will be based on BOTH the instantaneous "
    "dry-gap statistics and the density map; no arbitrary "
    "pass/fail threshold is imposed by this script."
)


# ============================================================
# FIGURE
# ============================================================

OUTPUT_FIGURE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


mesh = ax.pcolormesh(
    y_edges,
    z_edges,
    density2d.T,
    shading="auto",
)


fig.colorbar(
    mesh,
    ax=ax,
    label="Mean water O count per y-z bin",
)


ax.axhline(
    0.0,
    linewidth=1.5,
)


ax.set_xlabel(
    "Periodic y relative to dry-gap-centered droplet (nm)"
)

ax.set_ylabel(
    "Height above graphene plane (nm)"
)

ax.set_title(
    "Graphene-OPC Wetting: 300-500 ps Periodic-Gap Audit"
)


fig.tight_layout()


fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


results = {
    "analysis_interval_ps": [
        float(times[0]),
        float(times[-1]),
    ],

    "frames": int(
        len(times)
    ),

    "periodic_y_width_nm": (
        LY_NM
    ),

    "occupied_width_nm": {
        "mean": float(
            np.mean(
                occupied_widths
            )
        ),

        "max": float(
            np.max(
                occupied_widths
            )
        ),
    },

    "largest_dry_gap_nm": {
        "mean": float(
            np.mean(
                largest_gaps
            )
        ),

        "min": float(
            np.min(
                largest_gaps
            )
        ),

        "max": float(
            np.max(
                largest_gaps
            )
        ),
    },

    "mean_edge_occupancy_fraction": (
        mean_edge_occupancy_fraction
    ),

    "note": (
        "Diagnostic only. Oxygen-center dry gap and "
        "time-averaged density should be interpreted together."
    ),
}


OUTPUT_JSON.write_text(
    json.dumps(
        results,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    f"Figure:"
    f"\n  {OUTPUT_FIGURE}"
)

print(
    f"JSON:"
    f"\n  {OUTPUT_JSON}"
)

print()
print(
    "GRAPHENE_OPC_PERIODIC_GAP_AUDIT_COMPLETE"
)

print("=" * 80)