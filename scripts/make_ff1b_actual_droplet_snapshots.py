from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

INPUT = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_yb_300K_500ps_to_5000ps_oxygen_trajectory.npz"
)

OUTPUT_PNG = (
    ROOT
    / "analysis"
    / "figures"
    / "25_ff1b_actual_water_droplet_snapshots_5ns.png"
)

OUTPUT_PDF = (
    ROOT
    / "analysis"
    / "figures"
    / "25_ff1b_actual_water_droplet_snapshots_5ns.pdf"
)


LY_NM = 10.652112467

TARGET_TIMES_PS = [
    500.0,
    2500.0,
    5000.0,
]


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


def recenter_frame(y_raw):

    y = np.mod(
        y_raw,
        LY_NM,
    )

    y_sorted = np.sort(y)

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

    gap_start = float(
        y_sorted[imax]
    )

    shift_origin = (
        gap_start
        +
        0.5 * largest_gap
        +
        0.5 * LY_NM
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

    return (
        y_centered,
        largest_gap,
    )


fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.5),
    sharex=True,
    sharey=True,
)


print()
print("=" * 80)
print("FF-1B ACTUAL WATER-DROPLET SNAPSHOTS")
print("=" * 80)


for ax, target_ps in zip(
    axes,
    TARGET_TIMES_PS,
):

    index = int(
        np.argmin(
            np.abs(
                times - target_ps
            )
        )
    )

    actual_ps = float(
        times[index]
    )

    xyz = oxygen[index]

    y, largest_gap = recenter_frame(
        xyz[:, 1]
    )

    z = (
        xyz[:, 2]
        -
        carbon_plane[index]
    )

    occupied_width = (
        LY_NM
        -
        largest_gap
    )

    ax.scatter(
        y,
        z,
        s=7,
        alpha=0.45,
        linewidths=0,
    )

    ax.axhline(
        0.0,
        linewidth=2.0,
    )

    ax.set_title(
        f"{actual_ps/1000.0:.1f} ns"
    )

    ax.set_xlabel(
        "y position (nm)"
    )

    ax.set_xlim(
        -0.5 * LY_NM,
        0.5 * LY_NM,
    )

    ax.set_ylim(
        -0.15,
        3.2,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.text(
        0.03,
        0.95,
        f"dry gap = {largest_gap:.2f} nm",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    print(
        f"{actual_ps:7.1f} ps | "
        f"waters = {len(z)} | "
        f"highest O = {np.max(z):.3f} nm | "
        f"occupied width = {occupied_width:.3f} nm | "
        f"dry gap = {largest_gap:.3f} nm"
    )


axes[0].set_ylabel(
    "Height above graphene (nm)"
)


fig.suptitle(
    "Actual OPC Water Droplet on Supported Graphene\n"
    "Each point is one simulated water oxygen; graphene plane at z = 0",
    fontsize=12,
)


fig.tight_layout(
    rect=[0, 0, 1, 0.90]
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
print("FF1B_ACTUAL_DROPLET_SNAPSHOTS_COMPLETE")
print("=" * 80)
