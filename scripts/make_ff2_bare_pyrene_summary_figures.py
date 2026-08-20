from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HEIGHT_CSV = Path(
    "analysis/ff2_bare_pyrene_graphene_height_scan.csv"
)

REGISTRY_CSV = Path(
    "analysis/ff2_bare_pyrene_graphene_registry_scan.csv"
)

FIGDIR = Path(
    "analysis/figures"
)

FIGDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIGURE 26 — HEIGHT SCAN
# ============================================================

height = pd.read_csv(
    HEIGHT_CSV
)

z = height[
    "distance_A"
].to_numpy(float)

lj = height[
    "delta_LJ_kJmol"
].to_numpy(float)

coul = height[
    "delta_Coulomb_kJmol"
].to_numpy(float)

total = height[
    "delta_total_kJmol"
].to_numpy(float)


imin = int(
    np.argmin(total)
)

z_grid = float(
    z[imin]
)

E_grid = float(
    total[imin]
)


# Local quadratic interpolation using minimum +/- 2 points.
lo = max(
    0,
    imin - 2
)

hi = min(
    len(z),
    imin + 3
)

coef = np.polyfit(
    z[lo:hi],
    total[lo:hi],
    2
)

a, b, c = coef

z_quad = float(
    -b / (2.0 * a)
)

E_quad = float(
    np.polyval(
        coef,
        z_quad
    )
)


fig, ax = plt.subplots(
    figsize=(9.5, 6.2)
)

ax.plot(
    z,
    total,
    linewidth=2.4,
    label="Total interaction"
)

ax.plot(
    z,
    lj,
    linewidth=1.8,
    linestyle="--",
    label="Lennard-Jones"
)

ax.plot(
    z,
    coul,
    linewidth=1.6,
    linestyle=":",
    label="Coulomb"
)

ax.axhline(
    0.0,
    linewidth=1.0
)

ax.scatter(
    [z_grid],
    [E_grid],
    s=75,
    zorder=5,
    label=(
        f"Grid minimum: "
        f"{z_grid:.2f} A, "
        f"{E_grid:.1f} kJ/mol"
    )
)

ax.scatter(
    [z_quad],
    [E_quad],
    marker="x",
    s=90,
    linewidth=2.0,
    zorder=6,
    label=(
        f"Quadratic estimate: "
        f"{z_quad:.3f} A"
    )
)

ax.set_xlabel(
    "Pyrene carbon-plane distance from graphene (A)"
)

ax.set_ylabel(
    "Interaction energy relative to 50 A (kJ/mol)"
)

ax.set_title(
    "FF-2: Bare Pyrene / IFF Graphene Rigid Adsorption Scan"
)

ax.grid(
    linestyle="--",
    alpha=0.3
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIGDIR
    / "26_ff2_bare_pyrene_graphene_height_scan.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIGDIR
    / "26_ff2_bare_pyrene_graphene_height_scan.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 27 — REGISTRY / ROTATION CORRUGATION
# ============================================================

reg = pd.read_csv(
    REGISTRY_CSV
)

global_min = float(
    reg[
        "best_total_kJmol"
    ].min()
)

reg[
    "relative_energy_kJmol"
] = (
    reg[
        "best_total_kJmol"
    ]
    - global_min
)


fig, ax = plt.subplots(
    figsize=(9.5, 6.2)
)


rotations = sorted(
    reg[
        "rotation_deg"
    ].unique()
)

for rotation in rotations:

    subset = reg[
        np.isclose(
            reg["rotation_deg"],
            rotation
        )
    ]

    # Small deterministic horizontal offsets based on registry.
    offsets = (
        (
            subset["u_fraction"].to_numpy()
            - 0.4
        )
        * 0.18
        +
        (
            subset["v_fraction"].to_numpy()
            - 0.4
        )
        * 0.07
    )

    x = (
        np.full(
            len(subset),
            rotation,
            dtype=float
        )
        + offsets
    )

    ax.scatter(
        x,
        subset[
            "relative_energy_kJmol"
        ],
        s=28,
        alpha=0.7
    )

    ax.plot(
        [rotation],
        [
            subset[
                "relative_energy_kJmol"
            ].mean()
        ],
        marker="D",
        markersize=7
    )


spread = float(
    reg[
        "best_total_kJmol"
    ].max()
    -
    reg[
        "best_total_kJmol"
    ].min()
)

sd = float(
    reg[
        "best_total_kJmol"
    ].std(ddof=1)
)


ax.axhline(
    0.0,
    linewidth=1.0
)

ax.set_xticks(
    rotations
)

ax.set_xlabel(
    "In-plane pyrene rotation (degrees)"
)

ax.set_ylabel(
    "Energy above global minimum (kJ/mol)"
)

ax.set_title(
    "FF-2: Pyrene Registry / Orientation Energy Corrugation"
)

ax.text(
    0.02,
    0.96,
    (
        f"125 optimized configurations\n"
        f"Full spread = {spread:.3f} kJ/mol\n"
        f"SD = {sd:.3f} kJ/mol"
    ),
    transform=ax.transAxes,
    va="top"
)

ax.grid(
    linestyle="--",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    FIGDIR
    / "27_ff2_bare_pyrene_graphene_registry_orientation.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIGDIR
    / "27_ff2_bare_pyrene_graphene_registry_orientation.pdf",
    bbox_inches="tight"
)

plt.close(fig)


print()
print("FF-2 FIGURE SUMMARY")
print("-------------------")
print(
    f"Grid height minimum:       "
    f"{z_grid:.3f} A, "
    f"{E_grid:.3f} kJ/mol"
)
print(
    f"Quadratic height estimate: "
    f"{z_quad:.4f} A, "
    f"{E_quad:.3f} kJ/mol"
)
print(
    f"Registry energy spread:    "
    f"{spread:.3f} kJ/mol"
)
print(
    f"Registry energy SD:        "
    f"{sd:.3f} kJ/mol"
)
print()
print(
    "Saved Figure 26 PNG/PDF"
)
print(
    "Saved Figure 27 PNG/PDF"
)
print()
print(
    "FF2_SUMMARY_FIGURES_COMPLETE"
)
