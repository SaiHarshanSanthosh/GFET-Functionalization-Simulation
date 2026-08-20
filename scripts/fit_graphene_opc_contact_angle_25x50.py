from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

INPUT = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_yb_300K_500ps_oxygen_trajectory.npz"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_contact_angle_pilot.json"
)

OUTPUT_FIGURE = (
    ROOT
    / "analysis"
    / "figures"
    / "20_graphene_opc_wetting_25x50_contact_angle_pilot.png"
)


# ------------------------------------------------------------
# Geometry / analysis settings
# ------------------------------------------------------------

LX_NM = 6.150000000
LY_NM = 10.652112467

START_PS = 300.0

DY_NM = 0.05
DZ_NM = 0.05

Z_MAX_NM = 4.0

# Smooth density slightly before extracting contour.
SMOOTH_SIGMA_BINS = 1.0

# Region expected to behave approximately like bulk liquid.
BULK_Y_ABS_MAX_NM = 0.75
BULK_Z_MIN_NM = 0.80
BULK_Z_MAX_NM = 1.60

# Main pilot definition.
MAIN_DENSITY_FRACTION = 0.50
MAIN_FIT_Z_MIN_NM = 0.75

# Sensitivity analysis:
# change both the liquid-vapor density definition and
# how much near-graphene layering is excluded.
DENSITY_FRACTIONS = [
    0.45,
    0.50,
    0.55,
]

FIT_Z_MINS_NM = [
    0.60,
    0.75,
    0.90,
]


print()
print("=" * 80)
print("GRAPHENE / OPC 25x50 PILOT CONTACT ANGLE")
print("=" * 80)


# ------------------------------------------------------------
# Load trajectory
# ------------------------------------------------------------

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

if carbon_plane.ndim == 0:
    carbon_plane = np.full(
        len(times),
        float(carbon_plane),
        dtype=float,
    )
else:
    carbon_plane = carbon_plane[mask]


n_frames = len(times)
n_waters = oxygen.shape[1]


print(f"Frames:              {n_frames}")
print(
    f"Interval:            "
    f"{times[0]:.1f} -> {times[-1]:.1f} ps"
)
print(f"Waters:              {n_waters}")
print(f"Lx:                  {LX_NM:.6f} nm")
print(f"Ly:                  {LY_NM:.6f} nm")


# ------------------------------------------------------------
# Recenter every frame using largest periodic dry gap
#
# The dry-gap midpoint is moved to the periodic boundary.
# Therefore the droplet itself sits near y = 0.
# ------------------------------------------------------------

all_y = []
all_z = []

for frame in range(n_frames):

    y = np.mod(
        oxygen[frame, :, 1],
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

    gap_start = float(
        y_sorted[imax]
    )

    largest_gap = float(
        gaps[imax]
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

    z_relative = (
        oxygen[frame, :, 2]
        -
        carbon_plane[frame]
    )

    all_y.append(
        y_centered
    )

    all_z.append(
        z_relative
    )


all_y = np.concatenate(
    all_y
)

all_z = np.concatenate(
    all_z
)


# ------------------------------------------------------------
# 2D oxygen number density
# ------------------------------------------------------------

y_edges = np.arange(
    -0.5 * LY_NM,
    0.5 * LY_NM + DY_NM,
    DY_NM,
)

z_edges = np.arange(
    0.0,
    Z_MAX_NM + DZ_NM,
    DZ_NM,
)


counts, _, _ = np.histogram2d(
    all_y,
    all_z,
    bins=[
        y_edges,
        z_edges,
    ],
)


bin_volume_nm3 = (
    LX_NM
    *
    DY_NM
    *
    DZ_NM
)

density = (
    counts
    /
    (
        n_frames
        *
        bin_volume_nm3
    )
)


density_smooth = gaussian_filter(
    density,
    sigma=SMOOTH_SIGMA_BINS,
)


y_centers = 0.5 * (
    y_edges[:-1]
    +
    y_edges[1:]
)

z_centers = 0.5 * (
    z_edges[:-1]
    +
    z_edges[1:]
)


# ------------------------------------------------------------
# Estimate bulk-like oxygen density from center of droplet
# ------------------------------------------------------------

bulk_y_mask = (
    np.abs(y_centers)
    <=
    BULK_Y_ABS_MAX_NM
)

bulk_z_mask = (
    (z_centers >= BULK_Z_MIN_NM)
    &
    (z_centers <= BULK_Z_MAX_NM)
)

bulk_region = density_smooth[
    np.ix_(
        bulk_y_mask,
        bulk_z_mask,
    )
]

rho_bulk = float(
    np.median(
        bulk_region
    )
)


print()
print("=" * 80)
print("BULK-LIKE DENSITY")
print("=" * 80)

print(
    f"Estimated oxygen density: "
    f"{rho_bulk:.4f} nm^-3"
)

print(
    f"Main interface density:    "
    f"{MAIN_DENSITY_FRACTION*rho_bulk:.4f} nm^-3"
)


# ------------------------------------------------------------
# Circle fitting
# ------------------------------------------------------------

def extract_interface(level):

    fig_tmp, ax_tmp = plt.subplots()

    contour = ax_tmp.contour(
        y_centers,
        z_centers,
        density_smooth.T,
        levels=[level],
    )

    segments = contour.allsegs[0]

    plt.close(fig_tmp)

    if len(segments) == 0:
        raise RuntimeError(
            "No density contour found."
        )

    # Prefer the physically large droplet contour.
    segment = max(
        segments,
        key=lambda arr: (
            np.ptp(arr[:, 0])
            *
            np.ptp(arr[:, 1])
        ),
    )

    return np.asarray(
        segment,
        dtype=float,
    )


def circle_fit(points):

    y = points[:, 0]
    z = points[:, 1]

    # Circle:
    #
    # (y-y0)^2 + (z-z0)^2 = R^2
    #
    # Rearranged into a linear least-squares problem.
    A = np.column_stack(
        [
            2.0 * y,
            2.0 * z,
            np.ones_like(y),
        ]
    )

    b = (
        y*y
        +
        z*z
    )

    solution, _, _, _ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    y0 = float(
        solution[0]
    )

    z0 = float(
        solution[1]
    )

    c = float(
        solution[2]
    )

    radius_sq = (
        c
        +
        y0*y0
        +
        z0*z0
    )

    if radius_sq <= 0.0:
        raise RuntimeError(
            "Invalid fitted circle radius."
        )

    radius = float(
        np.sqrt(
            radius_sq
        )
    )


    radial_distance = np.sqrt(
        (y-y0)**2
        +
        (z-z0)**2
    )

    rms_residual = float(
        np.sqrt(
            np.mean(
                (
                    radial_distance
                    -
                    radius
                )**2
            )
        )
    )


    ratio = (
        -z0
        /
        radius
    )

    if (
        ratio < -1.0
        or
        ratio > 1.0
    ):
        raise RuntimeError(
            "Fitted circle does not intersect graphene plane."
        )

    theta_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    ratio,
                    -1.0,
                    1.0,
                )
            )
        )
    )


    base_half_width = float(
        np.sqrt(
            max(
                radius*radius
                -
                z0*z0,
                0.0,
            )
        )
    )


    droplet_height = float(
        z0
        +
        radius
    )


    return {
        "y0_nm": y0,
        "z0_nm": z0,
        "radius_nm": radius,
        "contact_angle_deg": theta_deg,
        "base_half_width_nm": base_half_width,
        "fitted_height_nm": droplet_height,
        "rms_circle_residual_nm": rms_residual,
        "n_fit_points": int(
            len(points)
        ),
    }


def perform_fit(
    density_fraction,
    fit_z_min_nm,
):

    level = (
        density_fraction
        *
        rho_bulk
    )

    interface = extract_interface(
        level
    )

    fit_mask = (
        interface[:, 1]
        >=
        fit_z_min_nm
    )

    points = interface[
        fit_mask
    ]

    if len(points) < 10:
        raise RuntimeError(
            "Too few interface points after z exclusion."
        )

    result = circle_fit(
        points
    )

    result[
        "density_fraction"
    ] = float(
        density_fraction
    )

    result[
        "interface_density_nm^-3"
    ] = float(
        level
    )

    result[
        "fit_z_min_nm"
    ] = float(
        fit_z_min_nm
    )

    result[
        "interface_points_total"
    ] = int(
        len(interface)
    )

    return (
        result,
        interface,
        points,
    )


# ------------------------------------------------------------
# Main fit
# ------------------------------------------------------------

main_result, main_interface, main_points = perform_fit(
    MAIN_DENSITY_FRACTION,
    MAIN_FIT_Z_MIN_NM,
)


print()
print("=" * 80)
print("MAIN PILOT CONTACT ANGLE")
print("=" * 80)

print(
    f"Density fraction:    "
    f"{MAIN_DENSITY_FRACTION:.2f}"
)

print(
    f"Excluded below:      "
    f"{MAIN_FIT_Z_MIN_NM:.2f} nm"
)

print(
    f"Circle center y:     "
    f"{main_result['y0_nm']:.4f} nm"
)

print(
    f"Circle center z:     "
    f"{main_result['z0_nm']:.4f} nm"
)

print(
    f"Circle radius:       "
    f"{main_result['radius_nm']:.4f} nm"
)

print(
    f"Contact angle:       "
    f"{main_result['contact_angle_deg']:.3f} deg"
)

print(
    f"Fitted height:       "
    f"{main_result['fitted_height_nm']:.4f} nm"
)

print(
    f"Base full width:     "
    f"{2.0*main_result['base_half_width_nm']:.4f} nm"
)

print(
    f"Circle RMS residual: "
    f"{main_result['rms_circle_residual_nm']:.4f} nm"
)

print(
    f"Fit points:          "
    f"{main_result['n_fit_points']}"
)


# ------------------------------------------------------------
# Sensitivity analysis
# ------------------------------------------------------------

sensitivity = []


print()
print("=" * 80)
print("FIT-SENSITIVITY CHECK")
print("=" * 80)

print(
    "rho_frac   z_min_nm   angle_deg   radius_nm   RMS_nm"
)


for fraction in DENSITY_FRACTIONS:

    for z_min in FIT_Z_MINS_NM:

        try:

            result, _, _ = perform_fit(
                fraction,
                z_min,
            )

            sensitivity.append(
                result
            )

            print(
                f"{fraction:8.2f}"
                f"   {z_min:8.2f}"
                f"   {result['contact_angle_deg']:9.3f}"
                f"   {result['radius_nm']:9.4f}"
                f"   {result['rms_circle_residual_nm']:7.4f}"
            )

        except Exception as exc:

            print(
                f"{fraction:8.2f}"
                f"   {z_min:8.2f}"
                f"   FAILED: {exc}"
            )


valid_angles = np.asarray(
    [
        row["contact_angle_deg"]
        for row in sensitivity
    ],
    dtype=float,
)


print()

if len(valid_angles) > 0:

    print(
        f"Sensitivity angle range: "
        f"{np.min(valid_angles):.3f}"
        f" -> "
        f"{np.max(valid_angles):.3f} deg"
    )

    print(
        f"Sensitivity mean +/- SD: "
        f"{np.mean(valid_angles):.3f}"
        f" +/- "
        f"{np.std(valid_angles):.3f} deg"
    )


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


mesh = ax.pcolormesh(
    y_edges,
    z_edges,
    density_smooth.T,
    shading="auto",
)


fig.colorbar(
    mesh,
    ax=ax,
    label="OPC oxygen number density (nm$^{-3}$)",
)


ax.plot(
    main_interface[:, 0],
    main_interface[:, 1],
    linewidth=2.0,
    label="50% bulk-density contour",
)


ax.scatter(
    main_points[:, 0],
    main_points[:, 1],
    s=10,
    label="Points used for circle fit",
)


phi = np.linspace(
    0.0,
    2.0*np.pi,
    600,
)

circle_y = (
    main_result["y0_nm"]
    +
    main_result["radius_nm"]
    *
    np.cos(phi)
)

circle_z = (
    main_result["z0_nm"]
    +
    main_result["radius_nm"]
    *
    np.sin(phi)
)


circle_mask = (
    circle_z >= 0.0
)

ax.plot(
    circle_y[circle_mask],
    circle_z[circle_mask],
    linestyle="--",
    linewidth=2.0,
    label="Fitted circular cap",
)


ax.axhline(
    0.0,
    linewidth=1.5,
    label="Graphene carbon plane",
)


ax.set_xlim(
    -3.5,
    3.5,
)

ax.set_ylim(
    0.0,
    3.5,
)


ax.set_xlabel(
    "Periodic y relative to droplet center (nm)"
)

ax.set_ylabel(
    "Height above graphene carbon plane (nm)"
)

ax.set_title(
    "Graphene-OPC 25x50 Pilot Contact-Angle Fit\n"
    f"theta = {main_result['contact_angle_deg']:.2f} deg"
)

ax.legend(
    loc="upper right",
    fontsize=8,
)


fig.tight_layout()


OUTPUT_FIGURE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

results = {
    "input": str(
        INPUT
    ),

    "analysis_interval_ps": [
        float(times[0]),
        float(times[-1]),
    ],

    "frames": int(
        n_frames
    ),

    "waters": int(
        n_waters
    ),

    "Lx_nm": float(
        LX_NM
    ),

    "Ly_nm": float(
        LY_NM
    ),

    "bulk_density_nm^-3": float(
        rho_bulk
    ),

    "main_fit": main_result,

    "sensitivity": sensitivity,

    "note": (
        "Pilot nanoscale cylindrical-droplet contact angle. "
        "Not yet a macroscopic contact angle. "
        "Near-surface density layering is excluded from the circle fit, "
        "and density-threshold / exclusion-height sensitivity is reported."
    ),

    "figure": str(
        OUTPUT_FIGURE
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
print("=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    f"Figure:\n  {OUTPUT_FIGURE}"
)

print(
    f"JSON:\n  {OUTPUT_JSON}"
)

print()

print(
    "GRAPHENE_OPC_CONTACT_ANGLE_PILOT_COMPLETE"
)

print("=" * 80)