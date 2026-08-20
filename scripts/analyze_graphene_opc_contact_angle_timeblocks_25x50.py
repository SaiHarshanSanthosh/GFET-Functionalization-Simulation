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
    / "graphene_opc_wetting_25x50_contact_angle_timeblocks.json"
)

OUTPUT_FIGURE = (
    ROOT
    / "analysis"
    / "figures"
    / "21_graphene_opc_wetting_25x50_contact_angle_timeblocks.png"
)


LX_NM = 6.150000000
LY_NM = 10.652112467

DY_NM = 0.05
DZ_NM = 0.05
Z_MAX_NM = 4.0

SMOOTH_SIGMA_BINS = 1.0

BULK_Y_ABS_MAX_NM = 0.75
BULK_Z_MIN_NM = 0.80
BULK_Z_MAX_NM = 1.60

DENSITY_FRACTION = 0.50
FIT_Z_MIN_NM = 0.75


BLOCKS_PS = [
    (300.0, 350.0),
    (350.0, 400.0),
    (400.0, 450.0),
    (450.0, 500.0),
]


print()
print("=" * 80)
print("GRAPHENE / OPC CONTACT-ANGLE TEMPORAL CONVERGENCE")
print("=" * 80)


data = np.load(INPUT)

times_all = np.asarray(
    data["times_ps"],
    dtype=float,
)

oxygen_all = np.asarray(
    data["oxygen_positions_nm"],
    dtype=float,
)

carbon_all = np.asarray(
    data["carbon_plane_nm"],
    dtype=float,
)


def recenter_block(
    oxygen,
    carbon_plane,
):

    all_y = []
    all_z = []

    for frame in range(len(oxygen)):

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

    return (
        np.concatenate(all_y),
        np.concatenate(all_z),
    )


def build_density(
    y,
    z,
    n_frames,
):

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
        y,
        z,
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

    density = gaussian_filter(
        density,
        sigma=SMOOTH_SIGMA_BINS,
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

    return (
        density,
        y_centers,
        z_centers,
    )


def estimate_bulk_density(
    density,
    y_centers,
    z_centers,
):

    y_mask = (
        np.abs(y_centers)
        <=
        BULK_Y_ABS_MAX_NM
    )

    z_mask = (
        (z_centers >= BULK_Z_MIN_NM)
        &
        (z_centers <= BULK_Z_MAX_NM)
    )

    region = density[
        np.ix_(
            y_mask,
            z_mask,
        )
    ]

    return float(
        np.median(region)
    )


def extract_interface(
    density,
    y_centers,
    z_centers,
    level,
):

    fig_tmp, ax_tmp = plt.subplots()

    contour = ax_tmp.contour(
        y_centers,
        z_centers,
        density.T,
        levels=[level],
    )

    segments = contour.allsegs[0]

    plt.close(fig_tmp)

    if len(segments) == 0:
        raise RuntimeError(
            "No liquid-vapor contour found."
        )

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

    y0 = float(solution[0])
    z0 = float(solution[1])
    c = float(solution[2])

    radius_sq = (
        c
        +
        y0*y0
        +
        z0*z0
    )

    if radius_sq <= 0.0:
        raise RuntimeError(
            "Invalid circle radius."
        )

    radius = float(
        np.sqrt(radius_sq)
    )

    radial = np.sqrt(
        (y-y0)**2
        +
        (z-z0)**2
    )

    rms = float(
        np.sqrt(
            np.mean(
                (radial-radius)**2
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
            "Circle does not intersect graphene plane."
        )

    theta = float(
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

    return {
        "contact_angle_deg": theta,
        "y0_nm": y0,
        "z0_nm": z0,
        "radius_nm": radius,
        "rms_residual_nm": rms,
        "n_fit_points": int(
            len(points)
        ),
    }


results = []


print()
print(
    "block_ps       frames   rho_bulk    angle_deg   radius_nm   RMS_nm"
)
print("-" * 80)


for i, (start_ps, end_ps) in enumerate(
    BLOCKS_PS
):

    if i < len(BLOCKS_PS) - 1:

        mask = (
            (times_all >= start_ps)
            &
            (times_all < end_ps)
        )

    else:

        mask = (
            (times_all >= start_ps)
            &
            (times_all <= end_ps)
        )


    times = times_all[mask]
    oxygen = oxygen_all[mask]

    if carbon_all.ndim == 0:

        carbon = np.full(
            len(times),
            float(carbon_all),
            dtype=float,
        )

    else:

        carbon = carbon_all[mask]


    y, z = recenter_block(
        oxygen,
        carbon,
    )


    density, y_centers, z_centers = build_density(
        y,
        z,
        len(times),
    )


    rho_bulk = estimate_bulk_density(
        density,
        y_centers,
        z_centers,
    )


    level = (
        DENSITY_FRACTION
        *
        rho_bulk
    )


    interface = extract_interface(
        density,
        y_centers,
        z_centers,
        level,
    )


    fit_points = interface[
        interface[:, 1]
        >=
        FIT_Z_MIN_NM
    ]


    if len(fit_points) < 10:

        raise RuntimeError(
            f"Too few fit points in "
            f"{start_ps}-{end_ps} ps block."
        )


    fit = circle_fit(
        fit_points
    )


    fit.update(
        {
            "start_ps": float(start_ps),
            "end_ps": float(end_ps),
            "midpoint_ps": float(
                0.5
                *
                (
                    start_ps
                    +
                    end_ps
                )
            ),
            "frames": int(
                len(times)
            ),
            "bulk_density_nm^-3": float(
                rho_bulk
            ),
            "interface_density_nm^-3": float(
                level
            ),
        }
    )


    results.append(
        fit
    )


    print(
        f"{start_ps:5.0f}-{end_ps:<5.0f}"
        f"      {len(times):3d}"
        f"      {rho_bulk:8.3f}"
        f"      {fit['contact_angle_deg']:9.3f}"
        f"      {fit['radius_nm']:8.4f}"
        f"      {fit['rms_residual_nm']:7.4f}"
    )


angles = np.asarray(
    [
        row["contact_angle_deg"]
        for row in results
    ],
    dtype=float,
)

midpoints = np.asarray(
    [
        row["midpoint_ps"]
        for row in results
    ],
    dtype=float,
)


slope_deg_per_ps = float(
    np.polyfit(
        midpoints,
        angles,
        1,
    )[0]
)


print()
print("=" * 80)
print("TEMPORAL STABILITY SUMMARY")
print("=" * 80)

print(
    f"Mean angle:          "
    f"{np.mean(angles):.3f} deg"
)

print(
    f"SD across blocks:    "
    f"{np.std(angles):.3f} deg"
)

print(
    f"Angle range:         "
    f"{np.min(angles):.3f}"
    f" -> "
    f"{np.max(angles):.3f} deg"
)

print(
    f"Total span:          "
    f"{np.ptp(angles):.3f} deg"
)

print(
    f"Linear time slope:   "
    f"{1000.0*slope_deg_per_ps:+.3f} deg/ns"
)


fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

ax.plot(
    midpoints,
    angles,
    marker="o",
)

ax.axhline(
    91.970,
    linestyle="--",
    label="300-500 ps full-window fit",
)

ax.set_xlabel(
    "Block midpoint (ps)"
)

ax.set_ylabel(
    "Contact angle (deg)"
)

ax.set_title(
    "Graphene-OPC Contact Angle: Temporal Stability"
)

ax.legend()

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


metadata = {
    "input": str(INPUT),

    "density_fraction": float(
        DENSITY_FRACTION
    ),

    "fit_z_min_nm": float(
        FIT_Z_MIN_NM
    ),

    "blocks": results,

    "summary": {
        "mean_angle_deg": float(
            np.mean(angles)
        ),

        "sd_angle_deg": float(
            np.std(angles)
        ),

        "min_angle_deg": float(
            np.min(angles)
        ),

        "max_angle_deg": float(
            np.max(angles)
        ),

        "span_angle_deg": float(
            np.ptp(angles)
        ),

        "linear_slope_deg_per_ns": float(
            1000.0
            *
            slope_deg_per_ps
        ),
    },

    "note": (
        "Temporal convergence diagnostic for the nanoscale "
        "cylindrical-droplet pilot contact angle. "
        "Each block independently estimates density and fits "
        "its liquid-vapor interface."
    ),
}


OUTPUT_JSON.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("Figure:")
print(f"  {OUTPUT_FIGURE}")

print("JSON:")
print(f"  {OUTPUT_JSON}")

print()

print(
    "GRAPHENE_OPC_CONTACT_ANGLE_TIMEBLOCKS_COMPLETE"
)

print("=" * 80)