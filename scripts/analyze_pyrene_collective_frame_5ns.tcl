from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT = Path(
    "analysis/pyrene_collective_frame_5ns.csv"
)

OUTPUT_CSV = Path(
    "analysis/pyrene_msd_5ns.csv"
)

OUTPUT_FIGURE = Path(
    "analysis/figures/14_pyrene_5ns_msd.png"
)

OUTPUT_FIGURE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD TRAJECTORY
# ============================================================

df = pd.read_csv(INPUT)

required = [
    "time_ps",
    "x_relative_nm",
    "y_relative_nm",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


time_ps = df["time_ps"].to_numpy(dtype=float)
x_nm = df["x_relative_nm"].to_numpy(dtype=float)
y_nm = df["y_relative_nm"].to_numpy(dtype=float)


if len(df) < 100:
    raise RuntimeError(
        "Trajectory unexpectedly short."
    )


dt_ps = float(
    np.median(
        np.diff(time_ps)
    )
)


print()
print("=" * 72)
print("PYRENE 2D MULTI-TIME-ORIGIN MSD ANALYSIS")
print("=" * 72)

print(f"Samples:             {len(df)}")
print(f"Sampling interval:   {dt_ps:.3f} ps")
print(
    f"Trajectory duration: "
    f"{(time_ps[-1] - time_ps[0]) / 1000.0:.3f} ns"
)


# ============================================================
# MULTI-TIME-ORIGIN MSD
#
# MSD(tau) =
# < |r(t + tau) - r(t)|^2 >_t
#
# Limit maximum lag to half the trajectory.
# Longer lags have too few independent time origins and
# become increasingly noisy.
# ============================================================

n = len(df)

max_lag_frames = n // 2

rows = []


for lag in range(
    1,
    max_lag_frames + 1,
):

    dx = (
        x_nm[lag:]
        -
        x_nm[:-lag]
    )

    dy = (
        y_nm[lag:]
        -
        y_nm[:-lag]
    )

    squared_displacements = (
        dx * dx
        +
        dy * dy
    )


    msd_nm2 = float(
        np.mean(
            squared_displacements
        )
    )


    sd_nm2 = float(
        np.std(
            squared_displacements,
            ddof=1,
        )
    )


    # Descriptive standard error only.
    # Overlapping time origins are correlated, so this is
    # NOT treated as a rigorous statistical uncertainty.
    sem_nm2 = float(
        sd_nm2
        /
        np.sqrt(
            len(
                squared_displacements
            )
        )
    )


    lag_ps = (
        lag
        *
        dt_ps
    )

    lag_ns = (
        lag_ps
        /
        1000.0
    )


    rows.append(
        {
            "lag_frames": lag,
            "lag_ps": lag_ps,
            "lag_ns": lag_ns,
            "n_time_origins": len(
                squared_displacements
            ),
            "msd_nm2": msd_nm2,
            "sd_squared_displacement_nm2": sd_nm2,
            "descriptive_sem_nm2": sem_nm2,
        }
    )


msd = pd.DataFrame(
    rows
)

msd.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# REPORT MSD AT SELECTED LAG TIMES
# ============================================================

requested_lags_ns = [
    0.010,
    0.020,
    0.050,
    0.100,
    0.200,
    0.500,
    1.000,
    1.500,
    2.000,
]


print()
print("SELECTED MSD VALUES")
print("-------------------")


for requested in requested_lags_ns:

    if requested > msd["lag_ns"].iloc[-1]:
        continue

    idx = int(
        np.argmin(
            np.abs(
                msd["lag_ns"].to_numpy()
                -
                requested
            )
        )
    )

    row = msd.iloc[idx]

    print(
        f"tau = {row['lag_ns']:6.3f} ns   "
        f"MSD = {row['msd_nm2']:9.5f} nm^2   "
        f"origins = {int(row['n_time_origins'])}"
    )


# ============================================================
# LOG-LOG SCALING DIAGNOSTIC
#
# If MSD ~ tau^alpha:
#
# alpha ~ 1 : normal diffusion
# alpha < 1 : subdiffusive / constrained behavior
# alpha > 1 : superdiffusive / directed behavior
#
# These are diagnostics only.
# We are NOT calculating D yet.
# ============================================================

windows = [
    (0.020, 0.200),
    (0.050, 0.500),
    (0.100, 1.000),
    (0.500, 2.000),
]


scaling_results = []


print()
print("LOG-LOG SCALING DIAGNOSTICS")
print("---------------------------")


for lower_ns, upper_ns in windows:

    mask = (
        (msd["lag_ns"] >= lower_ns)
        &
        (msd["lag_ns"] <= upper_ns)
        &
        (msd["msd_nm2"] > 0)
    )


    subset = msd.loc[
        mask
    ]


    if len(subset) < 5:

        continue


    log_t = np.log(
        subset["lag_ns"].to_numpy()
    )

    log_msd = np.log(
        subset["msd_nm2"].to_numpy()
    )


    slope, intercept = np.polyfit(
        log_t,
        log_msd,
        1,
    )


    predicted = (
        slope * log_t
        +
        intercept
    )


    ss_res = float(
        np.sum(
            (
                log_msd
                -
                predicted
            ) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                log_msd
                -
                np.mean(log_msd)
            ) ** 2
        )
    )


    if ss_tot > 0:

        r2 = (
            1.0
            -
            ss_res / ss_tot
        )

    else:

        r2 = np.nan


    scaling_results.append(
        {
            "lower_ns": lower_ns,
            "upper_ns": upper_ns,
            "alpha": float(slope),
            "r2": float(r2),
        }
    )


    print(
        f"{lower_ns:5.3f} - "
        f"{upper_ns:5.3f} ns:   "
        f"alpha = {slope:7.4f}   "
        f"R^2 = {r2:7.4f}"
    )


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.5),
)


# ------------------------------------------------------------
# A. Linear MSD
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    msd["lag_ns"],
    msd["msd_nm2"],
    linewidth=2,
)

ax.set_xlabel(
    "Lag time (ns)"
)

ax.set_ylabel(
    r"2D MSD (nm$^2$)"
)

ax.set_title(
    "A. Multi-time-origin lateral MSD"
)

ax.grid(
    alpha=0.25
)


# ------------------------------------------------------------
# B. Log-log MSD
# ------------------------------------------------------------

ax = axes[1]

ax.loglog(
    msd["lag_ns"],
    msd["msd_nm2"],
    linewidth=2,
    label="Pyrene MSD",
)


# Unit-slope reference line.
reference_mask = (
    (msd["lag_ns"] >= 0.05)
    &
    (msd["lag_ns"] <= 1.0)
)

reference_subset = msd.loc[
    reference_mask
]


if len(reference_subset) > 0:

    reference_index = (
        len(reference_subset)
        //
        2
    )

    reference_row = (
        reference_subset.iloc[
            reference_index
        ]
    )

    ref_tau = float(
        reference_row["lag_ns"]
    )

    ref_msd = float(
        reference_row["msd_nm2"]
    )

    reference_line = (
        ref_msd
        *
        (
            msd["lag_ns"]
            /
            ref_tau
        )
    )

    ax.loglog(
        msd["lag_ns"],
        reference_line,
        linestyle="--",
        linewidth=1.5,
        label=r"Reference: MSD $\propto \tau$",
    )


ax.set_xlabel(
    "Lag time (ns)"
)

ax.set_ylabel(
    r"2D MSD (nm$^2$)"
)

ax.set_title(
    "B. MSD scaling"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend(
    frameon=False
)


fig.suptitle(
    "Pyrene Lateral Mean-Squared Displacement on Graphene",
    fontsize=17,
)


fig.tight_layout()


fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 72)
print("MSD ANALYSIS COMPLETE")
print("=" * 72)

print()
print(f"CSV:    {OUTPUT_CSV}")
print(f"Figure: {OUTPUT_FIGURE}")

print()
print(
    "NOTE: No diffusion coefficient has been assigned yet."
)

print(
    "We first inspect whether a genuine linear/diffusive "
    "MSD regime exists."
)

print()
print("PYRENE_5NS_MSD_PASS")