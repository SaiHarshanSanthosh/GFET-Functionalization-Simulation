from pathlib import Path
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DROP-01 FORMAL 0 -> 8 ns ADSORPTION ANALYSIS
#
# READ ONLY with respect to MD:
#   - no OpenMM Context
#   - no MD
#   - no minimization
#   - no checkpoint modification
#   - no trajectory modification
#
# PREDECLARED CRITERIA:
#
# Surface contact:
#   pyrene height <= 0.45 nm
#   sustained >= 20 ps
#
# Stable face-on adsorption:
#   pyrene height <= 0.45 nm
#   AND tilt <= 30 deg
#   sustained >= 50 ps
# ============================================================


LOG_FILES = [
    Path(
        "analysis/"
        "drop01_faceon_2p00nm_fixedz0_stageA_20ps_log.csv"
    ),
    Path(
        "analysis/"
        "drop01_faceon_2p00nm_fixedz0_20ps_to_5ns_log.csv"
    ),
    Path(
        "analysis/"
        "drop01_faceon_2p00nm_fixedz0_5ns_to_8ns_log.csv"
    ),
]


OUTPUT_MERGED = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_0to8ns_merged.csv"
)

OUTPUT_SUMMARY = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_0to8ns_adsorption_summary.json"
)

OUTPUT_TRANSITION = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_adsorption_transition_window.csv"
)

OUTPUT_FIGURE = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_0to8ns_height_tilt.png"
)


CONTACT_HEIGHT_NM = 0.45
CONTACT_DURATION_PS = 20.0

FACEON_TILT_DEG = 30.0
STABLE_DURATION_PS = 50.0

TIME_DUPLICATE_TOL_PS = 1.0e-5
VALUE_DUPLICATE_TOL = 1.0e-5

# Largest legitimate reporting gap after 20 ps is 1 ps.
CONTINUITY_GAP_TOL_PS = 1.01


# ============================================================
# 1. LOAD
# ============================================================

all_rows = []
fieldnames = None


for source_index, path in enumerate(LOG_FILES):

    if not path.exists():
        raise RuntimeError(
            f"Missing log file: {path}"
        )

    with open(
        path,
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if fieldnames is None:

            fieldnames = list(
                reader.fieldnames
            )

        elif list(reader.fieldnames) != fieldnames:

            raise RuntimeError(
                f"Header mismatch in {path}"
            )

        for row in reader:

            numeric = {
                key: float(value)
                for key, value in row.items()
            }

            numeric["_source_index"] = (
                source_index
            )

            all_rows.append(
                numeric
            )


if not all_rows:
    raise RuntimeError(
        "No log rows loaded."
    )


print("=" * 82)
print("DROP-01 FORMAL 0 -> 8 ns ADSORPTION ANALYSIS")
print("=" * 82)
print(
    "Raw rows loaded:",
    len(all_rows)
)


# ============================================================
# 2. SORT + STRICTLY DEDUPLICATE CHECKPOINT BOUNDARIES
# ============================================================

all_rows.sort(
    key=lambda r:
        r["time_ps"]
)


merged = []
duplicate_times = []


def rows_match_except_source(
    a,
    b,
):

    for key in fieldnames:

        av = float(
            a[key]
        )

        bv = float(
            b[key]
        )

        if abs(
            av - bv
        ) > VALUE_DUPLICATE_TOL:

            return False

    return True


for row in all_rows:

    if not merged:

        merged.append(
            row
        )

        continue


    previous = merged[-1]

    dt = (
        row["time_ps"]
        -
        previous["time_ps"]
    )


    if abs(
        dt
    ) <= TIME_DUPLICATE_TOL_PS:

        if not rows_match_except_source(
            previous,
            row,
        ):

            raise RuntimeError(
                "Duplicate checkpoint-boundary rows "
                "do not reproduce the same state at "
                f"t={row['time_ps']:.12f} ps."
            )

        duplicate_times.append(
            row["time_ps"]
        )

        # Keep the earlier copy.
        continue


    if dt < 0:

        raise RuntimeError(
            "Non-monotonic time ordering."
        )


    merged.append(
        row
    )


print(
    "Duplicate boundary rows removed:",
    len(
        duplicate_times
    )
)

print(
    "Duplicate times:",
    [
        round(x, 6)
        for x in duplicate_times
    ],
)

print(
    "Unique merged rows:",
    len(
        merged
    )
)


# ============================================================
# 3. BASIC TIME-AXIS VALIDATION
# ============================================================

times = np.asarray(
    [
        r["time_ps"]
        for r in merged
    ],
    dtype=float,
)

gaps = np.diff(
    times
)


if abs(
    times[0]
) > 1.0e-6:

    raise RuntimeError(
        f"Trajectory does not begin at 0 ps: {times[0]}"
    )


if abs(
    times[-1]
    -
    8000.0
) > 2.0e-6:

    raise RuntimeError(
        "Trajectory does not end at 8000 ps: "
        f"{times[-1]:.12f}"
    )


if np.any(
    gaps <= 0
):

    raise RuntimeError(
        "Merged timeline is not strictly increasing."
    )


if float(
    np.max(
        gaps
    )
) > CONTINUITY_GAP_TOL_PS:

    raise RuntimeError(
        "Unexpected reporting gap detected: "
        f"{np.max(gaps):.6f} ps"
    )


print()
print("TIME AXIS")
print("---------")
print(
    f"Start: {times[0]:.6f} ps"
)
print(
    f"End:   {times[-1]:.6f} ps"
)
print(
    f"Minimum reporting gap: "
    f"{np.min(gaps):.6f} ps"
)
print(
    f"Maximum reporting gap: "
    f"{np.max(gaps):.6f} ps"
)


# ============================================================
# 4. EXTRACT METRICS
# ============================================================

height = np.asarray(
    [
        r[
            "pyrene_height_above_graphene_nm"
        ]
        for r in merged
    ],
    dtype=float,
)

tilt = np.asarray(
    [
        r[
            "pyrene_tilt_from_graphene_plane_deg"
        ]
        for r in merged
    ],
    dtype=float,
)

temperature = np.asarray(
    [
        r["temperature_K"]
        for r in merged
    ],
    dtype=float,
)

water_below = np.asarray(
    [
        int(
            round(
                r["water_below_graphene"]
            )
        )
        for r in merged
    ],
    dtype=int,
)


contact_mask = (
    height
    <=
    CONTACT_HEIGHT_NM
)

stable_mask = (
    contact_mask
    &
    (
        tilt
        <=
        FACEON_TILT_DEG
    )
)


# ============================================================
# 5. CONTIGUOUS EPISODE FINDER
# ============================================================

def find_episodes(
    mask,
):

    episodes = []

    start = None
    previous_index = None


    for i, active in enumerate(
        mask
    ):

        if active:

            if start is None:

                start = i

            elif (
                previous_index is not None
                and
                (
                    times[i]
                    -
                    times[previous_index]
                )
                >
                CONTINUITY_GAP_TOL_PS
            ):

                episodes.append(
                    (
                        start,
                        previous_index,
                    )
                )

                start = i

            previous_index = i


        else:

            if start is not None:

                episodes.append(
                    (
                        start,
                        i - 1,
                    )
                )

                start = None
                previous_index = None


    if start is not None:

        episodes.append(
            (
                start,
                len(mask) - 1,
            )
        )


    return episodes


contact_episodes_idx = find_episodes(
    contact_mask
)

stable_episodes_idx = find_episodes(
    stable_mask
)


def episode_record(
    start_i,
    end_i,
):

    return {
        "start_ps":
            float(
                times[start_i]
            ),

        "end_ps":
            float(
                times[end_i]
            ),

        "duration_ps":
            float(
                times[end_i]
                -
                times[start_i]
            ),

        "start_ns":
            float(
                times[start_i]
                /
                1000.0
            ),

        "end_ns":
            float(
                times[end_i]
                /
                1000.0
            ),
    }


contact_episodes = [
    episode_record(
        a,
        b,
    )
    for a, b
    in contact_episodes_idx
]

stable_episodes = [
    episode_record(
        a,
        b,
    )
    for a, b
    in stable_episodes_idx
]


# ============================================================
# 6. FIRST RAW CROSSINGS
# ============================================================

contact_indices = np.flatnonzero(
    contact_mask
)

stable_indices = np.flatnonzero(
    stable_mask
)


first_raw_contact_ps = (
    float(
        times[
            contact_indices[0]
        ]
    )
    if len(contact_indices)
    else None
)

first_raw_stable_ps = (
    float(
        times[
            stable_indices[0]
        ]
    )
    if len(stable_indices)
    else None
)


# ============================================================
# 7. FIRST SUSTAINED EPISODE + CONFIRMATION TIME
# ============================================================

def first_qualifying_episode(
    episodes_idx,
    required_duration_ps,
):

    for start_i, end_i in episodes_idx:

        duration = (
            times[end_i]
            -
            times[start_i]
        )

        if (
            duration
            +
            1.0e-6
            >=
            required_duration_ps
        ):

            confirmation_i = None

            for i in range(
                start_i,
                end_i + 1,
            ):

                if (
                    times[i]
                    -
                    times[start_i]
                    +
                    1.0e-6
                    >=
                    required_duration_ps
                ):

                    confirmation_i = i
                    break


            if confirmation_i is None:

                raise RuntimeError(
                    "Internal episode-confirmation error."
                )


            return {
                "onset_index":
                    start_i,

                "end_index":
                    end_i,

                "confirmation_index":
                    confirmation_i,

                "onset_ps":
                    float(
                        times[start_i]
                    ),

                "confirmation_ps":
                    float(
                        times[
                            confirmation_i
                        ]
                    ),

                "observed_end_ps":
                    float(
                        times[end_i]
                    ),

                "observed_duration_ps":
                    float(
                        times[end_i]
                        -
                        times[start_i]
                    ),

                "right_censored_at_8ns":
                    bool(
                        end_i
                        ==
                        len(times) - 1
                    ),
            }


    return None


sustained_contact = first_qualifying_episode(
    contact_episodes_idx,
    CONTACT_DURATION_PS,
)

stable_adsorption = first_qualifying_episode(
    stable_episodes_idx,
    STABLE_DURATION_PS,
)


# ============================================================
# 8. PRE/POST ADSORPTION STATISTICS
# ============================================================

def stats(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if len(values) > 1
            else 0.0,

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "p05":
            float(
                np.percentile(
                    values,
                    5,
                )
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),
    }


if stable_adsorption is not None:

    adsorption_start_i = (
        stable_adsorption[
            "onset_index"
        ]
    )

    pre_slice = slice(
        0,
        adsorption_start_i,
    )

    post_slice = slice(
        adsorption_start_i,
        None,
    )

    pre_height_stats = stats(
        height[
            pre_slice
        ]
    )

    pre_tilt_stats = stats(
        tilt[
            pre_slice
        ]
    )

    post_height_stats = stats(
        height[
            post_slice
        ]
    )

    post_tilt_stats = stats(
        tilt[
            post_slice
        ]
    )

else:

    adsorption_start_i = None

    pre_height_stats = stats(
        height
    )

    pre_tilt_stats = stats(
        tilt
    )

    post_height_stats = None
    post_tilt_stats = None


# ============================================================
# 9. THRESHOLD ESCAPES AFTER STABLE-ADSORPTION ONSET
# ============================================================

post_adsorption_contact_violations = []
post_adsorption_stable_violations = []


if adsorption_start_i is not None:

    for i in range(
        adsorption_start_i,
        len(times),
    ):

        if not contact_mask[i]:

            post_adsorption_contact_violations.append(
                float(
                    times[i]
                )
            )

        if not stable_mask[i]:

            post_adsorption_stable_violations.append(
                float(
                    times[i]
                )
            )


# ============================================================
# 10. WATER / TEMPERATURE / GLOBAL SANITY
# ============================================================

temperature_stats = stats(
    temperature
)

max_water_below = int(
    np.max(
        water_below
    )
)

height_global = stats(
    height
)

tilt_global = stats(
    tilt
)


# ============================================================
# 11. WRITE MERGED CSV
# ============================================================

OUTPUT_MERGED.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_MERGED,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in merged:

        writer.writerow(
            {
                key:
                    row[key]
                for key in fieldnames
            }
        )


# ============================================================
# 12. TRANSITION WINDOW
#
# +/- 150 ps around stable adsorption onset.
# ============================================================

if stable_adsorption is not None:

    center_ps = (
        stable_adsorption[
            "onset_ps"
        ]
    )

    transition_rows = [
        row
        for row in merged
        if (
            center_ps - 150.0
            <=
            row["time_ps"]
            <=
            center_ps + 150.0
        )
    ]


    with open(
        OUTPUT_TRANSITION,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in transition_rows:

            writer.writerow(
                {
                    key:
                        row[key]
                    for key in fieldnames
                }
            )


# ============================================================
# 13. FIGURE
# ============================================================

time_ns = (
    times
    /
    1000.0
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(
        11,
        7,
    ),
    sharex=True,
)


axes[0].plot(
    time_ns,
    height,
    linewidth=0.8,
)

axes[0].axhline(
    CONTACT_HEIGHT_NM,
    linestyle="--",
    linewidth=1.0,
    label="Contact threshold = 0.45 nm",
)

# Independent bare-pyrene FF-2 minimum,
# shown only as a geometric reference.
axes[0].axhline(
    0.33,
    linestyle=":",
    linewidth=1.0,
    label="Bare-pyrene reference ~0.33 nm",
)

axes[0].set_ylabel(
    "Pyrene height (nm)"
)

axes[0].legend()


axes[1].plot(
    time_ns,
    tilt,
    linewidth=0.8,
)

axes[1].axhline(
    FACEON_TILT_DEG,
    linestyle="--",
    linewidth=1.0,
    label="Face-on threshold = 30°",
)

axes[1].set_ylabel(
    "Pyrene tilt (degrees)"
)

axes[1].set_xlabel(
    "Simulation time (ns)"
)

axes[1].legend()


if stable_adsorption is not None:

    onset_ns = (
        stable_adsorption[
            "onset_ps"
        ]
        /
        1000.0
    )

    confirmation_ns = (
        stable_adsorption[
            "confirmation_ps"
        ]
        /
        1000.0
    )

    for ax in axes:

        ax.axvline(
            onset_ns,
            linestyle=":",
            linewidth=1.0,
        )

        ax.axvline(
            confirmation_ns,
            linestyle="-.",
            linewidth=1.0,
        )


fig.suptitle(
    "Drop-01: Unbiased Pyrene–PEG5 Adsorption onto Graphene"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_FIGURE,
    dpi=180,
)

plt.close(
    fig
)


# ============================================================
# 14. SUMMARY JSON
# ============================================================

summary = {
    "simulation":
        {
            "start_ps":
                float(
                    times[0]
                ),

            "end_ps":
                float(
                    times[-1]
                ),

            "unique_rows":
                int(
                    len(
                        merged
                    )
                ),

            "duplicate_boundary_rows_removed":
                int(
                    len(
                        duplicate_times
                    )
                ),
        },

    "predeclared_criteria":
        {
            "contact_height_nm":
                CONTACT_HEIGHT_NM,

            "contact_sustained_ps":
                CONTACT_DURATION_PS,

            "stable_faceon_height_nm":
                CONTACT_HEIGHT_NM,

            "stable_faceon_tilt_deg":
                FACEON_TILT_DEG,

            "stable_faceon_sustained_ps":
                STABLE_DURATION_PS,
        },

    "first_raw_contact_ps":
        first_raw_contact_ps,

    "first_raw_stable_geometry_ps":
        first_raw_stable_ps,

    "first_sustained_contact":
        None
        if sustained_contact is None
        else {
            key: value
            for key, value
            in sustained_contact.items()
            if not key.endswith(
                "_index"
            )
        },

    "first_stable_faceon_adsorption":
        None
        if stable_adsorption is None
        else {
            key: value
            for key, value
            in stable_adsorption.items()
            if not key.endswith(
                "_index"
            )
        },

    "contact_episodes":
        contact_episodes,

    "stable_geometry_episodes":
        stable_episodes,

    "global_height_nm":
        height_global,

    "global_tilt_deg":
        tilt_global,

    "pre_adsorption_height_nm":
        pre_height_stats,

    "pre_adsorption_tilt_deg":
        pre_tilt_stats,

    "post_adsorption_height_nm":
        post_height_stats,

    "post_adsorption_tilt_deg":
        post_tilt_stats,

    "post_adsorption_contact_violation_count":
        len(
            post_adsorption_contact_violations
        ),

    "post_adsorption_contact_violation_times_ps":
        post_adsorption_contact_violations,

    "post_adsorption_stable_geometry_violation_count":
        len(
            post_adsorption_stable_violations
        ),

    "post_adsorption_stable_geometry_violation_times_ps":
        post_adsorption_stable_violations,

    "temperature_K":
        temperature_stats,

    "maximum_water_below_graphene":
        max_water_below,
}


OUTPUT_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    +
    "\n"
)


# ============================================================
# 15. HUMAN-READABLE REPORT
# ============================================================

print()
print("=" * 82)
print("PREDECLARED ADSORPTION CRITERIA")
print("=" * 82)

print(
    "Surface contact:"
)
print(
    f"  h <= {CONTACT_HEIGHT_NM:.2f} nm "
    f"for >= {CONTACT_DURATION_PS:.0f} ps"
)

print(
    "Stable face-on adsorption:"
)
print(
    f"  h <= {CONTACT_HEIGHT_NM:.2f} nm"
)
print(
    f"  tilt <= {FACEON_TILT_DEG:.0f} deg"
)
print(
    f"  sustained >= {STABLE_DURATION_PS:.0f} ps"
)


print()
print("=" * 82)
print("EVENT TIMING")
print("=" * 82)

print(
    "First raw h<=0.45 crossing:",
    (
        f"{first_raw_contact_ps:.3f} ps "
        f"({first_raw_contact_ps/1000.0:.6f} ns)"
        if first_raw_contact_ps is not None
        else "NONE"
    )
)


if sustained_contact is None:

    print(
        "Sustained 20 ps contact: NOT OBSERVED"
    )

else:

    print(
        "Sustained-contact onset:",
        f"{sustained_contact['onset_ps']:.3f} ps "
        f"({sustained_contact['onset_ps']/1000.0:.6f} ns)"
    )

    print(
        "20 ps criterion formally confirmed:",
        f"{sustained_contact['confirmation_ps']:.3f} ps "
        f"({sustained_contact['confirmation_ps']/1000.0:.6f} ns)"
    )


if stable_adsorption is None:

    print(
        "Stable 50 ps face-on adsorption: NOT OBSERVED"
    )

else:

    print(
        "Stable-adsorption onset:",
        f"{stable_adsorption['onset_ps']:.3f} ps "
        f"({stable_adsorption['onset_ps']/1000.0:.6f} ns)"
    )

    print(
        "50 ps criterion formally confirmed:",
        f"{stable_adsorption['confirmation_ps']:.3f} ps "
        f"({stable_adsorption['confirmation_ps']/1000.0:.6f} ns)"
    )

    print(
        "Observed qualifying episode end:",
        f"{stable_adsorption['observed_end_ps']:.3f} ps"
    )

    print(
        "Observed qualifying residence:",
        f"{stable_adsorption['observed_duration_ps']:.3f} ps "
        f"({stable_adsorption['observed_duration_ps']/1000.0:.6f} ns)"
    )

    print(
        "Episode reaches 8 ns endpoint:",
        stable_adsorption[
            "right_censored_at_8ns"
        ],
    )


print()
print("=" * 82)
print("ADSORBED-STATE GEOMETRY")
print("=" * 82)

if post_height_stats is not None:

    print(
        "Post-onset pyrene height:"
    )

    print(
        f"  mean +/- SD: "
        f"{post_height_stats['mean']:.6f} +/- "
        f"{post_height_stats['std']:.6f} nm"
    )

    print(
        f"  median: "
        f"{post_height_stats['median']:.6f} nm"
    )

    print(
        f"  5-95%: "
        f"{post_height_stats['p05']:.6f} - "
        f"{post_height_stats['p95']:.6f} nm"
    )


    print(
        "Post-onset pyrene tilt:"
    )

    print(
        f"  mean +/- SD: "
        f"{post_tilt_stats['mean']:.3f} +/- "
        f"{post_tilt_stats['std']:.3f} deg"
    )

    print(
        f"  median: "
        f"{post_tilt_stats['median']:.3f} deg"
    )

    print(
        f"  5-95%: "
        f"{post_tilt_stats['p05']:.3f} - "
        f"{post_tilt_stats['p95']:.3f} deg"
    )


print()
print("=" * 82)
print("POST-ADSORPTION THRESHOLD CHECK")
print("=" * 82)

print(
    "Post-onset h>0.45 sampled frames:",
    len(
        post_adsorption_contact_violations
    )
)

print(
    "Post-onset frames violating "
    "(h<=0.45 AND tilt<=30):",
    len(
        post_adsorption_stable_violations
    )
)


print()
print("=" * 82)
print("SYSTEM SANITY")
print("=" * 82)

print(
    f"Temperature: "
    f"{temperature_stats['mean']:.3f} +/- "
    f"{temperature_stats['std']:.3f} K"
)

print(
    "Maximum waters below graphene:",
    max_water_below
)


print()
print("=" * 82)
print("OUTPUTS")
print("=" * 82)

print(
    "Merged log:",
    OUTPUT_MERGED
)

print(
    "Summary JSON:",
    OUTPUT_SUMMARY
)

if stable_adsorption is not None:

    print(
        "Transition window:",
        OUTPUT_TRANSITION
    )

print(
    "Height/tilt figure:",
    OUTPUT_FIGURE
)


print()
print("=" * 82)

if stable_adsorption is not None:

    print(
        "DROP-01 FORMAL CLASSIFICATION: "
        "STABLE FACE-ON ADSORPTION OBSERVED"
    )

else:

    print(
        "DROP-01 FORMAL CLASSIFICATION: "
        "STABLE FACE-ON ADSORPTION NOT OBSERVED"
    )

print("=" * 82)

print()
print(
    "No MD, checkpoint, trajectory, or force-field "
    "state was modified."
)
