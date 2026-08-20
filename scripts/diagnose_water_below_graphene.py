from pathlib import Path
import json

import numpy as np


# ============================================================
# DIAGNOSE WATER BELOW SUPPORTED GRAPHENE
#
# PURPOSE
# -------
# Quantify whether OPC waters exist on the substrate side of
# the graphene sheet in our supposedly one-sided solvent model.
#
# This script compares:
#
#   1. the CLEAN PRUNED pre-minimization water coordinates
#   2. the CONVERGED minimized full-system coordinates
#
# NO COORDINATES ARE MODIFIED.
# NO WATERS ARE REMOVED.
# NO MD.
# NO MINIMIZATION.
# ============================================================


# ============================================================
# 1. INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70
N_SOLUTE = 3820

SITES_PER_WATER = 4

WATER_O = 0


# ============================================================
# 2. FILES
# ============================================================

PREMIN_WATER_FILE = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_pruned_exact_nm.npy"
)

MINIMIZED_FULL_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "minimized_converged_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "water_below_graphene_diagnostic.json"
)


for path in [
    PREMIN_WATER_FILE,
    MINIMIZED_FULL_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 3. LOAD
# ============================================================

premin_water = np.load(
    PREMIN_WATER_FILE
)

minimized_positions = np.load(
    MINIMIZED_FULL_FILE
)


if premin_water.ndim != 3:

    raise RuntimeError(
        f"Unexpected pre-min water shape: {premin_water.shape}"
    )


if premin_water.shape[1:] != (
    4,
    3,
):

    raise RuntimeError(
        f"Unexpected OPC water shape: {premin_water.shape}"
    )


n_waters = (
    premin_water.shape[0]
)


expected_particles = (
    N_SOLUTE
    +
    n_waters
    * SITES_PER_WATER
)


if minimized_positions.shape != (
    expected_particles,
    3,
):

    raise RuntimeError(
        "Minimized coordinate count does not match "
        f"{n_waters} waters. "
        f"Expected {(expected_particles, 3)}, "
        f"found {minimized_positions.shape}."
    )


print()
print("=" * 72)
print("WATER-BELOW-GRAPHENE DIAGNOSTIC")
print("=" * 72)

print()
print("SYSTEM")
print("------")

print(
    "OPC waters:",
    n_waters,
)

print(
    "Total minimized particles:",
    len(
        minimized_positions
    ),
)


# ============================================================
# 4. GRAPHENE PLANE AFTER MINIMIZATION
# ============================================================

graphene = (
    minimized_positions[
        :N_GRAPHENE_CARBONS
    ]
)


graphene_z = (
    graphene[
        :,
        2
    ]
)


graphene_z_min = float(
    graphene_z.min()
)

graphene_z_max = float(
    graphene_z.max()
)

graphene_z_mean = float(
    graphene_z.mean()
)

graphene_z_median = float(
    np.median(
        graphene_z
    )
)


print()
print("MINIMIZED GRAPHENE")
print("------------------")

print(
    f"z minimum: "
    f"{graphene_z_min:.6f} nm"
)

print(
    f"z maximum: "
    f"{graphene_z_max:.6f} nm"
)

print(
    f"z mean:    "
    f"{graphene_z_mean:.6f} nm"
)

print(
    f"z median:  "
    f"{graphene_z_median:.6f} nm"
)

print(
    f"corrugation span: "
    f"{graphene_z_max - graphene_z_min:.6f} nm"
)


# ============================================================
# 5. PRE-MINIMIZATION OXYGEN Z
# ============================================================

premin_oxygen_z = (
    premin_water[
        :,
        WATER_O,
        2
    ]
)


# ============================================================
# 6. MINIMIZED OXYGEN Z
# ============================================================

oxygen_indices = (
    N_SOLUTE
    +
    np.arange(
        n_waters
    )
    * SITES_PER_WATER
)


minimized_oxygen_z = (
    minimized_positions[
        oxygen_indices,
        2
    ]
)


# ============================================================
# 7. DEFINE GEOMETRIC CATEGORIES
#
# Below graphene minimum:
#     oxygen is underneath every carbon in z.
#
# Below graphene mean:
#     oxygen is on the substrate side of the nominal plane.
#
# Below graphene maximum:
#     oxygen is below at least the highest graphene carbon.
#
# For a one-sided model, "below mean" is the main diagnostic.
# ============================================================

pre_below_min = (
    premin_oxygen_z
    <
    graphene_z_min
)

pre_below_mean = (
    premin_oxygen_z
    <
    graphene_z_mean
)

pre_below_max = (
    premin_oxygen_z
    <
    graphene_z_max
)


post_below_min = (
    minimized_oxygen_z
    <
    graphene_z_min
)

post_below_mean = (
    minimized_oxygen_z
    <
    graphene_z_mean
)

post_below_max = (
    minimized_oxygen_z
    <
    graphene_z_max
)


# ============================================================
# 8. REPORT COUNTS
# ============================================================

def report_counts(
    title,
    below_min,
    below_mean,
    below_max,
):

    print()
    print(title)
    print("-" * len(title))

    print(
        "Below graphene minimum:",
        int(
            np.sum(
                below_min
            )
        ),
    )

    print(
        "Below graphene mean:   ",
        int(
            np.sum(
                below_mean
            )
        ),
    )

    print(
        "Below graphene maximum:",
        int(
            np.sum(
                below_max
            )
        ),
    )


report_counts(
    "PRE-MINIMIZATION WATER OXYGENS",
    pre_below_min,
    pre_below_mean,
    pre_below_max,
)


report_counts(
    "POST-MINIMIZATION WATER OXYGENS",
    post_below_min,
    post_below_mean,
    post_below_max,
)


# ============================================================
# 9. DID WATERS CROSS THE GRAPHENE PLANE?
# ============================================================

started_above_mean = (
    premin_oxygen_z
    >= graphene_z_mean
)

ended_below_mean = (
    minimized_oxygen_z
    <
    graphene_z_mean
)


crossed_down = (
    started_above_mean
    &
    ended_below_mean
)


started_below_mean = (
    premin_oxygen_z
    <
    graphene_z_mean
)

ended_above_mean = (
    minimized_oxygen_z
    >= graphene_z_mean
)


crossed_up = (
    started_below_mean
    &
    ended_above_mean
)


print()
print("GRAPHENE-PLANE CROSSING")
print("-----------------------")

print(
    "Started above mean and ended below:",
    int(
        np.sum(
            crossed_down
        )
    ),
)

print(
    "Started below mean and ended above:",
    int(
        np.sum(
            crossed_up
        )
    ),
)


# ============================================================
# 10. FIRST WATER LAYER ABOVE GRAPHENE
# ============================================================

above_mean_mask = (
    minimized_oxygen_z
    >
    graphene_z_mean
)


above_max_mask = (
    minimized_oxygen_z
    >
    graphene_z_max
)


if not np.any(
    above_mean_mask
):

    raise RuntimeError(
        "No water oxygens found above graphene mean plane."
    )


if not np.any(
    above_max_mask
):

    raise RuntimeError(
        "No water oxygens found above graphene maximum."
    )


first_above_mean_z = float(
    np.min(
        minimized_oxygen_z[
            above_mean_mask
        ]
    )
)


first_above_max_z = float(
    np.min(
        minimized_oxygen_z[
            above_max_mask
        ]
    )
)


mean_plane_gap = (
    first_above_mean_z
    -
    graphene_z_mean
)


max_surface_gap = (
    first_above_max_z
    -
    graphene_z_max
)


print()
print("FIRST WATER OXYGEN ABOVE GRAPHENE")
print("---------------------------------")

print(
    f"Closest O above mean plane: "
    f"{first_above_mean_z:.6f} nm"
)

print(
    f"Vertical gap above mean:     "
    f"{mean_plane_gap:.6f} nm"
)

print(
    f"Closest O above max carbon:  "
    f"{first_above_max_z:.6f} nm"
)

print(
    f"Vertical gap above max C:     "
    f"{max_surface_gap:.6f} nm"
)


# ============================================================
# 11. LOWEST WATER OXYGENS
# ============================================================

order = np.argsort(
    minimized_oxygen_z
)


print()
print("20 LOWEST MINIMIZED WATER OXYGENS")
print("---------------------------------")


lowest_records = []


for rank, water_index in enumerate(
    order[:20],
    start=1,
):

    water_index = int(
        water_index
    )

    pre_z = float(
        premin_oxygen_z[
            water_index
        ]
    )

    post_z = float(
        minimized_oxygen_z[
            water_index
        ]
    )

    record = {
        "rank":
            rank,

        "water_index":
            water_index,

        "premin_z_nm":
            pre_z,

        "minimized_z_nm":
            post_z,

        "relative_to_graphene_mean_nm":
            float(
                post_z
                -
                graphene_z_mean
            ),
    }

    lowest_records.append(
        record
    )

    print(
        f"{rank:2d}. "
        f"water {water_index:4d} | "
        f"pre={pre_z:.6f} nm | "
        f"post={post_z:.6f} nm | "
        f"post-graphene_mean="
        f"{post_z - graphene_z_mean:+.6f} nm"
    )


# ============================================================
# 12. LIST ALL BELOW-MEAN WATERS
# ============================================================

below_mean_indices = np.where(
    post_below_mean
)[0]


below_min_indices = np.where(
    post_below_min
)[0]


print()
print("BELOW-PLANE INDEX SUMMARY")
print("-------------------------")

print(
    "Waters below graphene mean:",
    len(
        below_mean_indices
    ),
)

print(
    "Waters below graphene minimum:",
    len(
        below_min_indices
    ),
)


if len(
    below_mean_indices
) <= 100:

    print()
    print(
        "Below-mean water indices:"
    )

    print(
        below_mean_indices.tolist()
    )


# ============================================================
# 13. FRACTION OF SOLVENT
# ============================================================

below_mean_fraction = (
    len(
        below_mean_indices
    )
    /
    n_waters
)


below_min_fraction = (
    len(
        below_min_indices
    )
    /
    n_waters
)


print()
print("FRACTION OF SOLVENT BELOW GRAPHENE")
print("----------------------------------")

print(
    f"Below mean plane: "
    f"{100.0 * below_mean_fraction:.3f}%"
)

print(
    f"Below entire sheet: "
    f"{100.0 * below_min_fraction:.3f}%"
)


# ============================================================
# 14. DIAGNOSTIC CLASSIFICATION
# ============================================================

if len(
    below_mean_indices
) == 0:

    status = (
        "ONE_SIDED_SOLVENT_GEOMETRY_PASS"
    )


elif np.sum(
    crossed_down
) > 0:

    status = (
        "WATER_EXISTS_BELOW_GRAPHENE_AND_SOME_CROSSED_DURING_MINIMIZATION"
    )


else:

    status = (
        "WATER_WAS_ALREADY_PACKED_BELOW_GRAPHENE"
    )


print()
print("=" * 72)
print("DIAGNOSTIC RESULT")
print("=" * 72)

print()
print(
    status
)


# ============================================================
# 15. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "status":
        status,

    "n_waters":
        int(
            n_waters
        ),

    "graphene_z_min_nm":
        graphene_z_min,

    "graphene_z_max_nm":
        graphene_z_max,

    "graphene_z_mean_nm":
        graphene_z_mean,

    "graphene_z_median_nm":
        graphene_z_median,

    "premin_below_graphene_min":
        int(
            np.sum(
                pre_below_min
            )
        ),

    "premin_below_graphene_mean":
        int(
            np.sum(
                pre_below_mean
            )
        ),

    "premin_below_graphene_max":
        int(
            np.sum(
                pre_below_max
            )
        ),

    "postmin_below_graphene_min":
        int(
            np.sum(
                post_below_min
            )
        ),

    "postmin_below_graphene_mean":
        int(
            np.sum(
                post_below_mean
            )
        ),

    "postmin_below_graphene_max":
        int(
            np.sum(
                post_below_max
            )
        ),

    "crossed_from_above_to_below_mean":
        int(
            np.sum(
                crossed_down
            )
        ),

    "crossed_from_below_to_above_mean":
        int(
            np.sum(
                crossed_up
            )
        ),

    "first_water_above_mean_z_nm":
        first_above_mean_z,

    "first_water_above_mean_gap_nm":
        float(
            mean_plane_gap
        ),

    "first_water_above_max_z_nm":
        first_above_max_z,

    "first_water_above_max_gap_nm":
        float(
            max_surface_gap
        ),

    "below_mean_water_indices":
        [
            int(i)
            for i in (
                below_mean_indices
            )
        ],

    "below_min_water_indices":
        [
            int(i)
            for i in (
                below_min_indices
            )
        ],

    "below_mean_fraction":
        float(
            below_mean_fraction
        ),

    "below_min_fraction":
        float(
            below_min_fraction
        ),

    "lowest_20_waters":
        lowest_records,

    "coordinates_modified":
        False,

    "waters_removed":
        False,

    "md_run":
        False,
}


with open(
    OUTPUT_METADATA,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# 16. FINAL
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Metadata:",
    OUTPUT_METADATA
)


print()
print("=" * 72)
print("WATER-BELOW-GRAPHENE DIAGNOSTIC: COMPLETE")
print("=" * 72)

print()
print(
    "No coordinates were modified."
)

print(
    "No waters were removed."
)

print(
    "No MD was run."
)
