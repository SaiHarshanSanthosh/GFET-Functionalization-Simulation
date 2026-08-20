from pathlib import Path
import json

import numpy as np
from openmm import openmm, unit


# ============================================================
# PERIODIC WATER-SEAM DIAGNOSTIC
#
# PURPOSE
# -------
# Determine whether the severe OPC water-water overlaps are:
#
#   A) genuine local overlaps in the saved coordinates
#
# or
#
#   B) waters that appear far apart in Cartesian coordinates
#      but become very close after applying the skewed
#      graphene periodic box.
#
# NO MD.
# NO MINIMIZATION.
# NO COORDINATES ARE MODIFIED.
# ============================================================


# ============================================================
# 1. FILES
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_pme.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_positions_nm.npy"
)

CLASH_METADATA = Path(
    "analysis/"
    "starting_water_clash_diagnostic.json"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "periodic_water_seam_diagnostic.json"
)


# ============================================================
# 2. INDEXING
# ============================================================

N_SOLUTE = 3820
N_WATERS = 6196
SITES_PER_WATER = 4

O_SITE = 0


# ============================================================
# 3. VERIFY FILES
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    CLASH_METADATA,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 4. LOAD SYSTEM + POSITIONS
# ============================================================

with open(
    SYSTEM_XML,
    "r",
) as f:

    system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


positions = np.load(
    POSITIONS_FILE
)


with open(
    CLASH_METADATA,
    "r",
) as f:

    clash_metadata = json.load(
        f
    )


print()
print("=" * 72)
print("PERIODIC WATER-SEAM DIAGNOSTIC")
print("=" * 72)


# ============================================================
# 5. PERIODIC BOX
# ============================================================

a, b, c = (
    system.getDefaultPeriodicBoxVectors()
)


def vec_nm(v):

    return np.asarray(
        v.value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


box = np.vstack(
    [
        vec_nm(a),
        vec_nm(b),
        vec_nm(c),
    ]
)


inverse_box = np.linalg.inv(
    box
)


print()
print("PERIODIC BOX")
print("------------")

print(
    "a:",
    box[0],
)

print(
    "b:",
    box[1],
)

print(
    "c:",
    box[2],
)


# ============================================================
# 6. EXTRACT WATER OXYGENS
# ============================================================

oxygen_indices = (
    N_SOLUTE
    +
    np.arange(
        N_WATERS,
        dtype=int,
    )
    * SITES_PER_WATER
    +
    O_SITE
)


oxygen_positions = (
    positions[
        oxygen_indices
    ]
)


if oxygen_positions.shape != (
    N_WATERS,
    3,
):

    raise RuntimeError(
        "Failed to extract water oxygens."
    )


# ============================================================
# 7. PERIODIC DISPLACEMENT INFORMATION
# ============================================================

def analyze_pair(
    water_1,
    water_2,
):

    p1 = (
        oxygen_positions[
            water_1
        ]
    )

    p2 = (
        oxygen_positions[
            water_2
        ]
    )

    raw_delta = (
        p2
        - p1
    )

    raw_distance = float(
        np.linalg.norm(
            raw_delta
        )
    )

    # Convert displacement into fractional box coordinates.
    fractional = (
        raw_delta
        @ inverse_box
    )

    # Integer lattice translation OpenMM effectively uses
    # to select the nearest periodic image.
    lattice_shift = np.round(
        fractional
    )

    wrapped_fractional = (
        fractional
        - lattice_shift
    )

    wrapped_delta = (
        wrapped_fractional
        @ box
    )

    wrapped_distance = float(
        np.linalg.norm(
            wrapped_delta
        )
    )

    periodic_wrapping_used = bool(
        np.any(
            np.abs(
                lattice_shift
            ) > 0.5
        )
    )

    return {

        "water_1":
            int(
                water_1
            ),

        "water_2":
            int(
                water_2
            ),

        "raw_delta_nm":
            raw_delta.tolist(),

        "raw_distance_nm":
            raw_distance,

        "raw_fractional_displacement":
            fractional.tolist(),

        "lattice_shift":
            lattice_shift.astype(
                int
            ).tolist(),

        "wrapped_fractional_displacement":
            wrapped_fractional.tolist(),

        "wrapped_delta_nm":
            wrapped_delta.tolist(),

        "minimum_image_distance_nm":
            wrapped_distance,

        "periodic_wrapping_used":
            periodic_wrapping_used,
    }


# ============================================================
# 8. READ THE CLOSEST PAIRS FOUND BY PREVIOUS DIAGNOSTIC
# ============================================================

closest_pairs = (
    clash_metadata.get(
        "closest_water_pairs",
        []
    )
)


if len(
    closest_pairs
) == 0:

    raise RuntimeError(
        "No closest-water pairs were found "
        "in clash diagnostic metadata."
    )


print()
print("INSPECTING PREVIOUS CLOSEST WATER PAIRS")
print("---------------------------------------")


records = []


for rank, pair_record in enumerate(
    closest_pairs,
    start=1,
):

    w1 = int(
        pair_record[
            "water_1"
        ]
    )

    w2 = int(
        pair_record[
            "water_2"
        ]
    )

    record = analyze_pair(
        w1,
        w2,
    )

    records.append(
        record
    )

    print()
    print(
        f"PAIR {rank}"
    )

    print(
        "------"
    )

    print(
        f"Waters: "
        f"{w1} / {w2}"
    )

    print(
        f"Raw Cartesian distance:       "
        f"{record['raw_distance_nm']:.8f} nm"
    )

    print(
        f"Periodic minimum distance:    "
        f"{record['minimum_image_distance_nm']:.8f} nm"
    )

    print(
        "Raw fractional displacement: ",
        np.asarray(
            record[
                "raw_fractional_displacement"
            ]
        ),
    )

    print(
        "Periodic lattice shift:      ",
        record[
            "lattice_shift"
        ],
    )

    print(
        "Wrapped fractional displacement:",
        np.asarray(
            record[
                "wrapped_fractional_displacement"
            ]
        ),
    )

    print(
        "Periodic wrapping required:  ",
        (
            "YES"
            if record[
                "periodic_wrapping_used"
            ]
            else "NO"
        ),
    )


# ============================================================
# 9. CLASSIFY TOP PAIRS
# ============================================================

n_periodic_seam = sum(

    1

    for record in records

    if (
        record[
            "periodic_wrapping_used"
        ]

        and
        record[
            "raw_distance_nm"
        ] > 0.50

        and
        record[
            "minimum_image_distance_nm"
        ] < 0.24
    )
)


n_true_local = sum(

    1

    for record in records

    if (
        record[
            "raw_distance_nm"
        ] < 0.24
    )
)


n_other = (
    len(records)
    - n_periodic_seam
    - n_true_local
)


print()
print("=" * 72)
print("PAIR CLASSIFICATION")
print("=" * 72)

print(
    "Top pairs inspected:",
    len(records),
)

print(
    "Periodic-seam clashes:",
    n_periodic_seam,
)

print(
    "True local clashes:",
    n_true_local,
)

print(
    "Other:",
    n_other,
)


# ============================================================
# 10. FINAL DIAGNOSIS
# ============================================================

if (
    n_periodic_seam
    == len(records)
):

    status = (
        "PERIODIC_SEAM_PACKING_BUG_CONFIRMED"
    )


elif (
    n_true_local
    == len(records)
):

    status = (
        "TRUE_LOCAL_WATER_PACKING_OVERLAPS"
    )


elif (
    n_periodic_seam
    > 0
    and
    n_true_local
    > 0
):

    status = (
        "MIXED_PERIODIC_AND_LOCAL_OVERLAPS"
    )


else:

    status = (
        "OVERLAP_SOURCE_REQUIRES_FURTHER_DIAGNOSIS"
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
# 11. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "status":
        status,

    "box_vectors_nm":
        box.tolist(),

    "pairs_inspected":
        len(records),

    "periodic_seam_clashes":
        n_periodic_seam,

    "true_local_clashes":
        n_true_local,

    "other_pairs":
        n_other,

    "pair_details":
        records,

    "md_run":
        False,

    "minimization_run":
        False,

    "coordinates_modified":
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
# 12. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("PERIODIC WATER-SEAM DIAGNOSTIC: COMPLETE")
print("=" * 72)

print()
print(
    "No MD has been run."
)

print(
    "No minimization has been run."
)

print(
    "No coordinates were modified."
)
