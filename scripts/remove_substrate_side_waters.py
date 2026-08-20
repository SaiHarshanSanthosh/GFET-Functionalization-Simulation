from pathlib import Path
import json

import numpy as np


# ============================================================
# REMOVE SUBSTRATE-SIDE WATERS
#
# PURPOSE
# -------
# Our intended model is one-sided:
#
#        water
#        ligand
#   ================= graphene
#        substrate
#
# The solvent rebuild accidentally left a small number of
# waters underneath the graphene sheet.
#
# This script removes whole water molecules whose O atom lies
# below the supported graphene carbon plane.
#
# NO MD.
# NO MINIMIZATION.
# GRAPHENE/LIGAND COORDINATES ARE NOT MODIFIED.
# ============================================================


N_GRAPHENE_CARBONS = 1250

O = 0


SOLUTE_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

WATER_FILE = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_pruned_exact_nm.npy"
)

DIAGNOSTIC_FILE = Path(
    "analysis/"
    "water_below_graphene_diagnostic.json"
)


OUTPUT_WATER = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_one_sided_exact_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "one_sided_water_cleanup.json"
)


for path in [
    SOLUTE_FILE,
    WATER_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(path)


solute = np.load(
    SOLUTE_FILE
)

waters = np.load(
    WATER_FILE
)


if waters.ndim != 3 or waters.shape[1:] != (4, 3):
    raise RuntimeError(
        f"Unexpected water shape: {waters.shape}"
    )


graphene_z = (
    solute[
        :N_GRAPHENE_CARBONS,
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


print()
print("=" * 72)
print("ONE-SIDED WATER CLEANUP")
print("=" * 72)

print()
print("SUPPORTED GRAPHENE")
print("------------------")

print(
    f"z minimum: {graphene_z_min:.6f} nm"
)

print(
    f"z maximum: {graphene_z_max:.6f} nm"
)

print(
    f"z mean:    {graphene_z_mean:.6f} nm"
)


# ============================================================
# WATER OXYGENS
# ============================================================

oxygen_z = (
    waters[
        :,
        O,
        2
    ]
)


# Since the supported graphene is essentially planar, use the
# highest graphene carbon as the conservative one-sided cutoff.
#
# Any water O below this is on the substrate side.
keep = (
    oxygen_z
    >= graphene_z_max
)


remove = (
    ~keep
)


removed_indices = np.where(
    remove
)[0]


clean_waters = (
    waters[
        keep
    ]
)


print()
print("WATER CLEANUP")
print("-------------")

print(
    "Starting waters:",
    len(
        waters
    ),
)

print(
    "Substrate-side waters removed:",
    len(
        removed_indices
    ),
)

print(
    "Final waters:",
    len(
        clean_waters
    ),
)

print(
    f"Removed fraction: "
    f"{100.0 * len(removed_indices) / len(waters):.3f}%"
)


print()
print("REMOVED WATER INDICES")
print("---------------------")

print(
    removed_indices.tolist()
)


# ============================================================
# CROSS-CHECK WITH PRIOR DIAGNOSTIC
# ============================================================

diagnostic_match = None


if DIAGNOSTIC_FILE.exists():

    with open(
        DIAGNOSTIC_FILE,
        "r",
    ) as f:

        diagnostic = json.load(
            f
        )

    previous = sorted(
        diagnostic.get(
            "below_mean_water_indices",
            [],
        )
    )

    current = sorted(
        int(i)
        for i in removed_indices
    )

    diagnostic_match = (
        previous
        == current
    )

    print()
    print("PRIOR DIAGNOSTIC CROSS-CHECK")
    print("----------------------------")

    print(
        "Removed set matches prior diagnostic:",
        "YES" if diagnostic_match else "NO",
    )

    if not diagnostic_match:

        raise RuntimeError(
            "Water removal set does not match the "
            "previous below-graphene diagnostic."
        )


# ============================================================
# FINAL ONE-SIDED SANITY CHECK
# ============================================================

final_oxygen_z = (
    clean_waters[
        :,
        O,
        2
    ]
)


final_below = int(
    np.sum(
        final_oxygen_z
        < graphene_z_max
    )
)


first_water_z = float(
    final_oxygen_z.min()
)


first_water_gap = (
    first_water_z
    - graphene_z_max
)


print()
print("FINAL ONE-SIDED CHECK")
print("---------------------")

print(
    "Water oxygens below graphene:",
    final_below,
)

print(
    f"Lowest retained water O: "
    f"{first_water_z:.6f} nm"
)

print(
    f"Vertical distance above graphene: "
    f"{first_water_gap:.6f} nm"
)


if final_below != 0:

    raise RuntimeError(
        "Substrate-side waters remain."
    )


# ============================================================
# OPC GEOMETRY CHECK
# ============================================================

oh1 = np.linalg.norm(
    clean_waters[:, 1, :]
    -
    clean_waters[:, 0, :],
    axis=1,
)

oh2 = np.linalg.norm(
    clean_waters[:, 2, :]
    -
    clean_waters[:, 0, :],
    axis=1,
)


print()
print("OPC GEOMETRY SANITY")
print("-------------------")

print(
    f"Mean O-H1: {oh1.mean():.10f} nm"
)

print(
    f"Mean O-H2: {oh2.mean():.10f} nm"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_WATER.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_WATER,
    clean_waters,
)


OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "status":
        "PASS",

    "starting_waters":
        int(
            len(
                waters
            )
        ),

    "removed_substrate_side_waters":
        int(
            len(
                removed_indices
            )
        ),

    "removed_indices":
        [
            int(i)
            for i in removed_indices
        ],

    "final_waters":
        int(
            len(
                clean_waters
            )
        ),

    "graphene_z_min_nm":
        graphene_z_min,

    "graphene_z_max_nm":
        graphene_z_max,

    "graphene_z_mean_nm":
        graphene_z_mean,

    "lowest_retained_oxygen_z_nm":
        first_water_z,

    "lowest_retained_oxygen_gap_nm":
        float(
            first_water_gap
        ),

    "remaining_waters_below_graphene":
        final_below,

    "prior_diagnostic_match":
        diagnostic_match,

    "coordinates_modified":
        False,

    "graphene_modified":
        False,

    "ligand_modified":
        False,

    "md_run":
        False,

    "minimization_run":
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


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Clean one-sided water:",
    OUTPUT_WATER
)

print(
    "Metadata:",
    OUTPUT_METADATA
)


print()
print("=" * 72)
print("ONE-SIDED WATER CLEANUP: PASS")
print("=" * 72)

print()
print(
    "Only substrate-side whole water molecules were removed."
)

print(
    "Graphene and Pyrene-PEG5 were unchanged."
)

print(
    "No MD was run."
)

print(
    "No minimization was run."
)
