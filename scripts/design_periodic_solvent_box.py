from pathlib import Path
import json
import math
import numpy as np

from ase.build import graphene


# ============================================================
# INPUT
# ============================================================

INPUT_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_vacuum_positions_nm.npy"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_positions_nm.npy"
)

OUTPUT_BOX = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)

OUTPUT_JSON = Path(
    "analysis/periodic_solvent_box_design.json"
)


# ============================================================
# KNOWN PARTICLE COUNTS
# ============================================================

N_GRAPHENE_CARBON = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70

EXPECTED_TOTAL = (
    N_GRAPHENE_TOTAL
    + N_LIGAND
)


# ============================================================
# BOX DESIGN
#
# Give explicit solvent room on BOTH sides of graphene.
# This is a starting design choice, not a fitted parameter.
# ============================================================

WATER_PADDING_NM = 2.5

BOX_Z_ROUNDING_NM = 0.5


# ============================================================
# EXPERIMENTAL DPBS CONCENTRATIONS
#
# Sigma D8537:
# 8.0 g/L NaCl
# 0.2 g/L KCl
# 0.2 g/L KH2PO4
# 1.15 g/L anhydrous Na2HPO4
#
# Values below are mol/L.
# ============================================================

NA_CL = 0.136886
K_CL = 0.002683
KH2PO4 = 0.001470
NA2HPO4 = 0.008102


ION_CONCENTRATIONS_1X = {
    "Na+": (
        NA_CL
        + 2.0 * NA2HPO4
    ),

    "K+": (
        K_CL
        + KH2PO4
    ),

    "Cl-": (
        NA_CL
        + K_CL
    ),

    "H2PO4-": KH2PO4,

    "HPO4--": NA2HPO4,
}


AVOGADRO = 6.02214076e23

# Approximate bulk water concentration.
# Only used to ESTIMATE water count.
WATER_MOLARITY = 55.5


# ============================================================
# HELPER: MINIMUM LIGAND-IMAGE DISTANCE
#
# We explicitly test neighboring periodic copies so we know
# that the ligand is not accidentally almost touching itself.
# ============================================================

def minimum_periodic_ligand_distance(
    ligand_positions,
    box_vectors
):

    a = box_vectors[0]
    b = box_vectors[1]
    c = box_vectors[2]

    min_distance = np.inf
    min_shift = None

    for ia in (-1, 0, 1):

        for ib in (-1, 0, 1):

            for ic in (-1, 0, 1):

                if (
                    ia == 0
                    and ib == 0
                    and ic == 0
                ):
                    continue

                shift = (
                    ia * a
                    + ib * b
                    + ic * c
                )

                translated = (
                    ligand_positions
                    + shift
                )

                differences = (
                    ligand_positions[:, None, :]
                    - translated[None, :, :]
                )

                distances = np.linalg.norm(
                    differences,
                    axis=2
                )

                candidate = float(
                    np.min(distances)
                )

                if candidate < min_distance:

                    min_distance = candidate

                    min_shift = (
                        ia,
                        ib,
                        ic
                    )

    return (
        min_distance,
        min_shift
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_POSITIONS.exists():

        raise FileNotFoundError(
            f"Missing: {INPUT_POSITIONS}"
        )

    # --------------------------------------------------------
    # LOAD CURRENT VALIDATED VACUUM ASSEMBLY
    # --------------------------------------------------------

    positions = np.load(
        INPUT_POSITIONS
    )

    if positions.shape != (
        EXPECTED_TOTAL,
        3
    ):

        raise RuntimeError(
            "Expected positions shape "
            f"({EXPECTED_TOTAL}, 3), "
            f"found {positions.shape}."
        )

    graphene_positions = (
        positions[
            :N_GRAPHENE_TOTAL
        ]
    )

    carbon_positions = (
        positions[
            :N_GRAPHENE_CARBON
        ]
    )

    ligand_positions = (
        positions[
            N_GRAPHENE_TOTAL:
        ]
    )

    print(
        "Loaded validated combined positions:"
    )

    print(
        INPUT_POSITIONS
    )

    print()

    print(
        "Total particles:",
        len(positions)
    )

    print(
        "Graphene particles:",
        len(graphene_positions)
    )

    print(
        "Ligand atoms:",
        len(ligand_positions)
    )

    # ========================================================
    # REBUILD EXACT GRAPHENE XY CELL
    # ========================================================

    sheet = graphene(
        formula="C2",
        a=2.46,
        size=(25, 25, 1),
        vacuum=20.0
    )

    cell_A = (
        sheet.cell.array
    )

    # Preserve the exact graphene lattice vectors
    # in x and y.

    a_nm = (
        cell_A[0]
        / 10.0
    )

    b_nm = (
        cell_A[1]
        / 10.0
    )

    # ========================================================
    # MEASURE CURRENT SOLUTE HEIGHT
    # ========================================================

    solute_z_min = float(
        np.min(
            positions[:, 2]
        )
    )

    solute_z_max = float(
        np.max(
            positions[:, 2]
        )
    )

    solute_z_span = (
        solute_z_max
        - solute_z_min
    )

    graphene_plane_z = float(
        np.mean(
            carbon_positions[:, 2]
        )
    )

    ligand_z_max = float(
        np.max(
            ligand_positions[:, 2]
        )
    )

    ligand_height_above_graphene = (
        ligand_z_max
        - graphene_plane_z
    )

    print()

    print(
        "CURRENT SOLUTE HEIGHT"
    )

    print(
        "---------------------"
    )

    print(
        "Lowest particle z:",
        f"{solute_z_min:.6f} nm"
    )

    print(
        "Highest particle z:",
        f"{solute_z_max:.6f} nm"
    )

    print(
        "Total solute z-span:",
        f"{solute_z_span:.6f} nm"
    )

    print(
        "Ligand highest point above "
        "graphene:",
        f"{ligand_height_above_graphene:.6f} nm"
    )

    # ========================================================
    # CHOOSE Z BOX LENGTH
    # ========================================================

    raw_box_z = (
        solute_z_span
        + 2.0 * WATER_PADDING_NM
    )

    box_z_nm = (
        math.ceil(
            raw_box_z
            / BOX_Z_ROUNDING_NM
        )
        * BOX_Z_ROUNDING_NM
    )

    extra_space = (
        box_z_nm
        - raw_box_z
    )

    bottom_padding = (
        WATER_PADDING_NM
        + 0.5 * extra_space
    )

    top_padding = (
        bottom_padding
    )

    # ========================================================
    # SHIFT SYSTEM INTO NEW PERIODIC BOX
    #
    # Do not alter any internal geometry.
    # Only translate the entire system in z.
    # ========================================================

    shifted_positions = (
        positions.copy()
    )

    z_shift = (
        bottom_padding
        - solute_z_min
    )

    shifted_positions[
        :,
        2
    ] += z_shift

    shifted_z_min = float(
        np.min(
            shifted_positions[:, 2]
        )
    )

    shifted_z_max = float(
        np.max(
            shifted_positions[:, 2]
        )
    )

    actual_bottom_gap = (
        shifted_z_min
    )

    actual_top_gap = (
        box_z_nm
        - shifted_z_max
    )

    # ========================================================
    # FINAL TRICLINIC BOX VECTORS
    # ========================================================

    c_nm = np.array(
        [
            0.0,
            0.0,
            box_z_nm
        ]
    )

    box_vectors = np.vstack(
        [
            a_nm,
            b_nm,
            c_nm
        ]
    )

    volume_nm3 = float(
        abs(
            np.linalg.det(
                box_vectors
            )
        )
    )

    volume_L = (
        volume_nm3
        * 1.0e-24
    )

    print()

    print(
        "PERIODIC BOX"
    )

    print(
        "------------"
    )

    print(
        "a:",
        box_vectors[0]
    )

    print(
        "b:",
        box_vectors[1]
    )

    print(
        "c:",
        box_vectors[2]
    )

    print()

    print(
        "Box volume:",
        f"{volume_nm3:.3f} nm^3"
    )

    print(
        "Bottom solvent gap:",
        f"{actual_bottom_gap:.3f} nm"
    )

    print(
        "Top solvent gap:",
        f"{actual_top_gap:.3f} nm"
    )

    if (
        actual_bottom_gap
        < WATER_PADDING_NM - 1e-6
    ):

        raise RuntimeError(
            "Bottom padding too small."
        )

    if (
        actual_top_gap
        < WATER_PADDING_NM - 1e-6
    ):

        raise RuntimeError(
            "Top padding too small."
        )

    # ========================================================
    # CHECK LIGAND AGAINST PERIODIC COPIES
    # ========================================================

    shifted_ligand = (
        shifted_positions[
            N_GRAPHENE_TOTAL:
        ]
    )

    (
        min_image_distance,
        min_image_shift
    ) = minimum_periodic_ligand_distance(
        shifted_ligand,
        box_vectors
    )

    print()

    print(
        "PERIODIC IMAGE CHECK"
    )

    print(
        "--------------------"
    )

    print(
        "Minimum ligand-to-periodic-copy "
        "distance:",
        f"{min_image_distance:.3f} nm"
    )

    print(
        "Closest image translation:",
        min_image_shift
    )

    if (
        min_image_distance
        < 1.5
    ):

        raise RuntimeError(
            "Ligand periodic copies are "
            "too close for this box."
        )

    print(
        "Ligand periodic-image check: PASS"
    )

    # ========================================================
    # ESTIMATE WATER COUNT
    #
    # This is NOT the final water count because waters
    # overlapping graphene/ligand will later be removed.
    # ========================================================

    expected_water = (
        WATER_MOLARITY
        * volume_L
        * AVOGADRO
    )

    print()

    print(
        "ROUGH WATER ESTIMATE"
    )

    print(
        "--------------------"
    )

    print(
        "Full-box bulk-water estimate:",
        f"{expected_water:.1f} waters"
    )

    print(
        "Actual solvated count will be lower "
        "after overlap removal."
    )

    # ========================================================
    # EXPECTED DPBS COUNTS
    #
    # DO NOT ROUND THESE YET.
    #
    # These numbers show the finite-size problem directly.
    # ========================================================

    expected_counts_1x = {}

    expected_counts_001x = {}

    print()

    print(
        "EXPECTED 1X D8537 PARTICLE COUNTS"
    )

    print(
        "---------------------------------"
    )

    for species, concentration in (
        ION_CONCENTRATIONS_1X.items()
    ):

        count = (
            concentration
            * volume_L
            * AVOGADRO
        )

        expected_counts_1x[
            species
        ] = count

        print(
            f"{species:8s}: "
            f"{count:.4f}"
        )

    print()

    print(
        "EXPECTED 0.01X D8537 PARTICLE COUNTS"
    )

    print(
        "------------------------------------"
    )

    for species, concentration in (
        ION_CONCENTRATIONS_1X.items()
    ):

        diluted = (
            concentration
            / 100.0
        )

        count = (
            diluted
            * volume_L
            * AVOGADRO
        )

        expected_counts_001x[
            species
        ] = count

        print(
            f"{species:8s}: "
            f"{count:.6f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        OUTPUT_POSITIONS,
        shifted_positions
    )

    np.save(
        OUTPUT_BOX,
        box_vectors
    )

    metadata = {

        "water_padding_nm":
            WATER_PADDING_NM,

        "box_z_rounding_nm":
            BOX_Z_ROUNDING_NM,

        "solute_z_span_nm":
            solute_z_span,

        "box_vectors_nm":
            box_vectors.tolist(),

        "box_volume_nm3":
            volume_nm3,

        "bottom_gap_nm":
            actual_bottom_gap,

        "top_gap_nm":
            actual_top_gap,

        "minimum_ligand_image_distance_nm":
            min_image_distance,

        "minimum_ligand_image_translation":
            list(
                min_image_shift
            ),

        "rough_full_box_water_count":
            expected_water,

        "expected_1X_counts":
            expected_counts_1x,

        "expected_0.01X_counts":
            expected_counts_001x,

        "note":
            (
                "Expected ion counts are "
                "continuous statistical targets. "
                "They have intentionally NOT "
                "been rounded to whole particles."
            )
    }

    with open(
        OUTPUT_JSON,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()

    print(
        "SAVED"
    )

    print(
        "-----"
    )

    print(
        "Shifted positions:",
        OUTPUT_POSITIONS
    )

    print(
        "Box vectors:",
        OUTPUT_BOX
    )

    print(
        "Design metadata:",
        OUTPUT_JSON
    )

    print()

    print(
        "PERIODIC SOLVENT BOX DESIGN: PASS"
    )

    print()

    print(
        "No water or ions have been added yet."
    )


if __name__ == "__main__":
    main()
