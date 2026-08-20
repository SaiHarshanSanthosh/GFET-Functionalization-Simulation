from pathlib import Path
import json
import numpy as np


# ============================================================
# INPUTS
# ============================================================

INPUT_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_vacuum_positions_nm.npy"
)

INPUT_BOX = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "supported_one_sided_box_design.json"
)


# ============================================================
# KNOWN SYSTEM COUNTS
# ============================================================

N_GRAPHENE_CARBON = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70
N_TOTAL = 3820


# ============================================================
# SUPPORTED-SURFACE DESIGN
#
# These are simulation-boundary choices.
# They are NOT claimed SiO2 structural distances.
# ============================================================

# Put graphene near the bottom of the box.
TARGET_GRAPHENE_CARBON_Z_NM = 0.50

# Later this will become the lower substrate exclusion wall.
LOWER_WALL_Z_NM = 0.10

# Keep solvent away from the periodic z boundary.
TOP_WALL_Z_NM = 6.80


# Approximate water concentration, only for a rough count.
WATER_MOLARITY = 55.5
AVOGADRO = 6.02214076e23


def main():

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    positions = np.load(
        INPUT_POSITIONS
    )

    box = np.load(
        INPUT_BOX
    )

    if positions.shape != (N_TOTAL, 3):

        raise RuntimeError(
            f"Expected ({N_TOTAL}, 3), "
            f"found {positions.shape}."
        )

    if box.shape != (3, 3):

        raise RuntimeError(
            "Expected a 3x3 box matrix."
        )

    box_z = float(
        box[2, 2]
    )

    if abs(box_z - 7.0) > 1e-8:

        raise RuntimeError(
            f"Expected 7.0 nm box height; "
            f"found {box_z}."
        )

    # --------------------------------------------------------
    # CURRENT GEOMETRY
    # --------------------------------------------------------

    carbon_positions = (
        positions[:N_GRAPHENE_CARBON]
    )

    ligand_positions = (
        positions[N_GRAPHENE_TOTAL:]
    )

    current_graphene_z = float(
        np.mean(
            carbon_positions[:, 2]
        )
    )

    current_ligand_top = float(
        np.max(
            ligand_positions[:, 2]
        )
    )

    ligand_height = (
        current_ligand_top
        - current_graphene_z
    )

    print("CURRENT GEOMETRY")
    print("----------------")
    print(
        "Graphene carbon plane:",
        f"{current_graphene_z:.6f} nm"
    )

    print(
        "Ligand top:",
        f"{current_ligand_top:.6f} nm"
    )

    print(
        "Ligand height above graphene:",
        f"{ligand_height:.6f} nm"
    )

    # --------------------------------------------------------
    # TRANSLATE ENTIRE SOLUTE
    #
    # No internal coordinates change.
    # --------------------------------------------------------

    z_shift = (
        TARGET_GRAPHENE_CARBON_Z_NM
        - current_graphene_z
    )

    supported = (
        positions.copy()
    )

    supported[:, 2] += (
        z_shift
    )

    new_carbon_z = float(
        np.mean(
            supported[
                :N_GRAPHENE_CARBON,
                2
            ]
        )
    )

    graphene_all = (
        supported[
            :N_GRAPHENE_TOTAL
        ]
    )

    ligand = (
        supported[
            N_GRAPHENE_TOTAL:
        ]
    )

    graphene_min_z = float(
        np.min(
            graphene_all[:, 2]
        )
    )

    graphene_max_z = float(
        np.max(
            graphene_all[:, 2]
        )
    )

    ligand_top = float(
        np.max(
            ligand[:, 2]
        )
    )

    print()
    print("SUPPORTED GEOMETRY")
    print("------------------")

    print(
        "Graphene carbon plane:",
        f"{new_carbon_z:.6f} nm"
    )

    print(
        "Lowest IFF graphene particle:",
        f"{graphene_min_z:.6f} nm"
    )

    print(
        "Highest IFF graphene particle:",
        f"{graphene_max_z:.6f} nm"
    )

    print(
        "Ligand highest point:",
        f"{ligand_top:.6f} nm"
    )

    print(
        "Lower support boundary:",
        f"{LOWER_WALL_Z_NM:.6f} nm"
    )

    print(
        "Top solvent boundary:",
        f"{TOP_WALL_Z_NM:.6f} nm"
    )

    # --------------------------------------------------------
    # VALIDATE SUPPORT CLEARANCE
    # --------------------------------------------------------

    lower_clearance = (
        graphene_min_z
        - LOWER_WALL_Z_NM
    )

    water_headroom = (
        TOP_WALL_Z_NM
        - ligand_top
    )

    print()

    print(
        "Graphene-to-lower-boundary clearance:",
        f"{lower_clearance:.6f} nm"
    )

    print(
        "Water headroom above ligand:",
        f"{water_headroom:.6f} nm"
    )

    if lower_clearance <= 0.0:

        raise RuntimeError(
            "Graphene crosses the lower "
            "support boundary."
        )

    if water_headroom < 3.0:

        raise RuntimeError(
            "Less than 3 nm solvent "
            "headroom above ligand."
        )

    # --------------------------------------------------------
    # APPROXIMATE ONE-SIDED WATER REGION
    #
    # Water oxygen centers will later be allowed only above
    # the graphene carbon plane and below the top boundary.
    #
    # Overlap removal with graphene/ligand will further
    # reduce the actual water count.
    # --------------------------------------------------------

    a = box[0]
    b = box[1]

    surface_area_nm2 = float(
        np.linalg.norm(
            np.cross(
                a,
                b
            )
        )
    )

    nominal_water_height = (
        TOP_WALL_Z_NM
        - new_carbon_z
    )

    nominal_water_volume_nm3 = (
        surface_area_nm2
        * nominal_water_height
    )

    nominal_water_volume_L = (
        nominal_water_volume_nm3
        * 1e-24
    )

    rough_water_count = (
        WATER_MOLARITY
        * nominal_water_volume_L
        * AVOGADRO
    )

    print()
    print("ONE-SIDED AQUEOUS REGION")
    print("------------------------")

    print(
        "Surface area:",
        f"{surface_area_nm2:.3f} nm^2"
    )

    print(
        "Nominal aqueous height:",
        f"{nominal_water_height:.3f} nm"
    )

    print(
        "Nominal aqueous volume:",
        f"{nominal_water_volume_nm3:.3f} nm^3"
    )

    print(
        "Rough bulk-water estimate:",
        f"{rough_water_count:.1f} waters"
    )

    print(
        "(Actual count will be lower "
        "after overlap removal.)"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    np.save(
        OUTPUT_POSITIONS,
        supported
    )

    metadata = {

        "model":
            "supported_graphene_approximation",

        "explicit_SiO2":
            False,

        "graphene_carbon_plane_z_nm":
            new_carbon_z,

        "lowest_graphene_particle_z_nm":
            graphene_min_z,

        "lower_support_boundary_z_nm":
            LOWER_WALL_Z_NM,

        "top_solvent_boundary_z_nm":
            TOP_WALL_Z_NM,

        "ligand_top_z_nm":
            ligand_top,

        "solvent_headroom_above_ligand_nm":
            water_headroom,

        "nominal_aqueous_height_nm":
            nominal_water_height,

        "nominal_aqueous_volume_nm3":
            nominal_water_volume_nm3,

        "rough_water_count":
            rough_water_count,

        "box_vectors_nm":
            box.tolist(),

        "note":
            (
                "Version-1 supported graphene model. "
                "No explicit SiO2 atoms or chemistry. "
                "Future MD will restrain graphene and "
                "confine solution to the exposed side."
            )
    }

    with open(
        OUTPUT_METADATA,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("SAVED")
    print("-----")

    print(
        "Positions:",
        OUTPUT_POSITIONS
    )

    print(
        "Metadata:",
        OUTPUT_METADATA
    )

    print()
    print(
        "SUPPORTED ONE-SIDED BOX DESIGN: PASS"
    )

    print()
    print(
        "No water has been added."
    )

    print(
        "No force-field parameters "
        "were modified."
    )


if __name__ == "__main__":
    main()
