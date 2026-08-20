from pathlib import Path
import json
import numpy as np

from openmm import app, openmm, unit
from openmm.app import element


# ============================================================
# INPUTS
# ============================================================

SOLUTE_POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

BOX_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)

SOLUTE_SYSTEM_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_vacuum_assembly_nocutoff.xml"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_WATER_POSITIONS = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "one_sided_opc_geometry.json"
)

OUTPUT_XYZ = Path(
    "structures/"
    "pyrene_peg5_graphene_one_sided_opc.xyz"
)


# ============================================================
# COUNTS
# ============================================================

N_GRAPHENE_CARBON = 1250
N_GRAPHENE_TOTAL = 3750
N_SOLUTE = 3820


# ============================================================
# SUPPORTED-SURFACE GEOMETRY
# ============================================================

GRAPHENE_CARBON_Z_NM = 0.50

TOP_WATER_BOUNDARY_NM = 6.80


# ============================================================
# OPENMM SOLVATION GEOMETRY CONSTANTS
#
# Same radius convention used by OpenMM's addSolvent()
# when generating a TIP4P-Ew coordinate box.
# ============================================================

VDW_RADIUS_PER_SIGMA = (
    0.5612310241546864907
)

TIP4PEW_SIGMA_NM = (
    0.315365
)

WATER_RADIUS_NM = (
    TIP4PEW_SIGMA_NM
    * VDW_RADIUS_PER_SIGMA
)


# ============================================================
# DPBS CONCENTRATIONS
#
# mol/L for Sigma D8537 1X.
# ============================================================

ION_CONCENTRATIONS_1X = {

    "Na+": 0.153090,

    "K+": 0.004153,

    "Cl-": 0.139569,

    "H2PO4-": 0.001470,

    "HPO4--": 0.008102,
}


WATER_MOLARITY = 55.5


# ============================================================
# PERIODIC HELPERS
# ============================================================

def minimum_image_displacements(
    deltas,
    box,
    inverse_box
):

    fractional = (
        deltas @ inverse_box
    )

    fractional -= np.round(
        fractional
    )

    return (
        fractional @ box
    )


def wrap_water_by_oxygen(
    water_positions,
    oxygen_local_index,
    box,
    inverse_box
):

    water = np.asarray(
        water_positions,
        dtype=float
    ).copy()

    oxygen = (
        water[
            oxygen_local_index
        ]
    )

    fractional = (
        oxygen @ inverse_box
    )

    lattice_shift = (
        np.floor(
            fractional
        )
    )

    translation = (
        -lattice_shift @ box
    )

    water += translation

    return water


# ============================================================
# LOAD ACTUAL SOLUTE EXCLUSION RADII
# ============================================================

def load_solute_cutoffs():

    with open(
        SOLUTE_SYSTEM_FILE,
        "r"
    ) as f:

        system = (
            openmm.XmlSerializer
            .deserialize(
                f.read()
            )
        )

    if (
        system.getNumParticles()
        != N_SOLUTE
    ):

        raise RuntimeError(
            "Unexpected solute particle count."
        )

    nb_forces = [

        force

        for force in system.getForces()

        if isinstance(
            force,
            openmm.NonbondedForce
        )
    ]

    if len(nb_forces) != 1:

        raise RuntimeError(
            "Expected one NonbondedForce."
        )

    nb = nb_forces[0]

    cutoffs = np.zeros(
        N_SOLUTE,
        dtype=float
    )

    for i in range(
        N_SOLUTE
    ):

        _, sigma, epsilon = (
            nb.getParticleParameters(i)
        )

        sigma_nm = (
            sigma.value_in_unit(
                unit.nanometer
            )
        )

        epsilon_kj = (
            epsilon.value_in_unit(
                unit.kilojoule_per_mole
            )
        )

        radius = (
            WATER_RADIUS_NM
        )

        if abs(epsilon_kj) > 0.0:

            radius += (
                sigma_nm
                * VDW_RADIUS_PER_SIGMA
            )

        cutoffs[i] = radius

    return cutoffs


# ============================================================
# GENERATE PURE FOUR-SITE WATER BOX
# ============================================================

def generate_water_box(
    box
):

    topology = app.Topology()

    vectors = tuple(

        openmm.Vec3(
            *vector
        )
        * unit.nanometer

        for vector in box
    )

    topology.setPeriodicBoxVectors(
        vectors
    )

    modeller = app.Modeller(
        topology,
        []
    )

    forcefield = app.ForceField(
        "amber19/opc.xml"
    )

    modeller.addSolvent(

        forcefield,

        model="tip4pew",

        boxVectors=vectors,

        neutralize=False,

        ionicStrength=(
            0.0 * unit.molar
        )
    )

    positions = np.array(

        [

            [
                p.x,
                p.y,
                p.z
            ]

            for p in (
                modeller.positions
                .value_in_unit(
                    unit.nanometer
                )
            )
        ],

        dtype=float
    )

    records = []

    for residue in (
        modeller.topology.residues()
    ):

        atoms = list(
            residue.atoms()
        )

        if len(atoms) != 4:

            raise RuntimeError(
                "Expected four-site water."
            )

        oxygen_atoms = [

            atom

            for atom in atoms

            if (
                atom.element
                == element.oxygen
            )
        ]

        if len(oxygen_atoms) != 1:

            raise RuntimeError(
                "Expected exactly one "
                "oxygen per water."
            )

        oxygen_global = (
            oxygen_atoms[0].index
        )

        indices = [

            atom.index

            for atom in atoms
        ]

        oxygen_local = (
            indices.index(
                oxygen_global
            )
        )

        records.append(
            (
                indices,
                oxygen_local
            )
        )

    return (
        positions,
        records
    )


# ============================================================
# MAIN
# ============================================================

def main():

    solute = np.load(
        SOLUTE_POSITIONS_FILE
    )

    box = np.load(
        BOX_FILE
    )

    if solute.shape != (
        N_SOLUTE,
        3
    ):

        raise RuntimeError(
            "Unexpected solute shape."
        )

    if box.shape != (
        3,
        3
    ):

        raise RuntimeError(
            "Unexpected box shape."
        )

    inverse_box = np.linalg.inv(
        box
    )

    print(
        "Loaded supported solute."
    )

    print(
        "Particles:",
        len(solute)
    )

    print()

    print(
        "Graphene plane:",
        f"{GRAPHENE_CARBON_Z_NM:.3f} nm"
    )

    print(
        "Top water boundary:",
        f"{TOP_WATER_BOUNDARY_NM:.3f} nm"
    )

    # ========================================================
    # SOLUTE-WATER EXCLUSION RADII
    # ========================================================

    cutoffs = (
        load_solute_cutoffs()
    )

    print()

    print(
        "SOLUTE-WATER OVERLAP RADII"
    )

    print(
        "--------------------------"
    )

    print(
        "Minimum:",
        f"{cutoffs.min():.6f} nm"
    )

    print(
        "Maximum:",
        f"{cutoffs.max():.6f} nm"
    )

    # ========================================================
    # GENERATE FULL WATER BOX
    # ========================================================

    (
        all_positions,
        records
    ) = generate_water_box(
        box
    )

    print()

    print(
        "INITIAL OPC-COMPATIBLE WATER BOX"
    )

    print(
        "--------------------------------"
    )

    print(
        "Waters:",
        len(records)
    )

    print(
        "Particles:",
        len(all_positions)
    )

    if len(records) != 7377:

        raise RuntimeError(
            "Expected validated starting "
            "box of 7377 waters."
        )

    # ========================================================
    # FILTER WATER
    # ========================================================

    kept = []

    removed_below = 0

    removed_above = 0

    removed_overlap = 0

    minimum_retained_margin = np.inf

    for (
        indices,
        oxygen_local
    ) in records:

        water = (
            all_positions[
                indices
            ]
        )

        # ----------------------------------------------------
        # Wrap entire molecule according to oxygen.
        # ----------------------------------------------------

        water = (
            wrap_water_by_oxygen(
                water,
                oxygen_local,
                box,
                inverse_box
            )
        )

        oxygen = (
            water[
                oxygen_local
            ]
        )

        oxygen_z = float(
            oxygen[2]
        )

        # ----------------------------------------------------
        # One-sided experimental geometry
        # ----------------------------------------------------

        if (
            oxygen_z
            <= GRAPHENE_CARBON_Z_NM
        ):

            removed_below += 1

            continue

        if (
            oxygen_z
            >= TOP_WATER_BOUNDARY_NM
        ):

            removed_above += 1

            continue

        # ----------------------------------------------------
        # Solute overlap test using minimum-image distances.
        # ----------------------------------------------------

        deltas = (
            solute
            - oxygen
        )

        deltas = (
            minimum_image_displacements(
                deltas,
                box,
                inverse_box
            )
        )

        distances = np.linalg.norm(
            deltas,
            axis=1
        )

        margins = (
            distances
            - cutoffs
        )

        minimum_margin = float(
            np.min(
                margins
            )
        )

        if minimum_margin < 0.0:

            removed_overlap += 1

            continue

        minimum_retained_margin = min(
            minimum_retained_margin,
            minimum_margin
        )

        kept.append(
            water
        )

    kept = np.asarray(
        kept,
        dtype=float
    )

    n_kept = len(
        kept
    )

    print()

    print(
        "ONE-SIDED FILTER"
    )

    print(
        "----------------"
    )

    print(
        "Removed below graphene:",
        removed_below
    )

    print(
        "Removed above top boundary:",
        removed_above
    )

    print(
        "Removed for solute overlap:",
        removed_overlap
    )

    print(
        "Retained waters:",
        n_kept
    )

    total_accounted = (

        removed_below

        + removed_above

        + removed_overlap

        + n_kept
    )

    if (
        total_accounted
        != len(records)
    ):

        raise RuntimeError(
            "Water bookkeeping failed."
        )

    if kept.shape != (
        n_kept,
        4,
        3
    ):

        raise RuntimeError(
            "Unexpected retained-water shape."
        )

    print(
        "Minimum retained "
        "solute clearance:",
        f"{minimum_retained_margin:.6f} nm"
    )

    # ========================================================
    # EFFECTIVE SOLVENT VOLUME
    #
    # Convert actual water count back into the amount of
    # bulk water it represents at ~55.5 M.
    #
    # This is more useful for choosing integer ion counts
    # than using the entire simulation cell volume.
    # ========================================================

    effective_volume_L = (

        n_kept

        / (
            WATER_MOLARITY
            * 6.02214076e23
        )
    )

    effective_volume_nm3 = (
        effective_volume_L
        * 1e24
    )

    print()

    print(
        "EFFECTIVE AQUEOUS VOLUME"
    )

    print(
        "------------------------"
    )

    print(
        "From retained water count:",
        f"{effective_volume_nm3:.3f} nm^3"
    )

    # ========================================================
    # UPDATED DPBS TARGET COUNTS
    # ========================================================

    counts_1x = {}

    counts_001x = {}

    print()

    print(
        "UPDATED 1X DPBS TARGET COUNTS"
    )

    print(
        "-----------------------------"
    )

    for species, concentration in (
        ION_CONCENTRATIONS_1X.items()
    ):

        count = (

            concentration

            * effective_volume_L

            * 6.02214076e23
        )

        counts_1x[
            species
        ] = count

        print(
            f"{species:8s}: "
            f"{count:.4f}"
        )

    print()

    print(
        "UPDATED 0.01X DPBS TARGET COUNTS"
    )

    print(
        "---------------------------------"
    )

    for species, concentration in (
        ION_CONCENTRATIONS_1X.items()
    ):

        count = (

            concentration
            / 100.0

            * effective_volume_L

            * 6.02214076e23
        )

        counts_001x[
            species
        ] = count

        print(
            f"{species:8s}: "
            f"{count:.6f}"
        )

    # ========================================================
    # SAVE WATER POSITIONS
    # ========================================================

    np.save(
        OUTPUT_WATER_POSITIONS,
        kept
    )

    # ========================================================
    # VISUALIZATION XYZ
    #
    # We deliberately omit OPC M sites from this visualization
    # to make ASE much easier to look at.
    # Actual saved water array still contains all four sites.
    # ========================================================

    ligand_pdb = app.PDBFile(
        "structures/"
        "pyrene_peg5_minimized.pdb"
    )

    ligand_symbols = []

    for atom in (
        ligand_pdb.topology.atoms()
    ):

        if atom.element is None:

            ligand_symbols.append(
                "X"
            )

        else:

            ligand_symbols.append(
                atom.element.symbol
            )

    ligand = (
        solute[
            N_GRAPHENE_TOTAL:
        ]
    )

    n_xyz = (

        N_GRAPHENE_TOTAL

        + 70

        + 3 * n_kept
    )

    with open(
        OUTPUT_XYZ,
        "w"
    ) as f:

        f.write(
            f"{n_xyz}\n"
        )

        f.write(
            "Supported IFF graphene + "
            "GAFF2 Pyrene-PEG5 + "
            "one-sided OPC water; "
            "OPC M sites hidden\n"
        )

        # Graphene carbons

        for pos in (
            solute[
                :N_GRAPHENE_CARBON
            ]
        ):

            f.write(
                "C "
                f"{pos[0]*10:.6f} "
                f"{pos[1]*10:.6f} "
                f"{pos[2]*10:.6f}\n"
            )

        # IFF pi sites

        for pos in (
            solute[
                N_GRAPHENE_CARBON:
                N_GRAPHENE_TOTAL
            ]
        ):

            f.write(
                "X "
                f"{pos[0]*10:.6f} "
                f"{pos[1]*10:.6f} "
                f"{pos[2]*10:.6f}\n"
            )

        # Ligand

        for symbol, pos in zip(
            ligand_symbols,
            ligand
        ):

            f.write(
                f"{symbol} "
                f"{pos[0]*10:.6f} "
                f"{pos[1]*10:.6f} "
                f"{pos[2]*10:.6f}\n"
            )

        # Water O/H/H only

        for water in kept:

            for label, pos in zip(
                ("O", "H", "H"),
                water[:3]
            ):

                f.write(
                    f"{label} "
                    f"{pos[0]*10:.6f} "
                    f"{pos[1]*10:.6f} "
                    f"{pos[2]*10:.6f}\n"
                )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "initial_waters":
            len(records),

        "removed_below_graphene":
            removed_below,

        "removed_above_top_boundary":
            removed_above,

        "removed_for_solute_overlap":
            removed_overlap,

        "retained_waters":
            n_kept,

        "retained_water_particles":
            4 * n_kept,

        "minimum_retained_margin_nm":
            minimum_retained_margin,

        "effective_aqueous_volume_nm3":
            effective_volume_nm3,

        "expected_1X_counts":
            counts_1x,

        "expected_0.01X_counts":
            counts_001x,

        "graphene_plane_z_nm":
            GRAPHENE_CARBON_Z_NM,

        "top_water_boundary_nm":
            TOP_WATER_BOUNDARY_NM,

        "explicit_SiO2":
            False,

        "ions_added":
            False,

        "production_system":
            False,

        "note":
            (
                "Geometry-only supported-graphene "
                "one-sided OPC checkpoint. "
                "Graphene restraints, substrate wall, "
                "periodic electrostatics, and ions "
                "have not yet been added."
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

    print(
        "SAVED"
    )

    print(
        "-----"
    )

    print(
        "Water positions:",
        OUTPUT_WATER_POSITIONS
    )

    print(
        "Visualization:",
        OUTPUT_XYZ
    )

    print(
        "Metadata:",
        OUTPUT_METADATA
    )

    print()

    print(
        "ONE-SIDED OPC GEOMETRY: PASS"
    )

    print()

    print(
        "No MD has been run."
    )

    print(
        "No ions have been added."
    )

    print(
        "No support restraint has "
        "been added yet."
    )


if __name__ == "__main__":
    main()
