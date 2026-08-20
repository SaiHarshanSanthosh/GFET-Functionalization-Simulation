from pathlib import Path
import json
import numpy as np

from ase.build import graphene
from ase.neighborlist import neighbor_list

from openmm import app, openmm, unit


# ============================================================
# INPUT FILES
# ============================================================

GRAPHENE_XML = Path(
    "parameters/iff/graphene_iff_bonded_oop.xml"
)

LIGAND_PRMTOP = Path(
    "parameters/gaff2/pyrene_peg5_neutral.prmtop"
)

LIGAND_PDB = Path(
    "structures/pyrene_peg5_minimized.pdb"
)

LIGAND_MOL2 = Path(
    "parameters/gaff2/pyrene_peg5_gaff2_neutral.mol2"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DIR = Path(
    "parameters/combined"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_SYSTEM = (
    OUTPUT_DIR
    / "pyrene_peg5_graphene_vacuum_assembly_nocutoff.xml"
)

OUTPUT_POSITIONS = (
    OUTPUT_DIR
    / "pyrene_peg5_graphene_vacuum_positions_nm.npy"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "pyrene_peg5_graphene_vacuum_metadata.json"
)

OUTPUT_XYZ = Path(
    "structures/pyrene_peg5_graphene_vacuum.xyz"
)


# ============================================================
# INITIAL ADSORPTION GEOMETRY
#
# Based on our validated rigid interaction scan.
#
# This is NOT claiming that 3.50 A is the final equilibrium
# separation in water.
#
# It is simply a physically reasonable starting configuration.
# ============================================================

PYRENE_GRAPHENE_DISTANCE_A = 3.50


# ============================================================
# IFF NONBONDED PARAMETERS
#
# 2017 graphitic IFF model
#
# cg1:
#     q       = +0.2 e
#     Rmin    = 3.79 A
#     epsilon = 0.063 kcal/mol
#
# cge:
#     q       = -0.1 e
#     LJ      = 0
# ============================================================

Q_CG1 = +0.2
Q_CGE = -0.1

RMIN_CG1_A = 3.79

EPS_CG1_KCAL = 0.063


# Convert IFF/CHARMM Rmin -> OpenMM sigma

SIGMA_CG1_NM = (
    RMIN_CG1_A
    / (2.0 ** (1.0 / 6.0))
    / 10.0
)

EPS_CG1_KJ = (
    EPS_CG1_KCAL
    * 4.184
)

SIGMA_CGE_NM = 0.0
EPS_CGE_KJ = 0.0


# Coulomb constant for optional diagnostic calculation

ONE_4PI_EPS0 = 138.935456


# ============================================================
# ROTATION HELPER
# ============================================================

def rotation_matrix(a, b):

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)

    c = float(
        np.dot(a, b)
    )

    if np.isclose(
        c,
        1.0
    ):

        return np.eye(3)

    if np.isclose(
        c,
        -1.0
    ):

        trial = np.array(
            [
                1.0,
                0.0,
                0.0
            ]
        )

        if abs(
            np.dot(a, trial)
        ) > 0.9:

            trial = np.array(
                [
                    0.0,
                    1.0,
                    0.0
                ]
            )

        axis = np.cross(
            a,
            trial
        )

        axis /= np.linalg.norm(
            axis
        )

        return (
            2.0
            * np.outer(
                axis,
                axis
            )
            - np.eye(3)
        )

    s = np.linalg.norm(v)

    K = np.array(
        [
            [
                0.0,
                -v[2],
                v[1]
            ],
            [
                v[2],
                0.0,
                -v[0]
            ],
            [
                -v[1],
                v[0],
                0.0
            ],
        ]
    )

    return (
        np.eye(3)
        + K
        + (K @ K)
        * (
            (1.0 - c)
            / (s * s)
        )
    )


# ============================================================
# IDENTIFY PYRENE ATOMS FROM GAFF2 MOL2
# ============================================================

def get_pyrene_indices(
    mol2_file
):

    indices = []

    in_atoms = False

    with open(
        mol2_file,
        "r"
    ) as f:

        for line in f:

            stripped = (
                line.strip()
            )

            if (
                stripped
                == "@<TRIPOS>ATOM"
            ):

                in_atoms = True

                continue

            if (
                in_atoms
                and stripped.startswith(
                    "@<TRIPOS>"
                )
            ):

                break

            if (
                in_atoms
                and stripped
            ):

                fields = (
                    stripped.split()
                )

                # GAFF2 aromatic carbon = ca

                if (
                    len(fields) >= 6
                    and fields[5] == "ca"
                ):

                    indices.append(
                        int(
                            fields[0]
                        )
                        - 1
                    )

    if len(indices) != 16:

        raise RuntimeError(
            "Expected 16 aromatic "
            f"pyrene carbons; found {len(indices)}."
        )

    return indices


# ============================================================
# BUILD EXACT GRAPHENE GEOMETRY AND TOPOLOGY
# ============================================================

def build_graphene():

    sheet = graphene(
        formula="C2",
        a=2.46,
        size=(25, 25, 1),
        vacuum=20.0
    )

    sheet.pbc = (
        True,
        True,
        False
    )

    carbon_positions = (
        sheet.get_positions()
    )

    cell = (
        sheet.cell.array
    )

    n_carbon = len(
        carbon_positions
    )

    if n_carbon != 1250:

        raise RuntimeError(
            f"Expected 1250 graphene carbons; "
            f"found {n_carbon}."
        )

    # --------------------------------------------------------
    # IFF pi sites
    # --------------------------------------------------------

    normal = np.cross(
        cell[0],
        cell[1]
    )

    normal /= np.linalg.norm(
        normal
    )

    pi_up = (
        carbon_positions
        + 0.65 * normal
    )

    pi_down = (
        carbon_positions
        - 0.65 * normal
    )

    all_positions = np.vstack(
        [
            carbon_positions,
            pi_up,
            pi_down
        ]
    )

    if len(
        all_positions
    ) != 3750:

        raise RuntimeError(
            "Expected 3750 IFF graphene particles."
        )

    # --------------------------------------------------------
    # Periodic C-C topology
    # --------------------------------------------------------

    i_list, j_list, shifts = (
        neighbor_list(
            "ijS",
            sheet,
            cutoff=1.60,
            self_interaction=False
        )
    )

    cc_records = {}

    for i, j, shift in zip(
        i_list,
        j_list,
        shifts
    ):

        i = int(i)
        j = int(j)

        if i < j:

            key = (
                i,
                j
            )

            canonical_shift = (
                np.asarray(
                    shift,
                    dtype=int
                )
            )

        elif j < i:

            key = (
                j,
                i
            )

            canonical_shift = (
                -np.asarray(
                    shift,
                    dtype=int
                )
            )

        else:

            continue

        cc_records[
            key
        ] = canonical_shift

    cc_bonds = sorted(
        cc_records.keys()
    )

    if len(
        cc_bonds
    ) != 1875:

        raise RuntimeError(
            f"Expected 1875 C-C bonds; "
            f"found {len(cc_bonds)}."
        )

    # --------------------------------------------------------
    # Add C-pi topology bonds
    # --------------------------------------------------------

    cpi_bonds = []

    for carbon in range(
        n_carbon
    ):

        pi_up_index = (
            n_carbon
            + carbon
        )

        pi_down_index = (
            2 * n_carbon
            + carbon
        )

        cpi_bonds.append(
            (
                carbon,
                pi_up_index
            )
        )

        cpi_bonds.append(
            (
                carbon,
                pi_down_index
            )
        )

    graphene_bonds = (
        cc_bonds
        + cpi_bonds
    )

    if len(
        graphene_bonds
    ) != 4375:

        raise RuntimeError(
            "Expected 4375 total "
            "graphene topology bonds."
        )

    return (
        carbon_positions,
        all_positions,
        graphene_bonds,
        cell
    )


# ============================================================
# LOAD LIGAND SYSTEM
# ============================================================

def load_ligand():

    prmtop = (
        app.AmberPrmtopFile(
            str(
                LIGAND_PRMTOP
            )
        )
    )

    system = (
        prmtop.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False
        )
    )

    if (
        system.getNumParticles()
        != 70
    ):

        raise RuntimeError(
            "Expected 70 ligand atoms."
        )

    if (
        system.getNumConstraints()
        != 0
    ):

        raise RuntimeError(
            "Expected zero ligand constraints."
        )

    # Identify forces

    bond_forces = [
        force
        for force in system.getForces()
        if isinstance(
            force,
            openmm.HarmonicBondForce
        )
    ]

    angle_forces = [
        force
        for force in system.getForces()
        if isinstance(
            force,
            openmm.HarmonicAngleForce
        )
    ]

    torsion_forces = [
        force
        for force in system.getForces()
        if isinstance(
            force,
            openmm.PeriodicTorsionForce
        )
    ]

    nb_forces = [
        force
        for force in system.getForces()
        if isinstance(
            force,
            openmm.NonbondedForce
        )
    ]

    if (
        len(bond_forces) != 1
        or len(angle_forces) != 1
        or len(torsion_forces) != 1
        or len(nb_forces) != 1
    ):

        raise RuntimeError(
            "Unexpected ligand force inventory."
        )

    bonds = bond_forces[0]
    angles = angle_forces[0]
    torsions = torsion_forces[0]
    nonbonded = nb_forces[0]

    if bonds.getNumBonds() != 73:

        raise RuntimeError(
            "Expected 73 ligand bonds."
        )

    if angles.getNumAngles() != 127:

        raise RuntimeError(
            "Expected 127 ligand angles."
        )

    if torsions.getNumTorsions() != 234:

        raise RuntimeError(
            "Expected 234 ligand torsions."
        )

    if (
        nonbonded.getNumExceptions()
        != 357
    ):

        raise RuntimeError(
            "Expected 357 ligand "
            "nonbonded exceptions."
        )

    return (
        prmtop,
        system,
        bonds,
        angles,
        torsions,
        nonbonded
    )


# ============================================================
# ORIENT LIGAND
# ============================================================

def orient_ligand(
    coordinates,
    pyrene_indices
):

    coords = np.asarray(
        coordinates,
        dtype=float
    ).copy()

    pyrene = (
        coords[
            pyrene_indices
        ]
    )

    pyrene_center = (
        pyrene.mean(
            axis=0
        )
    )

    centered = (
        pyrene
        - pyrene_center
    )

    # Best-fit plane via SVD

    _, _, vh = (
        np.linalg.svd(
            centered,
            full_matrices=False
        )
    )

    pyrene_normal = (
        vh[-1]
    )

    # Rotate pyrene plane parallel to XY graphene plane

    R = rotation_matrix(
        pyrene_normal,
        np.array(
            [
                0.0,
                0.0,
                1.0
            ]
        )
    )

    coords = (
        coords
        - pyrene_center
    ) @ R.T

    pyrene_set = set(
        pyrene_indices
    )

    non_pyrene = [
        i
        for i in range(
            len(coords)
        )
        if i not in pyrene_set
    ]

    # Make PEG preferentially point away from graphene.

    if (
        coords[
            non_pyrene,
            2
        ].mean()
        < 0.0
    ):

        flip = np.diag(
            [
                1.0,
                -1.0,
                -1.0
            ]
        )

        coords = (
            coords
            @ flip.T
        )

    # Re-center exactly on pyrene centroid

    coords -= (
        coords[
            pyrene_indices
        ].mean(
            axis=0
        )
    )

    plane_deviation = float(
        np.max(
            np.abs(
                coords[
                    pyrene_indices,
                    2
                ]
            )
        )
    )

    return (
        coords,
        plane_deviation
    )


# ============================================================
# COPY LIGAND BONDED FORCES INTO COMBINED SYSTEM
# ============================================================

def add_ligand_bonded_forces(
    combined,
    ligand_bonds,
    ligand_angles,
    ligand_torsions,
    offset
):

    # --------------------------------------------------------
    # Bonds
    # --------------------------------------------------------

    bonds = (
        openmm.HarmonicBondForce()
    )

    bonds.setName(
        "GAFF2 ligand bonds"
    )

    bonds.setForceGroup(
        3
    )

    for i in range(
        ligand_bonds.getNumBonds()
    ):

        (
            p1,
            p2,
            length,
            k
        ) = (
            ligand_bonds
            .getBondParameters(i)
        )

        bonds.addBond(
            int(p1) + offset,
            int(p2) + offset,
            length,
            k
        )

    combined.addForce(
        bonds
    )

    # --------------------------------------------------------
    # Angles
    # --------------------------------------------------------

    angles = (
        openmm.HarmonicAngleForce()
    )

    angles.setName(
        "GAFF2 ligand angles"
    )

    angles.setForceGroup(
        4
    )

    for i in range(
        ligand_angles.getNumAngles()
    ):

        (
            p1,
            p2,
            p3,
            theta,
            k
        ) = (
            ligand_angles
            .getAngleParameters(i)
        )

        angles.addAngle(
            int(p1) + offset,
            int(p2) + offset,
            int(p3) + offset,
            theta,
            k
        )

    combined.addForce(
        angles
    )

    # --------------------------------------------------------
    # Proper + improper torsions
    # --------------------------------------------------------

    torsions = (
        openmm.PeriodicTorsionForce()
    )

    torsions.setName(
        "GAFF2 ligand torsions"
    )

    torsions.setForceGroup(
        5
    )

    for i in range(
        ligand_torsions.getNumTorsions()
    ):

        (
            p1,
            p2,
            p3,
            p4,
            periodicity,
            phase,
            k
        ) = (
            ligand_torsions
            .getTorsionParameters(i)
        )

        torsions.addTorsion(
            int(p1) + offset,
            int(p2) + offset,
            int(p3) + offset,
            int(p4) + offset,
            periodicity,
            phase,
            k
        )

    combined.addForce(
        torsions
    )

    return (
        bonds,
        angles,
        torsions
    )


# ============================================================
# BUILD ONE UNIFIED NONBONDED FORCE
#
# This is important:
#
# graphene-graphene
# ligand-ligand
# graphene-ligand
#
# all live in the SAME NonbondedForce.
#
# Therefore OpenMM automatically applies the same
# Lorentz-Berthelot mixing rules to cross interactions.
#
# This checkpoint still uses NoCutoff.
# PME comes later with water/PBS.
# ============================================================

def build_combined_nonbonded(
    combined,
    n_graphene,
    n_carbon,
    graphene_bonds,
    ligand_nb,
    ligand_offset
):

    nb = (
        openmm.NonbondedForce()
    )

    nb.setName(
        "IFF + GAFF2 unified "
        "nonbonded assembly"
    )

    nb.setForceGroup(
        6
    )

    nb.setNonbondedMethod(
        openmm.NonbondedForce.NoCutoff
    )

    # --------------------------------------------------------
    # Graphene cg1 particles
    # --------------------------------------------------------

    for _ in range(
        n_carbon
    ):

        nb.addParticle(
            Q_CG1
            * unit.elementary_charge,

            SIGMA_CG1_NM
            * unit.nanometer,

            EPS_CG1_KJ
            * unit.kilojoule_per_mole
        )

    # --------------------------------------------------------
    # Graphene cge particles
    # --------------------------------------------------------

    for _ in range(
        2 * n_carbon
    ):

        nb.addParticle(
            Q_CGE
            * unit.elementary_charge,

            SIGMA_CGE_NM
            * unit.nanometer,

            EPS_CGE_KJ
            * unit.kilojoule_per_mole
        )

    if (
        nb.getNumParticles()
        != n_graphene
    ):

        raise RuntimeError(
            "Graphene nonbonded particle "
            "count mismatch."
        )

    # --------------------------------------------------------
    # Copy ligand particle parameters EXACTLY from PRMTOP
    # --------------------------------------------------------

    ligand_charges = []
    ligand_sigmas = []
    ligand_epsilons = []

    for i in range(
        ligand_nb.getNumParticles()
    ):

        (
            q,
            sigma,
            epsilon
        ) = (
            ligand_nb
            .getParticleParameters(i)
        )

        nb.addParticle(
            q,
            sigma,
            epsilon
        )

        ligand_charges.append(
            q.value_in_unit(
                unit.elementary_charge
            )
        )

        ligand_sigmas.append(
            sigma.value_in_unit(
                unit.nanometer
            )
        )

        ligand_epsilons.append(
            epsilon.value_in_unit(
                unit.kilojoule_per_mole
            )
        )

    # --------------------------------------------------------
    # Graphene exceptions
    #
    # 1-2 excluded
    # 1-3 excluded
    # 1-4 full strength
    # --------------------------------------------------------

    nb.createExceptionsFromBonds(
        graphene_bonds,
        1.0,
        1.0
    )

    graphene_exception_count = (
        nb.getNumExceptions()
    )

    if (
        graphene_exception_count
        != 45000
    ):

        raise RuntimeError(
            "Expected 45000 graphene "
            f"exceptions; found "
            f"{graphene_exception_count}."
        )

    # --------------------------------------------------------
    # Copy AMBER/GAFF2 ligand exceptions EXACTLY.
    #
    # Do NOT regenerate them.
    #
    # This preserves the 1-4 scaling encoded by AmberTools.
    # --------------------------------------------------------

    for i in range(
        ligand_nb.getNumExceptions()
    ):

        (
            p1,
            p2,
            charge_product,
            sigma,
            epsilon
        ) = (
            ligand_nb
            .getExceptionParameters(i)
        )

        nb.addException(
            int(p1) + ligand_offset,
            int(p2) + ligand_offset,
            charge_product,
            sigma,
            epsilon
        )

    expected_exceptions = (
        45000
        + ligand_nb.getNumExceptions()
    )

    if (
        nb.getNumExceptions()
        != expected_exceptions
    ):

        raise RuntimeError(
            "Combined exception "
            "count mismatch."
        )

    nb.setExceptionsUsePeriodicBoundaryConditions(
        True
    )

    combined.addForce(
        nb
    )

    return (
        nb,
        np.asarray(
            ligand_charges
        ),
        np.asarray(
            ligand_sigmas
        ),
        np.asarray(
            ligand_epsilons
        )
    )


# ============================================================
# DIRECT CROSS-INTERACTION DIAGNOSTIC
#
# This does NOT add another force to the OpenMM system.
#
# It independently calculates only:
#
# graphene <-> ligand
#
# so we can verify that our starting geometry still gives
# approximately the attractive interaction seen in our
# rigid scan.
# ============================================================

def calculate_cross_energy(
    graphene_positions_A,
    ligand_positions_A,
    ligand_q,
    ligand_sigma,
    ligand_epsilon,
    n_carbon
):

    n_graphene = len(
        graphene_positions_A
    )

    graphene_q = np.concatenate(
        [
            np.full(
                n_carbon,
                Q_CG1
            ),
            np.full(
                2 * n_carbon,
                Q_CGE
            )
        ]
    )

    graphene_sigma = np.concatenate(
        [
            np.full(
                n_carbon,
                SIGMA_CG1_NM
            ),
            np.full(
                2 * n_carbon,
                SIGMA_CGE_NM
            )
        ]
    )

    graphene_epsilon = np.concatenate(
        [
            np.full(
                n_carbon,
                EPS_CG1_KJ
            ),
            np.full(
                2 * n_carbon,
                EPS_CGE_KJ
            )
        ]
    )

    if len(
        graphene_q
    ) != n_graphene:

        raise RuntimeError(
            "Cross-energy graphene "
            "parameter count mismatch."
        )

    g_nm = (
        graphene_positions_A
        / 10.0
    )

    l_nm = (
        ligand_positions_A
        / 10.0
    )

    difference = (
        g_nm[:, None, :]
        - l_nm[None, :, :]
    )

    r = np.linalg.norm(
        difference,
        axis=2
    )

    min_distance_A = float(
        np.min(r)
        * 10.0
    )

    if min_distance_A < 1.5:

        raise RuntimeError(
            "Ligand and graphene overlap: "
            f"minimum distance is "
            f"{min_distance_A:.3f} A."
        )

    # --------------------------------------------------------
    # Coulomb
    # --------------------------------------------------------

    q_product = (
        graphene_q[:, None]
        * ligand_q[None, :]
    )

    coulomb = float(
        np.sum(
            ONE_4PI_EPS0
            * q_product
            / r
        )
    )

    # --------------------------------------------------------
    # Lennard-Jones
    #
    # arithmetic sigma
    # geometric epsilon
    # --------------------------------------------------------

    sigma = (
        graphene_sigma[:, None]
        + ligand_sigma[None, :]
    ) / 2.0

    epsilon = np.sqrt(
        graphene_epsilon[:, None]
        * ligand_epsilon[None, :]
    )

    ratio = (
        sigma
        / r
    )

    lj = float(
        np.sum(
            4.0
            * epsilon
            * (
                ratio ** 12
                - ratio ** 6
            )
        )
    )

    return (
        lj,
        coulomb,
        lj + coulomb,
        min_distance_A
    )


# ============================================================
# FORCE-GROUP ENERGY HELPER
# ============================================================

def group_energy(
    context,
    group
):

    return (
        context
        .getState(
            getEnergy=True,
            groups=1 << group
        )
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Confirm files exist
    # --------------------------------------------------------

    for required in (
        GRAPHENE_XML,
        LIGAND_PRMTOP,
        LIGAND_PDB,
        LIGAND_MOL2
    ):

        if not required.exists():

            raise FileNotFoundError(
                f"Missing required file: "
                f"{required}"
            )

    # ========================================================
    # LOAD FROZEN GRAPHENE CHECKPOINT
    # ========================================================

    with open(
        GRAPHENE_XML,
        "r"
    ) as f:

        combined = (
            openmm.XmlSerializer
            .deserialize(
                f.read()
            )
        )

    if (
        combined.getNumParticles()
        != 3750
    ):

        raise RuntimeError(
            "Frozen graphene System "
            "should contain 3750 particles."
        )

    if (
        combined.getNumForces()
        != 3
    ):

        raise RuntimeError(
            "Frozen graphene checkpoint "
            "should contain exactly 3 forces."
        )

    print(
        "Loaded frozen graphene checkpoint:",
        GRAPHENE_XML
    )

    print(
        "Graphene particles:",
        combined.getNumParticles()
    )

    print(
        "Graphene forces:",
        combined.getNumForces()
    )

    # ========================================================
    # REBUILD GRAPHENE GEOMETRY + BOND GRAPH
    # ========================================================

    (
        graphene_carbon,
        graphene_all,
        graphene_bonds,
        cell
    ) = build_graphene()

    n_carbon = len(
        graphene_carbon
    )

    n_graphene = len(
        graphene_all
    )

    # ========================================================
    # LOAD EXACT NEUTRAL LIGAND FORCE FIELD
    # ========================================================

    (
        ligand_prmtop,
        ligand_system,
        ligand_bonds,
        ligand_angles,
        ligand_torsions,
        ligand_nb
    ) = load_ligand()

    n_ligand = (
        ligand_system
        .getNumParticles()
    )

    ligand_offset = (
        n_graphene
    )

    print()

    print(
        "Loaded neutral GAFF2 ligand"
    )

    print(
        "Ligand particles:",
        n_ligand
    )

    print(
        "Ligand bonds:",
        ligand_bonds.getNumBonds()
    )

    print(
        "Ligand angles:",
        ligand_angles.getNumAngles()
    )

    print(
        "Ligand torsions:",
        ligand_torsions.getNumTorsions()
    )

    print(
        "Ligand exceptions:",
        ligand_nb.getNumExceptions()
    )

    print(
        "Ligand particle offset:",
        ligand_offset
    )

    # ========================================================
    # APPEND LIGAND PARTICLES TO GRAPHENE SYSTEM
    # ========================================================

    for i in range(
        n_ligand
    ):

        mass = (
            ligand_system
            .getParticleMass(i)
        )

        combined.addParticle(
            mass
        )

    if (
        combined.getNumParticles()
        != 3820
    ):

        raise RuntimeError(
            "Expected 3820 total particles."
        )

    # ========================================================
    # COPY LIGAND BONDED PHYSICS
    # ========================================================

    (
        copied_bonds,
        copied_angles,
        copied_torsions
    ) = add_ligand_bonded_forces(
        combined,
        ligand_bonds,
        ligand_angles,
        ligand_torsions,
        ligand_offset
    )

    # ========================================================
    # BUILD UNIFIED NONBONDED PHYSICS
    # ========================================================

    (
        combined_nb,
        ligand_q,
        ligand_sigma,
        ligand_epsilon
    ) = build_combined_nonbonded(
        combined,
        n_graphene,
        n_carbon,
        graphene_bonds,
        ligand_nb,
        ligand_offset
    )

    print()

    print(
        "COMBINED FORCE FIELD"
    )

    print(
        "--------------------"
    )

    print(
        "Total particles:",
        combined.getNumParticles()
    )

    print(
        "Total forces:",
        combined.getNumForces()
    )

    print(
        "Combined nonbonded particles:",
        combined_nb.getNumParticles()
    )

    print(
        "Combined nonbonded exceptions:",
        combined_nb.getNumExceptions()
    )

    if (
        combined_nb.getNumParticles()
        != 3820
    ):

        raise RuntimeError(
            "Unified nonbonded force "
            "must contain 3820 particles."
        )

    # ========================================================
    # CHECK TOTAL FORMAL CHARGE
    # ========================================================

    total_charge = 0.0

    for i in range(
        combined_nb.getNumParticles()
    ):

        q, _, _ = (
            combined_nb
            .getParticleParameters(i)
        )

        total_charge += (
            q.value_in_unit(
                unit.elementary_charge
            )
        )

    print(
        "Combined formal charge:",
        f"{total_charge:+.12f} e"
    )

    if abs(
        total_charge
    ) > 1e-6:

        raise RuntimeError(
            "Combined graphene + ligand "
            "should be neutral."
        )

    # ========================================================
    # LOAD MINIMIZED LIGAND COORDINATES
    # ========================================================

    ligand_pdb = (
        app.PDBFile(
            str(
                LIGAND_PDB
            )
        )
    )

    ligand_positions_A = np.array(
        [
            [
                position.x,
                position.y,
                position.z
            ]
            for position
            in ligand_pdb.positions.value_in_unit(
                unit.angstrom
            )
        ],
        dtype=float
    )

    if (
        len(
            ligand_positions_A
        )
        != n_ligand
    ):

        raise RuntimeError(
            "Ligand PDB and topology "
            "particle counts differ."
        )

    # ========================================================
    # ORIENT PYRENE PARALLEL TO GRAPHENE
    # ========================================================

    pyrene_indices = (
        get_pyrene_indices(
            LIGAND_MOL2
        )
    )

    (
        ligand_oriented,
        pyrene_plane_deviation
    ) = orient_ligand(
        ligand_positions_A,
        pyrene_indices
    )

    # ========================================================
    # PLACE PYRENE ABOVE CENTER OF GRAPHENE
    # ========================================================

    graphene_center_xy = (
        graphene_carbon[
            :,
            :2
        ].mean(
            axis=0
        )
    )

    graphene_z = float(
        graphene_carbon[
            :,
            2
        ].mean()
    )

    ligand_oriented[
        :,
        0
    ] += (
        graphene_center_xy[0]
    )

    ligand_oriented[
        :,
        1
    ] += (
        graphene_center_xy[1]
    )

    ligand_oriented[
        :,
        2
    ] += (
        graphene_z
        + PYRENE_GRAPHENE_DISTANCE_A
    )

    # Confirm actual pyrene centroid separation

    pyrene_z = float(
        ligand_oriented[
            pyrene_indices,
            2
        ].mean()
    )

    actual_distance_A = (
        pyrene_z
        - graphene_z
    )

    print()

    print(
        "STARTING GEOMETRY"
    )

    print(
        "-----------------"
    )

    print(
        "Pyrene plane deviation:",
        f"{pyrene_plane_deviation:.6f} A"
    )

    print(
        "Target pyrene-graphene separation:",
        f"{PYRENE_GRAPHENE_DISTANCE_A:.3f} A"
    )

    print(
        "Actual pyrene-graphene separation:",
        f"{actual_distance_A:.6f} A"
    )

    if abs(
        actual_distance_A
        - PYRENE_GRAPHENE_DISTANCE_A
    ) > 1e-6:

        raise RuntimeError(
            "Pyrene placement failed."
        )

    # ========================================================
    # CROSS-INTERFACE DIAGNOSTIC
    # ========================================================

    (
        cross_lj,
        cross_coulomb,
        cross_total,
        minimum_distance_A
    ) = calculate_cross_energy(
        graphene_all,
        ligand_oriented,
        ligand_q,
        ligand_sigma,
        ligand_epsilon,
        n_carbon
    )

    print()

    print(
        "GRAPHENE <-> LIGAND DIAGNOSTIC"
    )

    print(
        "-------------------------------"
    )

    print(
        "Minimum inter-system distance:",
        f"{minimum_distance_A:.6f} A"
    )

    print(
        "Cross LJ energy:",
        f"{cross_lj:.6f} kJ/mol"
    )

    print(
        "Cross Coulomb energy:",
        f"{cross_coulomb:.6f} kJ/mol"
    )

    print(
        "Cross total energy:",
        f"{cross_total:.6f} kJ/mol"
    )

    if not np.isfinite(
        cross_total
    ):

        raise RuntimeError(
            "Cross-interface energy "
            "is not finite."
        )

    if cross_total >= 0:

        raise RuntimeError(
            "Expected attractive starting "
            "graphene-ligand interaction."
        )

    # ========================================================
    # BUILD COMBINED POSITIONS
    # ========================================================

    combined_positions_A = np.vstack(
        [
            graphene_all,
            ligand_oriented
        ]
    )

    combined_positions_nm = (
        combined_positions_A
        / 10.0
    )

    if (
        len(
            combined_positions_nm
        )
        != combined.getNumParticles()
    ):

        raise RuntimeError(
            "Combined position count "
            "does not match System."
        )

    positions_openmm = (
        unit.Quantity(
            [
                openmm.Vec3(
                    x,
                    y,
                    z
                )
                for x, y, z
                in combined_positions_nm
            ],
            unit.nanometer
        )
    )

    # ========================================================
    # OPENMM ENERGY SMOKE TEST
    #
    # STILL NOT MD.
    #
    # No minimization.
    # No integration.
    # ========================================================

    integrator = (
        openmm.VerletIntegrator(
            0.001
            * unit.picoseconds
        )
    )

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CUDA"
        )
    )

    context = openmm.Context(
        combined,
        integrator,
        platform
    )

    context.setPositions(
        positions_openmm
    )

    state = (
        context.getState(
            getEnergy=True,
            getForces=True
        )
    )

    total_energy = (
        state.getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if not np.isfinite(
        total_energy
    ):

        raise RuntimeError(
            "Combined potential energy "
            "is not finite."
        )

    print()

    print(
        "OPENMM COMBINED-SYSTEM SMOKE TEST"
    )

    print(
        "---------------------------------"
    )

    print(
        "OpenMM platform:",
        platform.getName()
    )

    print(
        "Total potential energy:",
        f"{total_energy:.6f} kJ/mol"
    )

    # --------------------------------------------------------
    # Energy components
    # --------------------------------------------------------

    labels = {
        0: "IFF graphene bonds",
        1: "IFF graphene angles",
        2: "IFF graphene OOP",
        3: "GAFF2 ligand bonds",
        4: "GAFF2 ligand angles",
        5: "GAFF2 ligand torsions",
        6: "Unified nonbonded"
    }

    component_sum = 0.0

    print()

    print(
        "ENERGY COMPONENTS"
    )

    print(
        "-----------------"
    )

    for group in range(
        7
    ):

        energy = group_energy(
            context,
            group
        )

        component_sum += (
            energy
        )

        print(
            f"{labels[group]:24s}: "
            f"{energy:14.6f} kJ/mol"
        )

    print(
        f"{'Component sum':24s}: "
        f"{component_sum:14.6f} kJ/mol"
    )

    if abs(
        component_sum
        - total_energy
    ) > 0.05:

        raise RuntimeError(
            "Force-group energies do not "
            "sum to total potential energy."
        )

    # --------------------------------------------------------
    # Maximum force sanity check
    # --------------------------------------------------------

    forces = (
        state.getForces(
            asNumpy=True
        )
        .value_in_unit(
            unit.kilojoule_per_mole
            / unit.nanometer
        )
    )

    force_magnitudes = (
        np.linalg.norm(
            forces,
            axis=1
        )
    )

    max_force = float(
        np.max(
            force_magnitudes
        )
    )

    max_force_particle = int(
        np.argmax(
            force_magnitudes
        )
    )

    print()

    print(
        "Maximum force magnitude:",
        f"{max_force:.6f} "
        "kJ/(mol nm)"
    )

    print(
        "Particle with maximum force:",
        max_force_particle
    )

    if not np.isfinite(
        max_force
    ):

        raise RuntimeError(
            "Combined forces are not finite."
        )

    print(
        "Combined-system energy/force "
        "smoke test: PASS"
    )

    # ========================================================
    # SAVE EXACT POSITIONS
    # ========================================================

    np.save(
        OUTPUT_POSITIONS,
        combined_positions_nm
    )

    # ========================================================
    # SAVE SYSTEM XML
    # ========================================================

    with open(
        OUTPUT_SYSTEM,
        "w"
    ) as f:

        f.write(
            openmm.XmlSerializer
            .serialize(
                combined
            )
        )

    # ========================================================
    # SAVE VISUALIZATION XYZ
    # ========================================================

    ligand_symbols = []

    for atom in (
        ligand_pdb
        .topology
        .atoms()
    ):

        if (
            atom.element
            is not None
        ):

            ligand_symbols.append(
                atom.element.symbol
            )

        else:

            ligand_symbols.append(
                atom.name[0]
            )

    if len(
        ligand_symbols
    ) != n_ligand:

        raise RuntimeError(
            "Ligand symbol count mismatch."
        )

    with open(
        OUTPUT_XYZ,
        "w"
    ) as f:

        f.write(
            f"{combined.getNumParticles()}\n"
        )

        f.write(
            "IFF graphene + neutral "
            "GAFF2 Pyrene-PEG5; "
            "X = cge pi pseudo-particle\n"
        )

        # Graphene carbon cores

        for position in (
            graphene_all[
                :n_carbon
            ]
        ):

            f.write(
                "C "
                f"{position[0]:.6f} "
                f"{position[1]:.6f} "
                f"{position[2]:.6f}\n"
            )

        # IFF cge sites

        for position in (
            graphene_all[
                n_carbon:
            ]
        ):

            f.write(
                "X "
                f"{position[0]:.6f} "
                f"{position[1]:.6f} "
                f"{position[2]:.6f}\n"
            )

        # Ligand

        for symbol, position in zip(
            ligand_symbols,
            ligand_oriented
        ):

            f.write(
                f"{symbol} "
                f"{position[0]:.6f} "
                f"{position[1]:.6f} "
                f"{position[2]:.6f}\n"
            )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "graphene_carbon_atoms":
            n_carbon,

        "graphene_pi_sites":
            2 * n_carbon,

        "graphene_total_particles":
            n_graphene,

        "ligand_particles":
            n_ligand,

        "ligand_offset":
            ligand_offset,

        "combined_particles":
            combined.getNumParticles(),

        "pyrene_graphene_starting_distance_A":
            PYRENE_GRAPHENE_DISTANCE_A,

        "pyrene_plane_deviation_A":
            pyrene_plane_deviation,

        "minimum_graphene_ligand_distance_A":
            minimum_distance_A,

        "cross_LJ_kJmol":
            cross_lj,

        "cross_Coulomb_kJmol":
            cross_coulomb,

        "cross_total_kJmol":
            cross_total,

        "total_formal_charge_e":
            total_charge,

        "nonbonded_method":
            "NoCutoff validation assembly",

        "production_ready":
            False,

        "notes":
            (
                "Vacuum assembly checkpoint only. "
                "Explicit water/PBS and PME not yet added."
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
        "SAVED CHECKPOINTS"
    )

    print(
        "-----------------"
    )

    print(
        "System:",
        OUTPUT_SYSTEM
    )

    print(
        "Positions:",
        OUTPUT_POSITIONS
    )

    print(
        "Metadata:",
        OUTPUT_METADATA
    )

    print(
        "Visualization:",
        OUTPUT_XYZ
    )

    print()

    print(
        "VACUUM ASSEMBLY VALIDATION: PASS"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is NOT the production MD system."
    )

    print(
        "The next stage is explicit water/PBS "
        "plus periodic electrostatics."
    )

    del context
    del integrator


if __name__ == "__main__":
    main()
