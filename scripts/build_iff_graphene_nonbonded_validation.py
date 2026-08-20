from pathlib import Path
import numpy as np

from ase.build import graphene
from ase.neighborlist import neighbor_list

from openmm import openmm, unit


# ============================================================
# FILES
# ============================================================

INPUT_XML = Path(
    "parameters/iff/graphene_iff_bonded_oop.xml"
)

OUTPUT_XML = Path(
    "parameters/iff/"
    "graphene_iff_full_terms_validation_nocutoff.xml"
)


# ============================================================
# PUBLISHED 2017 IFF NONBONDED PARAMETERS
#
# CHARMM-compatible 12-6 version
#
# cg1:
#     charge  = +0.2 e
#     Rmin    = 3.79 Angstrom
#     epsilon = 0.063 kcal/mol
#
# cge:
#     charge  = -0.1 e
#     no Lennard-Jones interaction
# ============================================================

Q_CG1 = +0.2
Q_CGE = -0.1

RMIN_CG1_A = 3.79
EPSILON_CG1_KCAL = 0.063


# ============================================================
# CHARMM -> OPENMM LJ CONVERSION
#
# CHARMM:
#
# E = epsilon * [
#       (Rmin/r)^12
#       - 2*(Rmin/r)^6
#     ]
#
# OpenMM:
#
# E = 4*epsilon * [
#       (sigma/r)^12
#       - (sigma/r)^6
#     ]
#
# therefore:
#
# Rmin = 2^(1/6) * sigma
# ============================================================

SIGMA_CG1_NM = (
    RMIN_CG1_A
    / (2.0 ** (1.0 / 6.0))
    / 10.0
)

EPSILON_CG1_KJ = (
    EPSILON_CG1_KCAL
    * 4.184
)


# cge has no LJ interaction.
#
# With epsilon = 0, sigma is irrelevant.

SIGMA_CGE_NM = 0.0
EPSILON_CGE_KJ = 0.0


# CHARMM-compatible 1-4 scaling

COULOMB_14_SCALE = 1.0
LJ_14_SCALE = 1.0


# ============================================================
# REBUILD EXACT GEOMETRY + CONNECTIVITY
# ============================================================

def build_geometry_and_bonds():

    sheet = graphene(
        formula="C2",
        a=2.46,
        size=(25, 25, 1),
        vacuum=20.0,
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

    # --------------------------------------------------------
    # Add the two cge pseudo-particles to every carbon
    # --------------------------------------------------------

    normal = np.cross(
        cell[0],
        cell[1]
    )

    normal /= np.linalg.norm(
        normal
    )

    pi_above = (
        carbon_positions
        + 0.65 * normal
    )

    pi_below = (
        carbon_positions
        - 0.65 * normal
    )

    all_positions = np.vstack(
        [
            carbon_positions,
            pi_above,
            pi_below
        ]
    )

    # --------------------------------------------------------
    # Find periodic C-C bonds
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

        cc_records[key] = (
            canonical_shift
        )

    cc_bonds = sorted(
        cc_records
    )

    # --------------------------------------------------------
    # Check carbon connectivity
    # --------------------------------------------------------

    carbon_neighbors = {
        i: set()
        for i in range(n_carbon)
    }

    for i, j in cc_bonds:

        carbon_neighbors[i].add(j)
        carbon_neighbors[j].add(i)

    if not all(
        len(carbon_neighbors[i]) == 3
        for i in range(n_carbon)
    ):

        raise RuntimeError(
            "Every periodic graphene carbon "
            "must have exactly 3 carbon neighbors."
        )

    expected_cc = (
        n_carbon
        * 3
        // 2
    )

    if len(cc_bonds) != expected_cc:

        raise RuntimeError(
            f"Expected {expected_cc} C-C bonds, "
            f"found {len(cc_bonds)}."
        )

    # --------------------------------------------------------
    # Add C-pi topology bonds
    # --------------------------------------------------------

    cpi_bonds = []

    for i in range(n_carbon):

        pi_up = (
            n_carbon
            + i
        )

        pi_down = (
            2 * n_carbon
            + i
        )

        cpi_bonds.append(
            (
                i,
                pi_up
            )
        )

        cpi_bonds.append(
            (
                i,
                pi_down
            )
        )

    all_bonds = (
        cc_bonds
        + cpi_bonds
    )

    return (
        all_positions,
        n_carbon,
        all_bonds
    )


# ============================================================
# FIND SHORTEST BOND DISTANCES
#
# Used to independently verify OpenMM's:
#
# 1-2
# 1-3
# 1-4
#
# exception generation.
# ============================================================

def shortest_bond_distances(
    n_particles,
    bonds,
    max_depth=3
):

    adjacency = [
        set()
        for _ in range(n_particles)
    ]

    for i, j in bonds:

        adjacency[i].add(j)
        adjacency[j].add(i)

    pair_distance = {}

    for start in range(
        n_particles
    ):

        visited = {
            start
        }

        frontier = {
            start
        }

        for depth in range(
            1,
            max_depth + 1
        ):

            next_frontier = set()

            for node in frontier:

                for nbr in adjacency[node]:

                    if nbr not in visited:

                        next_frontier.add(
                            nbr
                        )

            for node in next_frontier:

                if start < node:

                    pair_distance[
                        (
                            start,
                            node
                        )
                    ] = depth

            visited.update(
                next_frontier
            )

            frontier = (
                next_frontier
            )

            if not frontier:

                break

    return pair_distance


# ============================================================
# UNIT TEST LJ CONVERSION
#
# At the published Rmin = 3.79 A,
# two cg1 sites should have:
#
#     E_LJ = -epsilon
#
# This tests the CHARMM -> OpenMM conversion directly.
# ============================================================

def validate_lj_conversion():

    toy = openmm.System()

    toy.addParticle(
        10.011
        * unit.dalton
    )

    toy.addParticle(
        10.011
        * unit.dalton
    )

    nb = openmm.NonbondedForce()

    nb.setNonbondedMethod(
        openmm.NonbondedForce.NoCutoff
    )

    for _ in range(2):

        nb.addParticle(
            0.0
            * unit.elementary_charge,

            SIGMA_CG1_NM
            * unit.nanometer,

            EPSILON_CG1_KJ
            * unit.kilojoule_per_mole
        )

    toy.addForce(
        nb
    )

    integrator = (
        openmm.VerletIntegrator(
            0.001
            * unit.picoseconds
        )
    )

    context = openmm.Context(
        toy,
        integrator,
        openmm.Platform.getPlatformByName(
            "Reference"
        )
    )

    positions = unit.Quantity(
        [
            openmm.Vec3(
                0.0,
                0.0,
                0.0
            ),

            openmm.Vec3(
                RMIN_CG1_A / 10.0,
                0.0,
                0.0
            )
        ],
        unit.nanometer
    )

    context.setPositions(
        positions
    )

    energy = (
        context
        .getState(
            getEnergy=True
        )
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    expected = (
        -EPSILON_CG1_KJ
    )

    print()

    print(
        "LJ CONVERSION UNIT TEST"
    )

    print(
        "-----------------------"
    )

    print(
        f"OpenMM energy at 3.79 A: "
        f"{energy:.9f} kJ/mol"
    )

    print(
        f"Expected energy:          "
        f"{expected:.9f} kJ/mol"
    )

    if abs(
        energy
        - expected
    ) > 1e-6:

        raise RuntimeError(
            "Rmin -> OpenMM sigma "
            "conversion failed."
        )

    print(
        "Rmin -> OpenMM sigma "
        "validation: PASS"
    )

    del context
    del integrator


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_XML.exists():

        raise FileNotFoundError(
            f"Missing frozen checkpoint: "
            f"{INPUT_XML}"
        )

    (
        all_positions,
        n_carbon,
        all_bonds
    ) = build_geometry_and_bonds()

    n_total = len(
        all_positions
    )

    # --------------------------------------------------------
    # Exact expected topology
    # --------------------------------------------------------

    if (
        n_carbon != 1250
        or n_total != 3750
        or len(all_bonds) != 4375
    ):

        raise RuntimeError(
            "Unexpected IFF topology: "
            f"C={n_carbon}, "
            f"particles={n_total}, "
            f"bonds={len(all_bonds)}"
        )

    # --------------------------------------------------------
    # Load frozen bonded + OOP model
    # --------------------------------------------------------

    with open(
        INPUT_XML,
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
        != n_total
    ):

        raise RuntimeError(
            f"Loaded System has "
            f"{system.getNumParticles()} "
            f"particles; expected {n_total}."
        )

    # Make sure we do NOT accidentally add
    # nonbonded physics twice.

    if any(
        isinstance(
            force,
            openmm.NonbondedForce
        )
        for force
        in system.getForces()
    ):

        raise RuntimeError(
            "Frozen checkpoint already contains "
            "a NonbondedForce."
        )

    print(
        "Loaded frozen bonded checkpoint:",
        INPUT_XML
    )

    print(
        "Particles:",
        system.getNumParticles()
    )

    print(
        "Existing forces:",
        system.getNumForces()
    )

    # ========================================================
    # BUILD PER-PARTICLE NONBONDED PARAMETERS
    # ========================================================

    charges = np.full(
        n_total,
        Q_CGE,
        dtype=float
    )

    sigmas = np.full(
        n_total,
        SIGMA_CGE_NM,
        dtype=float
    )

    epsilons = np.full(
        n_total,
        EPSILON_CGE_KJ,
        dtype=float
    )

    # First 1250 particles are cg1.

    charges[:n_carbon] = (
        Q_CG1
    )

    sigmas[:n_carbon] = (
        SIGMA_CG1_NM
    )

    epsilons[:n_carbon] = (
        EPSILON_CG1_KJ
    )

    total_charge = float(
        np.sum(
            charges
        )
    )

    print()

    print(
        "IFF NONBONDED PARAMETERS"
    )

    print(
        "------------------------"
    )

    print(
        f"cg1 charge: "
        f"{Q_CG1:+.3f} e"
    )

    print(
        f"cge charge: "
        f"{Q_CGE:+.3f} e"
    )

    print(
        f"Published cg1 Rmin: "
        f"{RMIN_CG1_A:.3f} A"
    )

    print(
        f"OpenMM cg1 sigma: "
        f"{SIGMA_CG1_NM:.9f} nm"
    )

    print(
        f"cg1 epsilon: "
        f"{EPSILON_CG1_KJ:.9f} kJ/mol"
    )

    print(
        f"cge epsilon: "
        f"{EPSILON_CGE_KJ:.1f} kJ/mol"
    )

    print(
        f"Total graphene charge: "
        f"{total_charge:.12f} e"
    )

    if abs(
        total_charge
    ) > 1e-10:

        raise RuntimeError(
            "IFF graphene should be "
            "electrically neutral."
        )

    # ========================================================
    # TEST LJ CONVERSION BEFORE USING IT
    # ========================================================

    validate_lj_conversion()

    # ========================================================
    # CREATE OPENMM NONBONDED FORCE
    #
    # IMPORTANT:
    #
    # NoCutoff is intentionally used ONLY for this validation
    # checkpoint.
    #
    # We are NOT claiming this is the final production
    # periodic electrostatics protocol.
    #
    # We will choose PME/cutoff settings when the actual
    # water + PBS simulation box is constructed.
    # ========================================================

    nonbonded = (
        openmm.NonbondedForce()
    )

    nonbonded.setName(
        "IFF graphene nonbonded "
        "VALIDATION ONLY"
    )

    nonbonded.setForceGroup(
        3
    )

    nonbonded.setNonbondedMethod(
        openmm.NonbondedForce.NoCutoff
    )

    for (
        q,
        sigma,
        epsilon
    ) in zip(
        charges,
        sigmas,
        epsilons
    ):

        nonbonded.addParticle(
            q
            * unit.elementary_charge,

            sigma
            * unit.nanometer,

            epsilon
            * unit.kilojoule_per_mole
        )

    # ========================================================
    # CREATE 1-2 / 1-3 / 1-4 EXCEPTIONS
    #
    # 1-2 = excluded
    # 1-3 = excluded
    # 1-4 = full CHARMM-compatible strength
    # ========================================================

    nonbonded.createExceptionsFromBonds(
        all_bonds,
        COULOMB_14_SCALE,
        LJ_14_SCALE
    )

    # This has no effect while NoCutoff is used.
    #
    # We store it now because in the eventual periodic model,
    # graphene bonds can cross the simulation-cell boundary.
    # OpenMM should then apply minimum-image PBC to the
    # corresponding 1-4 exception terms.

    nonbonded.setExceptionsUsePeriodicBoundaryConditions(
        True
    )

    system.addForce(
        nonbonded
    )

    # ========================================================
    # INDEPENDENT EXCEPTION VALIDATION
    #
    # Do not simply trust OpenMM.
    #
    # Independently compute shortest paths through the bond
    # network and compare them to OpenMM's exception table.
    # ========================================================

    pair_distance = (
        shortest_bond_distances(
            n_total,
            all_bonds,
            max_depth=3
        )
    )

    expected_pairs = set(
        pair_distance
    )

    actual_pairs = set()

    counts = {
        1: 0,
        2: 0,
        3: 0
    }

    for ex_index in range(
        nonbonded.getNumExceptions()
    ):

        (
            p1,
            p2,
            qprod,
            sigma,
            epsilon
        ) = (
            nonbonded
            .getExceptionParameters(
                ex_index
            )
        )

        p1 = int(p1)
        p2 = int(p2)

        pair = tuple(
            sorted(
                (
                    p1,
                    p2
                )
            )
        )

        actual_pairs.add(
            pair
        )

        if pair not in pair_distance:

            raise RuntimeError(
                f"Unexpected exception pair: "
                f"{pair}"
            )

        depth = (
            pair_distance[
                pair
            ]
        )

        counts[
            depth
        ] += 1

        qprod_value = (
            qprod
            .value_in_unit(
                unit.elementary_charge ** 2
            )
        )

        sigma_value = (
            sigma
            .value_in_unit(
                unit.nanometer
            )
        )

        epsilon_value = (
            epsilon
            .value_in_unit(
                unit.kilojoule_per_mole
            )
        )

        # ----------------------------------------------------
        # 1-2 and 1-3 must be fully excluded
        # ----------------------------------------------------

        if depth in (
            1,
            2
        ):

            if (
                abs(qprod_value) > 1e-12
                or abs(epsilon_value) > 1e-12
            ):

                raise RuntimeError(
                    f"1-{depth + 1} pair "
                    f"{pair} was not "
                    f"fully excluded."
                )

        # ----------------------------------------------------
        # 1-4 must remain at full strength
        # ----------------------------------------------------

        elif depth == 3:

            expected_qprod = (
                charges[p1]
                * charges[p2]
                * COULOMB_14_SCALE
            )

            expected_sigma = (
                sigmas[p1]
                + sigmas[p2]
            ) / 2.0

            expected_epsilon = (
                np.sqrt(
                    epsilons[p1]
                    * epsilons[p2]
                )
                * LJ_14_SCALE
            )

            if abs(
                qprod_value
                - expected_qprod
            ) > 1e-10:

                raise RuntimeError(
                    f"Wrong 1-4 Coulomb "
                    f"scaling for pair {pair}."
                )

            if abs(
                sigma_value
                - expected_sigma
            ) > 1e-10:

                raise RuntimeError(
                    f"Wrong 1-4 sigma "
                    f"for pair {pair}."
                )

            if abs(
                epsilon_value
                - expected_epsilon
            ) > 1e-10:

                raise RuntimeError(
                    f"Wrong 1-4 LJ scaling "
                    f"for pair {pair}."
                )

    # --------------------------------------------------------
    # Make sure OpenMM generated exactly the same pair set
    # our independent graph calculation predicts.
    # --------------------------------------------------------

    if (
        actual_pairs
        != expected_pairs
    ):

        raise RuntimeError(
            "OpenMM exception set differs "
            "from independent bond-graph "
            "calculation: "
            f"missing="
            f"{len(expected_pairs - actual_pairs)}, "
            f"extra="
            f"{len(actual_pairs - expected_pairs)}."
        )

    print()

    print(
        "NONBONDED EXCEPTION VALIDATION"
    )

    print(
        "------------------------------"
    )

    print(
        "1-2 excluded pairs:",
        counts[1]
    )

    print(
        "1-3 excluded pairs:",
        counts[2]
    )

    print(
        "1-4 full-strength pairs:",
        counts[3]
    )

    print(
        "Total exceptions:",
        nonbonded.getNumExceptions()
    )

    print(
        "Exception topology validation: PASS"
    )

    # ========================================================
    # READ BACK EVERY PARTICLE PARAMETER
    #
    # Makes sure OpenMM stored exactly what we gave it.
    # ========================================================

    for i in range(
        n_total
    ):

        (
            q,
            sigma,
            epsilon
        ) = (
            nonbonded
            .getParticleParameters(i)
        )

        q = q.value_in_unit(
            unit.elementary_charge
        )

        sigma = sigma.value_in_unit(
            unit.nanometer
        )

        epsilon = epsilon.value_in_unit(
            unit.kilojoule_per_mole
        )

        if abs(
            q - charges[i]
        ) > 1e-12:

            raise RuntimeError(
                f"Charge mismatch "
                f"at particle {i}."
            )

        if abs(
            sigma - sigmas[i]
        ) > 1e-12:

            raise RuntimeError(
                f"Sigma mismatch "
                f"at particle {i}."
            )

        if abs(
            epsilon
            - epsilons[i]
        ) > 1e-12:

            raise RuntimeError(
                f"Epsilon mismatch "
                f"at particle {i}."
            )

    print(
        "Per-particle nonbonded "
        "parameter validation: PASS"
    )

    # ========================================================
    # FINITE-CLUSTER ENERGY SMOKE TEST
    #
    # IMPORTANT:
    #
    # NoCutoff does NOT represent an infinite periodic
    # graphene sheet.
    #
    # This number is NOT a physical observable.
    #
    # We only check that:
    #
    #     OpenMM builds the Context
    #     the resulting energy is finite
    # ========================================================

    positions = unit.Quantity(
        [
            openmm.Vec3(
                x,
                y,
                z
            )
            for x, y, z
            in (
                all_positions
                / 10.0
            )
        ],
        unit.nanometer
    )

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
        system,
        integrator,
        platform
    )

    context.setPositions(
        positions
    )

    nb_energy = (
        context
        .getState(
            getEnergy=True,
            groups=1 << 3
        )
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if not np.isfinite(
        nb_energy
    ):

        raise RuntimeError(
            "Nonbonded smoke-test "
            "energy is not finite."
        )

    print()

    print(
        "NONBONDED SMOKE TEST"
    )

    print(
        "--------------------"
    )

    print(
        f"Finite-cluster nonbonded energy: "
        f"{nb_energy:.6f} kJ/mol"
    )

    print(
        "OpenMM platform:",
        platform.getName()
    )

    print(
        "Finite-energy smoke test: PASS"
    )

    print(
        "NOTE: Do NOT interpret this "
        "NoCutoff energy physically."
    )

    del context
    del integrator

    # ========================================================
    # SAVE VALIDATION CHECKPOINT
    #
    # Contains all IFF force terms, but the nonbonded method
    # is intentionally NOT the final production periodic
    # solvent/PBS protocol.
    # ========================================================

    OUTPUT_XML.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_XML,
        "w"
    ) as f:

        f.write(
            openmm.XmlSerializer
            .serialize(
                system
            )
        )

    print()

    print(
        "Saved validation checkpoint:",
        OUTPUT_XML
    )

    print(
        "IFF nonbonded parameter "
        "validation: PASS"
    )

    print(
        "This XML is NOT yet the "
        "production solvated/PBS MD system."
    )


if __name__ == "__main__":
    main()
