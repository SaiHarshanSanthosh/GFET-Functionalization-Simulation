from pathlib import Path
import csv
import numpy as np

from ase.build import graphene
from openmm import app, openmm, unit


# ============================================================
# FILES / CONSTANTS
# ============================================================

PRMTOP = Path("parameters/gaff2/pyrene_peg5.prmtop")
PDB = Path("structures/pyrene_peg5_minimized.pdb")
MOL2 = Path("parameters/gaff2/pyrene_peg5_gaff2.mol2")

OUT = Path("analysis/pyrene_graphene_rigid_scan.csv")


# Validated IFF graphene nonbonded parameters

Q_CG1 = +0.2
Q_CGE = -0.1

RMIN_CG1_A = 3.79
EPS_CG1_KCAL = 0.063

# CHARMM/IFF Rmin -> OpenMM sigma
SIGMA_CG1_NM = (
    RMIN_CG1_A
    / (2.0 ** (1.0 / 6.0))
    / 10.0
)

EPS_CG1_KJ = (
    EPS_CG1_KCAL
    * 4.184
)

# cge has charge but no LJ term
SIGMA_CGE_NM = 0.0
EPS_CGE_KJ = 0.0


# OpenMM Coulomb constant:
# kJ mol^-1 nm e^-2

ONE_4PI_EPS0 = 138.935456


# Distance means:
#
# pyrene carbon plane
#        ↓
# graphene carbon plane

DISTANCES_A = (
    list(
        np.arange(
            2.50,
            8.01,
            0.25
        )
    )
    + [
        10.0,
        12.0,
        15.0,
        20.0
    ]
)

REFERENCE_A = 20.0


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

    # Already aligned

    if np.isclose(
        c,
        1.0
    ):

        return np.eye(3)

    # Opposite directions:
    # choose a perpendicular rotation axis

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
            ]
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
# FIND THE PYRENE CARBONS
# ============================================================

def pyrene_indices_from_mol2(path):

    indices = []

    in_atoms = False

    with open(
        path,
        "r"
    ) as f:

        for line in f:

            text = line.strip()

            if text == "@<TRIPOS>ATOM":

                in_atoms = True

                continue

            if (
                in_atoms
                and text.startswith(
                    "@<TRIPOS>"
                )
            ):

                break

            if (
                in_atoms
                and text
            ):

                fields = (
                    text.split()
                )

                # MOL2 column 6 = atom type

                if (
                    len(fields) >= 6
                    and fields[5] == "ca"
                ):

                    # MOL2 indices are 1-based

                    indices.append(
                        int(
                            fields[0]
                        )
                        - 1
                    )

    # Pyrene = C16H10 aromatic system,
    # so our pyrene portion should have
    # 16 aromatic carbons.

    if len(
        indices
    ) != 16:

        raise RuntimeError(
            "Expected exactly 16 pyrene "
            f"'ca' atoms, found {len(indices)}."
        )

    return indices


# ============================================================
# REBUILD VALIDATED IFF GRAPHENE GEOMETRY
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

    carbon = (
        sheet.get_positions()
    )

    cell = (
        sheet.cell.array
    )

    normal = np.cross(
        cell[0],
        cell[1]
    )

    normal /= np.linalg.norm(
        normal
    )

    pi_up = (
        carbon
        + 0.65 * normal
    )

    pi_down = (
        carbon
        - 0.65 * normal
    )

    all_positions = np.vstack(
        [
            carbon,
            pi_up,
            pi_down
        ]
    )

    if (
        len(carbon) != 1250
        or len(all_positions) != 3750
    ):

        raise RuntimeError(
            "Unexpected graphene particle count."
        )

    return (
        carbon,
        all_positions
    )


# ============================================================
# READ GAFF2 PARAMETERS DIRECTLY FROM PRMTOP
# ============================================================

def ligand_parameters():

    prmtop = (
        app.AmberPrmtopFile(
            str(PRMTOP)
        )
    )

    ligand_system = (
        prmtop.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=None
        )
    )

    nb_forces = [
        force
        for force
        in ligand_system.getForces()
        if isinstance(
            force,
            openmm.NonbondedForce
        )
    ]

    if len(
        nb_forces
    ) != 1:

        raise RuntimeError(
            "Expected exactly one ligand "
            f"NonbondedForce, found {len(nb_forces)}."
        )

    nb = nb_forces[0]

    n = (
        ligand_system
        .getNumParticles()
    )

    charges = np.zeros(n)

    sigmas = np.zeros(n)

    epsilons = np.zeros(n)

    masses = []

    for i in range(n):

        (
            q,
            sigma,
            epsilon
        ) = (
            nb.getParticleParameters(i)
        )

        charges[i] = (
            q.value_in_unit(
                unit.elementary_charge
            )
        )

        sigmas[i] = (
            sigma.value_in_unit(
                unit.nanometer
            )
        )

        epsilons[i] = (
            epsilon.value_in_unit(
                unit.kilojoule_per_mole
            )
        )

        masses.append(
            ligand_system
            .getParticleMass(i)
        )

    return (
        charges,
        sigmas,
        epsilons,
        masses
    )


# ============================================================
# ORIENT PYRENE PARALLEL TO GRAPHENE
# ============================================================

def orient_ligand(
    coordinates,
    pyrene_indices
):

    coords = np.asarray(
        coordinates,
        dtype=float
    ).copy()

    pyrene = coords[
        pyrene_indices
    ]

    pyrene_center = (
        pyrene.mean(
            axis=0
        )
    )

    # Fit a plane through the 16 pyrene carbons.

    centered = (
        pyrene
        - pyrene_center
    )

    _, _, vh = (
        np.linalg.svd(
            centered,
            full_matrices=False
        )
    )

    normal = vh[-1]

    # Rotate pyrene normal onto +z.

    R = rotation_matrix(
        normal,
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

    # We want PEG mainly pointing away
    # from the graphene for this first scan.

    if (
        coords[
            non_pyrene,
            2
        ].mean()
        < 0.0
    ):

        # 180-degree rotation around x.
        #
        # This is a proper rigid rotation,
        # not a mirror reflection.

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

    # Center exactly on pyrene.

    coords -= (
        coords[
            pyrene_indices
        ].mean(
            axis=0
        )
    )

    plane_deviation = (
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
        float(
            plane_deviation
        ),
        non_pyrene
    )


# ============================================================
# LJ CROSS FORCE
# ============================================================

def make_lj_force(
    sigma,
    epsilon,
    group_a,
    group_b,
    force_group
):

    expression = (
        "4*eps*((sig/r)^12-(sig/r)^6);"
        "sig=0.5*(sigma1+sigma2);"
        "eps=sqrt(epsilon1*epsilon2)"
    )

    force = (
        openmm.CustomNonbondedForce(
            expression
        )
    )

    force.addPerParticleParameter(
        "sigma"
    )

    force.addPerParticleParameter(
        "epsilon"
    )

    force.setNonbondedMethod(
        openmm.CustomNonbondedForce.NoCutoff
    )

    force.setForceGroup(
        force_group
    )

    for s, e in zip(
        sigma,
        epsilon
    ):

        force.addParticle(
            [
                float(s),
                float(e)
            ]
        )

    force.addInteractionGroup(
        set(group_a),
        set(group_b)
    )

    return force


# ============================================================
# COULOMB CROSS FORCE
# ============================================================

def make_coulomb_force(
    charges,
    group_a,
    group_b,
    force_group
):

    force = (
        openmm.CustomNonbondedForce(
            "ONE_4PI_EPS0*charge1*charge2/r"
        )
    )

    force.addGlobalParameter(
        "ONE_4PI_EPS0",
        ONE_4PI_EPS0
    )

    force.addPerParticleParameter(
        "charge"
    )

    force.setNonbondedMethod(
        openmm.CustomNonbondedForce.NoCutoff
    )

    force.setForceGroup(
        force_group
    )

    for charge in charges:

        force.addParticle(
            [
                float(charge)
            ]
        )

    force.addInteractionGroup(
        set(group_a),
        set(group_b)
    )

    return force


# ============================================================
# READ ONE FORCE GROUP'S ENERGY
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

    for path in (
        PRMTOP,
        PDB,
        MOL2
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    # --------------------------------------------------------
    # Graphene
    # --------------------------------------------------------

    (
        graphene_carbon,
        graphene_all
    ) = build_graphene()

    n_graphene_carbon = (
        len(
            graphene_carbon
        )
    )

    n_graphene = (
        len(
            graphene_all
        )
    )

    # --------------------------------------------------------
    # Ligand GAFF2 parameters
    # --------------------------------------------------------

    (
        ligand_q,
        ligand_sigma,
        ligand_epsilon,
        ligand_masses
    ) = ligand_parameters()

    n_ligand = (
        len(
            ligand_q
        )
    )

    if n_ligand != 70:

        raise RuntimeError(
            "Expected 70 ligand atoms, "
            f"found {n_ligand}."
        )

    # --------------------------------------------------------
    # Minimized ligand coordinates
    # --------------------------------------------------------

    pdb = (
        app.PDBFile(
            str(PDB)
        )
    )

    ligand_coordinates = np.array(
        [
            [
                p.x,
                p.y,
                p.z
            ]
            for p
            in pdb.positions.value_in_unit(
                unit.angstrom
            )
        ]
    )

    if (
        len(
            ligand_coordinates
        )
        != n_ligand
    ):

        raise RuntimeError(
            "PDB and PRMTOP atom counts do not match."
        )

    # --------------------------------------------------------
    # Identify pyrene
    # --------------------------------------------------------

    pyrene_local = (
        pyrene_indices_from_mol2(
            MOL2
        )
    )

    (
        ligand_oriented,
        plane_deviation,
        non_pyrene
    ) = (
        orient_ligand(
            ligand_coordinates,
            pyrene_local
        )
    )

    print(
        "Graphene particles:",
        n_graphene
    )

    print(
        "Ligand atoms:",
        n_ligand
    )

    print(
        "Pyrene aromatic carbons:",
        len(
            pyrene_local
        )
    )

    print(
        "Ligand charge from PRMTOP:",
        f"{ligand_q.sum():+.6f} e"
    )

    print(
        "Maximum pyrene plane deviation:",
        f"{plane_deviation:.6f} A"
    )

    print(
        "Non-pyrene z range relative to pyrene plane:",
        f"{ligand_oriented[non_pyrene, 2].min():.3f}",
        "to",
        f"{ligand_oriented[non_pyrene, 2].max():.3f}",
        "A"
    )

    # ========================================================
    # CROSS-INTERACTION-ONLY SYSTEM
    #
    # There are NO internal ligand or graphene forces here.
    #
    # Every energy below is therefore purely:
    #
    # graphene <-> ligand
    # ========================================================

    system = openmm.System()

    for _ in range(
        n_graphene_carbon
    ):

        system.addParticle(
            10.011
            * unit.dalton
        )

    for _ in range(
        2 * n_graphene_carbon
    ):

        system.addParticle(
            1.000
            * unit.dalton
        )

    for mass in ligand_masses:

        system.addParticle(
            mass
        )

    # --------------------------------------------------------
    # Graphene nonbonded parameters
    # --------------------------------------------------------

    graphene_q = np.concatenate(
        [
            np.full(
                n_graphene_carbon,
                Q_CG1
            ),
            np.full(
                2 * n_graphene_carbon,
                Q_CGE
            )
        ]
    )

    graphene_sigma = np.concatenate(
        [
            np.full(
                n_graphene_carbon,
                SIGMA_CG1_NM
            ),
            np.full(
                2 * n_graphene_carbon,
                SIGMA_CGE_NM
            )
        ]
    )

    graphene_epsilon = np.concatenate(
        [
            np.full(
                n_graphene_carbon,
                EPS_CG1_KJ
            ),
            np.full(
                2 * n_graphene_carbon,
                EPS_CGE_KJ
            )
        ]
    )

    if abs(
        graphene_q.sum()
    ) > 1e-10:

        raise RuntimeError(
            "Graphene should be neutral."
        )

    all_q = np.concatenate(
        [
            graphene_q,
            ligand_q
        ]
    )

    all_sigma = np.concatenate(
        [
            graphene_sigma,
            ligand_sigma
        ]
    )

    all_epsilon = np.concatenate(
        [
            graphene_epsilon,
            ligand_epsilon
        ]
    )

    n_total = (
        n_graphene
        + n_ligand
    )

    graphene_group = set(
        range(
            n_graphene
        )
    )

    ligand_group = set(
        range(
            n_graphene,
            n_total
        )
    )

    pyrene_group = set(
        n_graphene + i
        for i in pyrene_local
    )

    # Force groups:
    #
    # 0 = full ligand LJ
    # 1 = full ligand electrostatics
    # 2 = pyrene contribution LJ
    # 3 = pyrene contribution electrostatics

    system.addForce(
        make_lj_force(
            all_sigma,
            all_epsilon,
            graphene_group,
            ligand_group,
            0
        )
    )

    system.addForce(
        make_coulomb_force(
            all_q,
            graphene_group,
            ligand_group,
            1
        )
    )

    system.addForce(
        make_lj_force(
            all_sigma,
            all_epsilon,
            graphene_group,
            pyrene_group,
            2
        )
    )

    system.addForce(
        make_coulomb_force(
            all_q,
            graphene_group,
            pyrene_group,
            3
        )
    )

    # --------------------------------------------------------
    # OpenMM
    # --------------------------------------------------------

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

    print(
        "OpenMM platform:",
        platform.getName()
    )

    # --------------------------------------------------------
    # Put pyrene over center of graphene
    # --------------------------------------------------------

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

    results = []

    print()

    print(
        "RIGID INTERACTION SCAN"
    )

    print(
        "Distance = pyrene-carbon plane "
        "to graphene-carbon plane."
    )

    print()

    header = (
        f"{'d(A)':>6} "
        f"{'Py-LJ':>12} "
        f"{'Py-Coul':>12} "
        f"{'Py-total':>12} "
        f"{'Full-LJ':>12} "
        f"{'Full-Coul':>12} "
        f"{'Full-total':>12}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    # ========================================================
    # SCAN
    # ========================================================

    for distance in (
        DISTANCES_A
    ):

        ligand_here = (
            ligand_oriented.copy()
        )

        ligand_here[
            :,
            0
        ] += graphene_center_xy[0]

        ligand_here[
            :,
            1
        ] += graphene_center_xy[1]

        ligand_here[
            :,
            2
        ] += (
            graphene_z
            + float(distance)
        )

        combined_nm = (
            np.vstack(
                [
                    graphene_all,
                    ligand_here
                ]
            )
            / 10.0
        )

        positions = unit.Quantity(
            [
                openmm.Vec3(
                    x,
                    y,
                    z
                )
                for x, y, z
                in combined_nm
            ],
            unit.nanometer
        )

        context.setPositions(
            positions
        )

        full_lj = (
            group_energy(
                context,
                0
            )
        )

        full_coulomb = (
            group_energy(
                context,
                1
            )
        )

        pyrene_lj = (
            group_energy(
                context,
                2
            )
        )

        pyrene_coulomb = (
            group_energy(
                context,
                3
            )
        )

        row = {

            "distance_A":
                float(
                    distance
                ),

            "pyrene_LJ_kJmol":
                pyrene_lj,

            "pyrene_Coulomb_kJmol":
                pyrene_coulomb,

            "pyrene_total_kJmol":
                pyrene_lj
                + pyrene_coulomb,

            "full_LJ_kJmol":
                full_lj,

            "full_Coulomb_kJmol":
                full_coulomb,

            "full_total_kJmol":
                full_lj
                + full_coulomb,
        }

        results.append(
            row
        )

        print(

            f"{distance:6.2f} "

            f"{pyrene_lj:12.3f} "

            f"{pyrene_coulomb:12.3f} "

            f"{row['pyrene_total_kJmol']:12.3f} "

            f"{full_lj:12.3f} "

            f"{full_coulomb:12.3f} "

            f"{row['full_total_kJmol']:12.3f}"
        )

    # ========================================================
    # SET 20 A AS APPROXIMATE FAR-DISTANCE REFERENCE
    # ========================================================

    reference = next(

        row

        for row in results

        if np.isclose(
            row["distance_A"],
            REFERENCE_A
        )
    )

    for row in results:

        row[
            "delta_pyrene_total_vs_20A_kJmol"
        ] = (

            row[
                "pyrene_total_kJmol"
            ]

            - reference[
                "pyrene_total_kJmol"
            ]
        )

        row[
            "delta_full_total_vs_20A_kJmol"
        ] = (

            row[
                "full_total_kJmol"
            ]

            - reference[
                "full_total_kJmol"
            ]
        )

    # ========================================================
    # FIND MINIMUM
    # ========================================================

    pyrene_minimum = min(

        results,

        key=lambda row:
        row[
            "delta_pyrene_total_vs_20A_kJmol"
        ]
    )

    full_minimum = min(

        results,

        key=lambda row:
        row[
            "delta_full_total_vs_20A_kJmol"
        ]
    )

    print()

    print(
        "SCAN SUMMARY"
    )

    print(
        "------------"
    )

    print(

        "Pyrene contribution minimum:",

        f"{pyrene_minimum['distance_A']:.2f} A,",

        "DeltaE =",

        f"{pyrene_minimum['delta_pyrene_total_vs_20A_kJmol']:.3f}",

        "kJ/mol"
    )

    print(

        "Full-ligand minimum:",

        f"{full_minimum['distance_A']:.2f} A,",

        "DeltaE =",

        f"{full_minimum['delta_full_total_vs_20A_kJmol']:.3f}",

        "kJ/mol"
    )

    if np.isclose(

        pyrene_minimum[
            "distance_A"
        ],

        min(
            DISTANCES_A
        )
    ):

        print()

        print(
            "WARNING: pyrene minimum occurs "
            "at the closest scanned distance."
        )

        print(
            "That would mean we may not yet "
            "have reached the repulsive wall."
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUT,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()

    print(
        "Saved:",
        OUT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a rigid, vacuum, finite-sheet "
        "force-field sanity scan."
    )

    print(
        "It is NOT an adsorption free energy "
        "and NOT the final PBS simulation."
    )

    del context
    del integrator


if __name__ == "__main__":
    main()
