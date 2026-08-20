from pathlib import Path
import csv

import numpy as np

import openmm as mm
from openmm import app, unit

from ase.build import graphene


# ============================================================
# FILES
# ============================================================

PYRENE_PRMTOP = Path(
    "parameters/ff2_pyrene/pyrene_gaff2.prmtop"
)

PYRENE_INPCRD = Path(
    "parameters/ff2_pyrene/pyrene_gaff2.inpcrd"
)

OUTPUT = Path(
    "analysis/ff2_bare_pyrene_graphene_height_scan.csv"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IFF GRAPHENE PARAMETERS
#
# Same values used in the validated graphene workflow.
# ============================================================

Q_CG1 = +0.2
Q_CGE = -0.1

RMIN_CG1_A = 3.79
EPS_CG1_KCAL = 0.063

SIGMA_CG1_NM = (
    RMIN_CG1_A
    / (2.0 ** (1.0 / 6.0))
    / 10.0
)

EPS_CG1_KJ = (
    EPS_CG1_KCAL
    * 4.184
)

# Virtual pi particles carry electrostatics only.
SIGMA_CGE_NM = 0.0
EPS_CGE_KJ = 0.0

ONE_4PI_EPS0 = 138.935456


# ============================================================
# SCAN
# ============================================================

DISTANCE_MIN_A = 2.50
DISTANCE_MAX_A = 8.00
DISTANCE_STEP_A = 0.05

REFERENCE_DISTANCE_A = 50.0


# ============================================================
# GRAPHENE
# ============================================================

def build_graphene():

    sheet = graphene(
        formula="C2",
        a=2.46,
        size=(25, 25, 1),
        vacuum=20.0,
    )

    sheet.pbc = (
        True,
        True,
        False,
    )

    carbon = np.asarray(
        sheet.get_positions(),
        dtype=float,
    )

    cell = np.asarray(
        sheet.cell.array,
        dtype=float,
    )

    if len(carbon) != 1250:
        raise RuntimeError(
            f"Expected 1250 graphene carbons, found {len(carbon)}."
        )

    normal = np.cross(
        cell[0],
        cell[1],
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
            pi_down,
        ]
    )

    if len(all_positions) != 3750:
        raise RuntimeError(
            "Expected 3750 IFF graphene particles."
        )

    return (
        carbon,
        all_positions,
    )


# ============================================================
# ROTATION
# ============================================================

def rotation_matrix(a, b):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))

    if np.isclose(c, 1.0):
        return np.eye(3)

    if np.isclose(c, -1.0):

        axis = np.array(
            [1.0, 0.0, 0.0]
        )

        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array(
                [0.0, 1.0, 0.0]
            )

        axis -= (
            np.dot(axis, a)
            * a
        )

        axis /= np.linalg.norm(axis)

        x, y, z = axis

        return np.array([
            [
                2*x*x - 1,
                2*x*y,
                2*x*z,
            ],
            [
                2*x*y,
                2*y*y - 1,
                2*y*z,
            ],
            [
                2*x*z,
                2*y*z,
                2*z*z - 1,
            ],
        ])

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    return (
        np.eye(3)
        + vx
        + vx @ vx
        * (
            (1.0 - c)
            / np.dot(v, v)
        )
    )


# ============================================================
# PYRENE
# ============================================================

def load_pyrene():

    prmtop = app.AmberPrmtopFile(
        str(PYRENE_PRMTOP)
    )

    inpcrd = app.AmberInpcrdFile(
        str(PYRENE_INPCRD)
    )

    pyrene_system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )

    if pyrene_system.getNumParticles() != 26:
        raise RuntimeError(
            "Expected standalone C16H10 pyrene to contain 26 atoms."
        )

    atoms = list(
        prmtop.topology.atoms()
    )

    carbon_indices = [
        i
        for i, atom in enumerate(atoms)
        if atom.element is not None
        and atom.element.symbol == "C"
    ]

    hydrogen_indices = [
        i
        for i, atom in enumerate(atoms)
        if atom.element is not None
        and atom.element.symbol == "H"
    ]

    if len(carbon_indices) != 16:
        raise RuntimeError(
            "Expected 16 carbon atoms."
        )

    if len(hydrogen_indices) != 10:
        raise RuntimeError(
            "Expected 10 hydrogen atoms."
        )

    xyz = np.array(
        [
            [
                p.x,
                p.y,
                p.z,
            ]
            for p
            in inpcrd.positions.value_in_unit(
                unit.angstrom
            )
        ],
        dtype=float,
    )

    # Fit plane through the 16 pyrene carbons.
    carbon_xyz = xyz[
        carbon_indices
    ]

    center = carbon_xyz.mean(
        axis=0
    )

    centered = (
        carbon_xyz
        - center
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]

    R = rotation_matrix(
        normal,
        np.array(
            [0.0, 0.0, 1.0]
        ),
    )

    xyz = (
        xyz
        - center
    ) @ R.T

    # Re-center exactly on carbon-plane centroid.
    carbon_center = xyz[
        carbon_indices
    ].mean(axis=0)

    xyz -= carbon_center

    carbon_z = xyz[
        carbon_indices,
        2
    ]

    plane_rms_A = float(
        np.sqrt(
            np.mean(
                carbon_z ** 2
            )
        )
    )

    plane_max_A = float(
        np.max(
            np.abs(
                carbon_z
            )
        )
    )

    # Extract exact PRMTOP nonbonded parameters.
    nb = next(
        force
        for force in pyrene_system.getForces()
        if isinstance(
            force,
            mm.NonbondedForce
        )
    )

    q = []
    sigma = []
    epsilon = []

    for i in range(
        pyrene_system.getNumParticles()
    ):

        qi, si, ei = (
            nb.getParticleParameters(i)
        )

        q.append(
            qi.value_in_unit(
                unit.elementary_charge
            )
        )

        sigma.append(
            si.value_in_unit(
                unit.nanometer
            )
        )

        epsilon.append(
            ei.value_in_unit(
                unit.kilojoule_per_mole
            )
        )

    q = np.asarray(q)
    sigma = np.asarray(sigma)
    epsilon = np.asarray(epsilon)

    if abs(q.sum()) > 1e-10:
        raise RuntimeError(
            f"Pyrene is not neutral: {q.sum():+.12f} e"
        )

    masses = [
        pyrene_system.getParticleMass(i)
        for i in range(
            pyrene_system.getNumParticles()
        )
    ]

    return (
        xyz,
        q,
        sigma,
        epsilon,
        masses,
        plane_rms_A,
        plane_max_A,
    )


# ============================================================
# CROSS-INTERACTION SYSTEM
# ============================================================

def build_cross_system(
    graphene_q,
    graphene_sigma,
    graphene_epsilon,
    pyrene_q,
    pyrene_sigma,
    pyrene_epsilon,
    pyrene_masses,
):

    system = mm.System()

    n_graphene = len(
        graphene_q
    )

    n_pyrene = len(
        pyrene_q
    )

    # Graphene masses are irrelevant for a static energy scan,
    # but use physical carbon masses for cores and zero-mass pi sites.
    for i in range(
        n_graphene
    ):

        if i < 1250:
            system.addParticle(
                12.011 * unit.dalton
            )
        else:
            system.addParticle(
                0.0 * unit.dalton
            )

    for mass in pyrene_masses:
        system.addParticle(
            mass
        )

    all_q = np.concatenate(
        [
            graphene_q,
            pyrene_q,
        ]
    )

    all_sigma = np.concatenate(
        [
            graphene_sigma,
            pyrene_sigma,
        ]
    )

    all_epsilon = np.concatenate(
        [
            graphene_epsilon,
            pyrene_epsilon,
        ]
    )

    graphene_group = set(
        range(
            n_graphene
        )
    )

    pyrene_group = set(
        range(
            n_graphene,
            n_graphene + n_pyrene,
        )
    )

    # Lennard-Jones cross energy.
    lj = mm.CustomNonbondedForce(
        "4*eps*((sig/r)^12-(sig/r)^6);"
        "sig=0.5*(sigma1+sigma2);"
        "eps=sqrt(epsilon1*epsilon2)"
    )

    lj.addPerParticleParameter(
        "sigma"
    )

    lj.addPerParticleParameter(
        "epsilon"
    )

    lj.setNonbondedMethod(
        mm.CustomNonbondedForce.NoCutoff
    )

    lj.setForceGroup(0)

    for s, e in zip(
        all_sigma,
        all_epsilon,
    ):

        lj.addParticle(
            [
                float(s),
                float(e),
            ]
        )

    lj.addInteractionGroup(
        graphene_group,
        pyrene_group,
    )

    system.addForce(lj)

    # Electrostatic cross energy.
    coul = mm.CustomNonbondedForce(
        "ONE_4PI_EPS0*q1*q2/r"
    )

    coul.addGlobalParameter(
        "ONE_4PI_EPS0",
        ONE_4PI_EPS0,
    )

    coul.addPerParticleParameter(
        "q"
    )

    coul.setNonbondedMethod(
        mm.CustomNonbondedForce.NoCutoff
    )

    coul.setForceGroup(1)

    for q in all_q:

        coul.addParticle(
            [
                float(q)
            ]
        )

    coul.addInteractionGroup(
        graphene_group,
        pyrene_group,
    )

    system.addForce(coul)

    return system


# ============================================================
# FORCE-GROUP ENERGY
# ============================================================

def group_energy(
    context,
    group,
):

    return (
        context.getState(
            getEnergy=True,
            groups=1 << group,
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
        PYRENE_PRMTOP,
        PYRENE_INPCRD,
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    (
        graphene_carbon,
        graphene_all,
    ) = build_graphene()

    (
        pyrene_xyz,
        pyrene_q,
        pyrene_sigma,
        pyrene_epsilon,
        pyrene_masses,
        plane_rms_A,
        plane_max_A,
    ) = load_pyrene()

    n_carbon = len(
        graphene_carbon
    )

    n_graphene = len(
        graphene_all
    )

    graphene_q = np.concatenate(
        [
            np.full(
                n_carbon,
                Q_CG1,
            ),
            np.full(
                2 * n_carbon,
                Q_CGE,
            ),
        ]
    )

    graphene_sigma = np.concatenate(
        [
            np.full(
                n_carbon,
                SIGMA_CG1_NM,
            ),
            np.full(
                2 * n_carbon,
                SIGMA_CGE_NM,
            ),
        ]
    )

    graphene_epsilon = np.concatenate(
        [
            np.full(
                n_carbon,
                EPS_CG1_KJ,
            ),
            np.full(
                2 * n_carbon,
                EPS_CGE_KJ,
            ),
        ]
    )

    print()
    print("=" * 76)
    print("FF-2 BARE PYRENE / IFF GRAPHENE RIGID HEIGHT SCAN")
    print("=" * 76)

    print(
        f"Graphene carbons:        {n_carbon}"
    )

    print(
        f"Graphene particles:      {n_graphene}"
    )

    print(
        f"Graphene total charge:   {graphene_q.sum():+.12f} e"
    )

    print(
        f"Pyrene particles:        {len(pyrene_q)}"
    )

    print(
        f"Pyrene total charge:     {pyrene_q.sum():+.12f} e"
    )

    print(
        f"Pyrene plane RMS:        {plane_rms_A:.6f} A"
    )

    print(
        f"Pyrene plane MAX:        {plane_max_A:.6f} A"
    )

    system = build_cross_system(
        graphene_q,
        graphene_sigma,
        graphene_epsilon,
        pyrene_q,
        pyrene_sigma,
        pyrene_epsilon,
        pyrene_masses,
    )

    if system.getNumParticles() != 3776:
        raise RuntimeError(
            f"Expected 3776 total particles, found "
            f"{system.getNumParticles()}."
        )

    platforms = [
        mm.Platform.getPlatform(i).getName()
        for i in range(
            mm.Platform.getNumPlatforms()
        )
    ]

    if "CPU" in platforms:
        platform = (
            mm.Platform.getPlatformByName(
                "CPU"
            )
        )
    else:
        platform = (
            mm.Platform.getPlatformByName(
                "Reference"
            )
        )

    integrator = mm.VerletIntegrator(
        1.0 * unit.femtosecond
    )

    context = mm.Context(
        system,
        integrator,
        platform,
    )

    graphene_center_xy = (
        graphene_carbon[
            :,
            :2
        ].mean(axis=0)
    )

    graphene_z = float(
        graphene_carbon[
            :,
            2
        ].mean()
    )

    def positions_at_distance(
        distance_A,
    ):

        p = np.array(
            pyrene_xyz,
            copy=True,
        )

        p[:, 0] += (
            graphene_center_xy[0]
        )

        p[:, 1] += (
            graphene_center_xy[1]
        )

        p[:, 2] += (
            graphene_z
            + float(distance_A)
        )

        combined_nm = (
            np.vstack(
                [
                    graphene_all,
                    p,
                ]
            )
            / 10.0
        )

        return (
            combined_nm
            * unit.nanometer
        )

    # --------------------------------------------------------
    # Far-distance zero
    # --------------------------------------------------------

    context.setPositions(
        positions_at_distance(
            REFERENCE_DISTANCE_A
        )
    )

    ref_lj = group_energy(
        context,
        0,
    )

    ref_coul = group_energy(
        context,
        1,
    )

    print(
        f"Platform:                {platform.getName()}"
    )

    print(
        f"Far reference:           {REFERENCE_DISTANCE_A:.1f} A"
    )

    print(
        f"Reference LJ:            {ref_lj:.9f} kJ/mol"
    )

    print(
        f"Reference Coulomb:       {ref_coul:.9f} kJ/mol"
    )

    distances = np.arange(
        DISTANCE_MIN_A,
        DISTANCE_MAX_A
        + 0.5 * DISTANCE_STEP_A,
        DISTANCE_STEP_A,
    )

    rows = []

    print()
    print("Starting rigid scan...")
    print()

    for number, distance_A in enumerate(
        distances,
        start=1,
    ):

        context.setPositions(
            positions_at_distance(
                distance_A
            )
        )

        raw_lj = group_energy(
            context,
            0,
        )

        raw_coul = group_energy(
            context,
            1,
        )

        delta_lj = (
            raw_lj
            - ref_lj
        )

        delta_coul = (
            raw_coul
            - ref_coul
        )

        delta_total = (
            delta_lj
            + delta_coul
        )

        rows.append({
            "distance_A":
                float(distance_A),

            "delta_LJ_kJmol":
                float(delta_lj),

            "delta_Coulomb_kJmol":
                float(delta_coul),

            "delta_total_kJmol":
                float(delta_total),
        })

        if (
            number == 1
            or number % 10 == 0
            or number == len(distances)
        ):

            print(
                f"{number:3d}/{len(distances)}  "
                f"d={distance_A:5.2f} A  "
                f"LJ={delta_lj:10.3f}  "
                f"Coul={delta_coul:10.3f}  "
                f"Total={delta_total:10.3f}"
            )

    minimum = min(
        rows,
        key=lambda row:
            row[
                "delta_total_kJmol"
            ],
    )

    with open(
        OUTPUT,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print("=" * 76)
    print("FF-2 HEIGHT-SCAN RESULT")
    print("=" * 76)

    print(
        f"Minimum distance:        "
        f"{minimum['distance_A']:.3f} A"
    )

    print(
        f"LJ at minimum:           "
        f"{minimum['delta_LJ_kJmol']:.3f} kJ/mol"
    )

    print(
        f"Coulomb at minimum:      "
        f"{minimum['delta_Coulomb_kJmol']:.3f} kJ/mol"
    )

    print(
        f"Total interaction:       "
        f"{minimum['delta_total_kJmol']:.3f} kJ/mol"
    )

    print(
        f"Saved:                   {OUTPUT}"
    )

    if np.isclose(
        minimum["distance_A"],
        DISTANCE_MIN_A,
    ):

        raise RuntimeError(
            "Minimum occurs at lower scan boundary; "
            "repulsive wall was not resolved."
        )

    if np.isclose(
        minimum["distance_A"],
        DISTANCE_MAX_A,
    ):

        raise RuntimeError(
            "Minimum occurs at upper scan boundary."
        )

    print()
    print(
        "IMPORTANT: rigid vacuum finite-sheet unit test only."
    )

    print(
        "This is NOT an adsorption free energy."
    )

    print()
    print(
        "FF2_BARE_PYRENE_HEIGHT_SCAN_COMPLETE"
    )


if __name__ == "__main__":
    main()
