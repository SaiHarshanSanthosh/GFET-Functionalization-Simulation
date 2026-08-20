from pathlib import Path
import csv
import numpy as np

import openmm as mm
from openmm import app, unit
from ase.build import graphene


PYRENE_PRMTOP = Path(
    "parameters/ff2_pyrene/pyrene_gaff2.prmtop"
)

PYRENE_INPCRD = Path(
    "parameters/ff2_pyrene/pyrene_gaff2.inpcrd"
)

OUTPUT = Path(
    "analysis/ff2_bare_pyrene_graphene_registry_scan.csv"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IFF GRAPHENE PARAMETERS
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

SIGMA_CGE_NM = 0.0
EPS_CGE_KJ = 0.0

ONE_4PI_EPS0 = 138.935456


# ============================================================
# REGISTRY / ORIENTATION SETTINGS
# ============================================================

ROTATIONS_DEG = [
    0.0,
    15.0,
    30.0,
    45.0,
    60.0,
]

FRACTIONS = np.linspace(
    0.0,
    0.8,
    5,
)

HEIGHTS_A = np.arange(
    3.15,
    3.60 + 0.025,
    0.05,
)

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
            f"Expected 1250 carbons, found {len(carbon)}."
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

    # ASE cell is 25 x 25 primitive cells.
    a1 = cell[0] / 25.0
    a2 = cell[1] / 25.0

    return (
        carbon,
        all_positions,
        a1,
        a2,
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

    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )

    if system.getNumParticles() != 26:
        raise RuntimeError(
            "Expected 26-atom C16H10 pyrene."
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

    if len(carbon_indices) != 16:
        raise RuntimeError(
            "Expected 16 pyrene carbons."
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

    # Rodrigues rotation: pyrene normal -> +z
    zhat = np.array(
        [0.0, 0.0, 1.0]
    )

    normal /= np.linalg.norm(
        normal
    )

    v = np.cross(
        normal,
        zhat,
    )

    c = float(
        np.dot(
            normal,
            zhat,
        )
    )

    if np.linalg.norm(v) < 1e-12:

        if c > 0.0:
            R = np.eye(3)

        else:
            R = np.diag(
                [1.0, -1.0, -1.0]
            )

    else:

        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

        R = (
            np.eye(3)
            + vx
            + vx @ vx
            * (
                (1.0 - c)
                / np.dot(v, v)
            )
        )

    xyz = (
        xyz
        - center
    ) @ R.T

    xyz -= xyz[
        carbon_indices
    ].mean(axis=0)

    nb = next(
        force
        for force in system.getForces()
        if isinstance(
            force,
            mm.NonbondedForce
        )
    )

    q = []
    sigma = []
    epsilon = []

    for i in range(
        system.getNumParticles()
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
            "Pyrene is not neutral."
        )

    masses = [
        system.getParticleMass(i)
        for i in range(
            system.getNumParticles()
        )
    ]

    return (
        xyz,
        q,
        sigma,
        epsilon,
        masses,
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
            n_graphene + len(pyrene_q),
        )
    )

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
            [float(q)]
        )

    coul.addInteractionGroup(
        graphene_group,
        pyrene_group,
    )

    system.addForce(coul)

    return system


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

    (
        graphene_carbon,
        graphene_all,
        a1,
        a2,
    ) = build_graphene()

    (
        pyrene_xyz,
        pyrene_q,
        pyrene_sigma,
        pyrene_epsilon,
        pyrene_masses,
    ) = load_pyrene()

    n_carbon = len(
        graphene_carbon
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
            "Expected 3776 total particles."
        )

    platform_names = [
        mm.Platform.getPlatform(i).getName()
        for i in range(
            mm.Platform.getNumPlatforms()
        )
    ]

    if "CPU" in platform_names:
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

    graphene_center = (
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

    def rotate_z(
        xyz,
        angle_deg,
    ):

        angle = np.deg2rad(
            angle_deg
        )

        c = np.cos(angle)
        s = np.sin(angle)

        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

        return (
            xyz
            @ R.T
        )

    def positions_for(
        u,
        v,
        angle_deg,
        height_A,
    ):

        p = rotate_z(
            pyrene_xyz,
            angle_deg,
        )

        shift = (
            u * a1
            + v * a2
        )

        p[:, 0] += (
            graphene_center[0]
            + shift[0]
        )

        p[:, 1] += (
            graphene_center[1]
            + shift[1]
        )

        p[:, 2] += (
            graphene_z
            + height_A
        )

        return (
            np.vstack(
                [
                    graphene_all,
                    p,
                ]
            )
            / 10.0
            * unit.nanometer
        )

    # --------------------------------------------------------
    # Reference energy
    # --------------------------------------------------------

    context.setPositions(
        positions_for(
            0.0,
            0.0,
            0.0,
            REFERENCE_DISTANCE_A,
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

    print()
    print("=" * 76)
    print("FF-2 BARE PYRENE REGISTRY / ROTATION SCAN")
    print("=" * 76)

    print(
        f"Primitive a1: "
        f"[{a1[0]:.6f}, {a1[1]:.6f}] A"
    )

    print(
        f"Primitive a2: "
        f"[{a2[0]:.6f}, {a2[1]:.6f}] A"
    )

    print(
        f"Registries:              "
        f"{len(FRACTIONS) ** 2}"
    )

    print(
        f"Rotations:               "
        f"{len(ROTATIONS_DEG)}"
    )

    print(
        f"Local heights:           "
        f"{len(HEIGHTS_A)}"
    )

    print(
        f"Total energy evaluations:"
        f" {len(FRACTIONS)**2 * len(ROTATIONS_DEG) * len(HEIGHTS_A)}"
    )

    print(
        f"Platform:                "
        f"{platform.getName()}"
    )

    print()
    print("Starting scan...")
    print()

    rows = []

    total_configs = (
        len(FRACTIONS) ** 2
        * len(ROTATIONS_DEG)
    )

    config_number = 0

    for u in FRACTIONS:

        for v in FRACTIONS:

            for angle in ROTATIONS_DEG:

                config_number += 1

                local = []

                for height_A in HEIGHTS_A:

                    context.setPositions(
                        positions_for(
                            float(u),
                            float(v),
                            float(angle),
                            float(height_A),
                        )
                    )

                    lj = (
                        group_energy(
                            context,
                            0,
                        )
                        - ref_lj
                    )

                    coul = (
                        group_energy(
                            context,
                            1,
                        )
                        - ref_coul
                    )

                    total = (
                        lj
                        + coul
                    )

                    local.append({
                        "height_A":
                            float(height_A),

                        "LJ_kJmol":
                            float(lj),

                        "Coulomb_kJmol":
                            float(coul),

                        "total_kJmol":
                            float(total),
                    })

                best = min(
                    local,
                    key=lambda x:
                        x["total_kJmol"],
                )

                row = {
                    "u_fraction":
                        float(u),

                    "v_fraction":
                        float(v),

                    "rotation_deg":
                        float(angle),

                    "best_height_A":
                        best["height_A"],

                    "best_LJ_kJmol":
                        best["LJ_kJmol"],

                    "best_Coulomb_kJmol":
                        best["Coulomb_kJmol"],

                    "best_total_kJmol":
                        best["total_kJmol"],
                }

                rows.append(
                    row
                )

                if (
                    config_number == 1
                    or config_number % 10 == 0
                    or config_number == total_configs
                ):

                    print(
                        f"{config_number:3d}/"
                        f"{total_configs}  "
                        f"u={u:.2f} "
                        f"v={v:.2f} "
                        f"rot={angle:5.1f}  "
                        f"z*={best['height_A']:.2f} A  "
                        f"E*={best['total_kJmol']:.3f}"
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

    energies = np.asarray(
        [
            row["best_total_kJmol"]
            for row in rows
        ]
    )

    heights = np.asarray(
        [
            row["best_height_A"]
            for row in rows
        ]
    )

    best_global = min(
        rows,
        key=lambda x:
            x["best_total_kJmol"],
    )

    worst_global = max(
        rows,
        key=lambda x:
            x["best_total_kJmol"],
    )

    print()
    print("=" * 76)
    print("REGISTRY / ORIENTATION SUMMARY")
    print("=" * 76)

    print(
        f"Best interaction:        "
        f"{best_global['best_total_kJmol']:.3f} kJ/mol"
    )

    print(
        f"Best registry:           "
        f"u={best_global['u_fraction']:.2f}, "
        f"v={best_global['v_fraction']:.2f}"
    )

    print(
        f"Best rotation:           "
        f"{best_global['rotation_deg']:.1f} deg"
    )

    print(
        f"Best height:             "
        f"{best_global['best_height_A']:.2f} A"
    )

    print()

    print(
        f"Weakest interaction:     "
        f"{worst_global['best_total_kJmol']:.3f} kJ/mol"
    )

    print(
        f"Registry/rotation spread:"
        f" {energies.max() - energies.min():.3f} kJ/mol"
    )

    print(
        f"Mean interaction:        "
        f"{energies.mean():.3f} kJ/mol"
    )

    print(
        f"SD interaction:          "
        f"{energies.std(ddof=1):.3f} kJ/mol"
    )

    print(
        f"Height range:            "
        f"{heights.min():.2f} - "
        f"{heights.max():.2f} A"
    )

    print(
        f"Saved:                   "
        f"{OUTPUT}"
    )

    if (
        np.isclose(
            heights.min(),
            HEIGHTS_A.min(),
        )
        or np.isclose(
            heights.max(),
            HEIGHTS_A.max(),
        )
    ):

        print()
        print(
            "WARNING: At least one local optimum "
            "lies on the height-scan boundary."
        )

    print()
    print(
        "FF2_BARE_PYRENE_REGISTRY_SCAN_COMPLETE"
    )


if __name__ == "__main__":
    main()
