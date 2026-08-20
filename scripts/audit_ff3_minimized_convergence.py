from pathlib import Path
import json
import numpy as np
from openmm import openmm, unit

SYSTEM = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb.xml"
)

POSITIONS = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_minimized_positions_nm.npy"
)

OUTPUT = Path(
    "analysis/"
    "ff3_minimized_convergence_audit.json"
)

N_GRAPHENE = 3750
N_SOLUTE = 3820
SITES_PER_WATER = 4

TOLERANCE = 10.0

system = openmm.XmlSerializer.deserialize(
    SYSTEM.read_text()
)

positions = np.load(
    POSITIONS
)

n_particles = system.getNumParticles()

water_sites = (
    n_particles
    -
    N_SOLUTE
)

if water_sites % SITES_PER_WATER != 0:
    raise RuntimeError(
        "Water-site count is not divisible by 4."
    )

n_waters = (
    water_sites
    //
    SITES_PER_WATER
)

if system.getNumConstraints() != (
    3
    *
    n_waters
):
    raise RuntimeError(
        "Expected exactly three rigid-water constraints "
        "per OPC molecule."
    )

integrator = openmm.VerletIntegrator(
    0.001
    *
    unit.picoseconds
)

integrator.setConstraintTolerance(
    1.0e-6
)

platform = openmm.Platform.getPlatformByName(
    "CUDA"
)

context = openmm.Context(
    system,
    integrator,
    platform,
)

context.setPositions(
    positions
    *
    unit.nanometer
)

context.applyConstraints(
    1.0e-8
)

context.computeVirtualSites()

state = context.getState(
    getPositions=True,
    getForces=True,
    getEnergy=True,
)

x = state.getPositions(
    asNumpy=True
).value_in_unit(
    unit.nanometer
)

forces = state.getForces(
    asNumpy=True
).value_in_unit(
    unit.kilojoule_per_mole
    /
    unit.nanometer
)

energy = state.getPotentialEnergy().value_in_unit(
    unit.kilojoule_per_mole
)

box = state.getPeriodicBoxVectors(
    asNumpy=True
).value_in_unit(
    unit.nanometer
)

inv_box = np.linalg.inv(
    box
)

masses = np.array([
    system.getParticleMass(i).value_in_unit(
        unit.dalton
    )
    for i in range(
        n_particles
    )
])

massive = (
    masses
    >
    0.0
)

projected = forces.copy()


def minimum_image(d):

    frac = (
        d
        @
        inv_box
    )

    frac -= np.round(
        frac
    )

    return (
        frac
        @
        box
    )


for w in range(
    n_waters
):

    base = (
        N_SOLUTE
        +
        4*w
    )

    ids = [
        base,
        base+1,
        base+2,
    ]

    r = x[
        ids
    ]

    f = forces[
        ids
    ].reshape(
        9
    )

    pairs = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]

    J = np.zeros(
        (
            3,
            9,
        )
    )

    for row, (
        a,
        b,
    ) in enumerate(
        pairs
    ):

        d = minimum_image(
            r[b]
            -
            r[a]
        )

        u = (
            d
            /
            np.linalg.norm(
                d
            )
        )

        J[
            row,
            3*a:3*a+3
        ] = -u

        J[
            row,
            3*b:3*b+3
        ] = +u

    coeff = np.linalg.lstsq(
        J @ J.T,
        J @ f,
        rcond=None,
    )[0]

    projected[
        ids
    ] = (
        f
        -
        J.T @ coeff
    ).reshape(
        3,
        3,
    )


raw_rms = float(
    np.sqrt(
        np.mean(
            forces[
                massive
            ]**2
        )
    )
)

projected_rms = float(
    np.sqrt(
        np.mean(
            projected[
                massive
            ]**2
        )
    )
)

projected_magnitude = np.linalg.norm(
    projected,
    axis=1,
)

projected_magnitude[
    ~massive
] = 0.0

max_index = int(
    np.argmax(
        projected_magnitude
    )
)

max_projected_force = float(
    projected_magnitude[
        max_index
    ]
)

graphene_z_max = float(
    np.max(
        x[
            :1250,
            2
        ]
    )
)

oxygen_indices = (
    N_SOLUTE
    +
    np.arange(
        n_waters
    )
    * 4
)

waters_below = int(
    np.sum(
        x[
            oxygen_indices,
            2
        ]
        <
        graphene_z_max
    )
)

passed = (
    projected_rms
    <=
    TOLERANCE
    and
    waters_below
    ==
    0
)

result = {
    "status":
        (
            "PASS"
            if passed
            else "FAIL"
        ),

    "potential_energy_kj_mol":
        float(
            energy
        ),

    "raw_rms_force_components_kj_mol_nm":
        raw_rms,

    "constraint_projected_rms_force_components_kj_mol_nm":
        projected_rms,

    "minimization_tolerance_kj_mol_nm":
        TOLERANCE,

    "max_projected_force_kj_mol_nm":
        max_projected_force,

    "max_projected_force_particle":
        max_index,

    "waters_below_graphene":
        waters_below,

    "opc_waters":
        int(
            n_waters
        ),

    "method":
        (
            "Rigid-OPC constraint-normal force components "
            "removed before RMS convergence assessment."
        ),
}

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print()
print("=" * 72)
print("FF-3 MINIMIZED CONVERGENCE AUDIT")
print("=" * 72)

for key, value in result.items():
    print(
        f"{key}: {value}"
    )

del context
del integrator
