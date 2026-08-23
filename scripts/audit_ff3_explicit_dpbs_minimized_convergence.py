from pathlib import Path
import json
import numpy as np
from openmm import openmm, unit

SYSTEM = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb.xml"
)

POSITIONS = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_minimization_continue1_positions_nm.npy"
)

OUTPUT = Path(
    "analysis/"
    "ff3_explicit_dpbs_minimization_convergence_audit.json"
)

ASSEMBLY_METADATA = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_assembly_check.json"
)

with open(ASSEMBLY_METADATA, "r") as f:
    assembly_metadata = json.load(f)

N_CARBON = 1250
N_GRAPHENE = 3750
N_SOLUTE = 3820
SITES_PER_WATER = 4

TOLERANCE = 10.0

N_RETAINED_WATERS = int(
    assembly_metadata["opc_waters"]
)

N_RETAINED_WATER_SITES = (
    N_RETAINED_WATERS
    * SITES_PER_WATER
)

N_NA = int(assembly_metadata["na_ions"])
N_K = int(assembly_metadata["k_ions"])
N_CL = int(assembly_metadata["cl_ions"])
N_H2PO4 = int(assembly_metadata["h2po4_ions"])
N_HPO4 = int(assembly_metadata["hpo4_ions"])

WATER_START_INDEX = N_SOLUTE
WATER_END_INDEX = (
    WATER_START_INDEX
    + N_RETAINED_WATER_SITES
)

NA_START_INDEX = int(
    assembly_metadata["na_particle_start_index"]
)
NA_END_INDEX = int(
    assembly_metadata["na_particle_end_index_exclusive"]
)

K_START_INDEX = int(
    assembly_metadata["k_particle_start_index"]
)
K_END_INDEX = int(
    assembly_metadata["k_particle_end_index_exclusive"]
)

CL_START_INDEX = int(
    assembly_metadata["cl_particle_start_index"]
)
CL_END_INDEX = int(
    assembly_metadata["cl_particle_end_index_exclusive"]
)

H2PO4_START_INDEX = int(
    assembly_metadata["h2po4_particle_start_index"]
)
H2PO4_END_INDEX = int(
    assembly_metadata["h2po4_particle_end_index_exclusive"]
)

HPO4_START_INDEX = int(
    assembly_metadata["hpo4_particle_start_index"]
)
HPO4_END_INDEX = int(
    assembly_metadata["hpo4_particle_end_index_exclusive"]
)

N_TOTAL_EXPECTED = int(
    assembly_metadata["total_particles"]
)

if WATER_END_INDEX != NA_START_INDEX:
    raise RuntimeError("Water/Na particle boundary mismatch.")

if NA_END_INDEX != K_START_INDEX:
    raise RuntimeError("Na/K particle boundary mismatch.")

if K_END_INDEX != CL_START_INDEX:
    raise RuntimeError("K/Cl particle boundary mismatch.")

if CL_END_INDEX != H2PO4_START_INDEX:
    raise RuntimeError("Cl/H2PO4 particle boundary mismatch.")

if H2PO4_END_INDEX != HPO4_START_INDEX:
    raise RuntimeError("H2PO4/HPO4 particle boundary mismatch.")

if HPO4_END_INDEX != N_TOTAL_EXPECTED:
    raise RuntimeError("Final DPBS particle boundary mismatch.")

system = openmm.XmlSerializer.deserialize(
    SYSTEM.read_text()
)

positions = np.load(
    POSITIONS
)

n_particles = system.getNumParticles()

if n_particles != N_TOTAL_EXPECTED:
    raise RuntimeError(
        f"Particle count mismatch: {n_particles} vs {N_TOTAL_EXPECTED}"
    )

if positions.shape != (n_particles, 3):
    raise RuntimeError(
        f"Position shape mismatch: {positions.shape}"
    )

n_waters = N_RETAINED_WATERS

if system.getNumConstraints() != 3 * n_waters:
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
        x[:N_CARBON, 2]
    )
)

oxygen_indices = (
    WATER_START_INDEX
    + np.arange(n_waters) * 4
)

na_indices = np.arange(
    NA_START_INDEX,
    NA_END_INDEX,
)

k_indices = np.arange(
    K_START_INDEX,
    K_END_INDEX,
)

cl_indices = np.arange(
    CL_START_INDEX,
    CL_END_INDEX,
)

h2po4_indices = np.arange(
    H2PO4_START_INDEX,
    H2PO4_END_INDEX,
)

hpo4_indices = np.arange(
    HPO4_START_INDEX,
    HPO4_END_INDEX,
)

waters_below = int(
    np.sum(x[oxygen_indices, 2] < graphene_z_max)
)

na_below = int(
    np.sum(x[na_indices, 2] < graphene_z_max)
)

k_below = int(
    np.sum(x[k_indices, 2] < graphene_z_max)
)

cl_below = int(
    np.sum(x[cl_indices, 2] < graphene_z_max)
)

h2po4_below = int(
    np.sum(x[h2po4_indices, 2] < graphene_z_max)
)

hpo4_below = int(
    np.sum(x[hpo4_indices, 2] < graphene_z_max)
)

one_sided_pass = (
    waters_below == 0
    and na_below == 0
    and k_below == 0
    and cl_below == 0
    and h2po4_below == 0
    and hpo4_below == 0
)

passed = (
    projected_rms <= TOLERANCE
    and one_sided_pass
)

result = {
    "status": "PASS" if passed else "FAIL",

    "potential_energy_kj_mol":
        float(energy),

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

    "na_below_graphene":
        na_below,

    "k_below_graphene":
        k_below,

    "cl_below_graphene":
        cl_below,

    "h2po4_atoms_below_graphene":
        h2po4_below,

    "hpo4_atoms_below_graphene":
        hpo4_below,

    "opc_waters":
        int(n_waters),

    "na_ions":
        N_NA,

    "k_ions":
        N_K,

    "cl_ions":
        N_CL,

    "h2po4_ions":
        N_H2PO4,

    "hpo4_ions":
        N_HPO4,

    "method":
        (
            "Rigid-OPC constraint-normal force components "
            "removed before RMS convergence assessment; "
            "all explicit-DPBS species checked for "
            "one-sided geometry."
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
