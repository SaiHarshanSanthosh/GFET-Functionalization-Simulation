from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from openmm import (
    CustomNonbondedForce,
    LangevinMiddleIntegrator,
    Platform,
    System,
    Vec3,
    XmlSerializer,
    unit,
)
import openmm as mm


# ============================================================
# FILES
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_vacuum_assembly_nocutoff.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "equilibrated_300K_500ps_positions_nm.npy"
)

OUTPUT_DIR = Path(
    "analysis/interaction_energy_scan"
)

FIGURE_DIR = Path(
    "analysis/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CSV_OUT = (
    OUTPUT_DIR
    / "pyrene_graphene_interaction_energy_scan.csv"
)

FIGURE_OUT = (
    FIGURE_DIR
    / "07_actual_interaction_energy_scan.png"
)


# ============================================================
# SYSTEM INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250

# Graphene:
# 0-1249       carbon cores
# 1250-2499    upper pi particles
# 2500-3749    lower pi particles

GRAPHENE_START = 0
GRAPHENE_END = 3750

# Ligand:
# 3750-3819

LIGAND_START = 3750
LIGAND_END = 3820

# Exact aromatic pyrene atoms previously identified from MOL2:
# local ligand indices 2-17

PYRENE_LOCAL_INDICES = np.arange(
    2,
    18,
    dtype=int,
)

PYRENE_GLOBAL_INDICES = (
    LIGAND_START
    +
    PYRENE_LOCAL_INDICES
)

GRAPHENE_INDICES = np.arange(
    GRAPHENE_START,
    GRAPHENE_END,
    dtype=int,
)

LIGAND_INDICES = np.arange(
    LIGAND_START,
    LIGAND_END,
    dtype=int,
)


# ============================================================
# SCAN PARAMETERS
# ============================================================

DISTANCE_MIN_A = 2.50
DISTANCE_MAX_A = 8.00
DISTANCE_STEP_A = 0.05

# Separate far-away reference used to define zero interaction.
REFERENCE_DISTANCE_A = 20.0


# ============================================================
# LOAD FORCE FIELD SYSTEM
# ============================================================

if not SYSTEM_XML.exists():
    raise FileNotFoundError(
        f"System XML not found:\n{SYSTEM_XML}"
    )

if not POSITIONS_FILE.exists():
    raise FileNotFoundError(
        f"500 ps positions not found:\n{POSITIONS_FILE}"
    )


system_full = XmlSerializer.deserialize(
    SYSTEM_XML.read_text()
)

print()
print("=" * 72)
print("REAL PYRENE-GRAPHENE INTERACTION ENERGY SCAN")
print("=" * 72)

print(
    f"Vacuum system particles: "
    f"{system_full.getNumParticles()}"
)


if system_full.getNumParticles() != 3820:
    raise RuntimeError(
        "Expected 3820 particles in vacuum assembly."
    )


# ============================================================
# LOAD 500 ps COORDINATES
# ============================================================

all_positions_nm = np.load(
    POSITIONS_FILE
)

print(
    f"500 ps state particles: "
    f"{len(all_positions_nm)}"
)


if len(all_positions_nm) < 3820:
    raise RuntimeError(
        "500 ps position array contains fewer than "
        "3820 particles."
    )


# Only take graphene + ligand.
positions_nm = np.array(
    all_positions_nm[:3820],
    dtype=float,
    copy=True,
)


# ============================================================
# FIND ORIGINAL NONBONDED FORCE
# ============================================================

original_nb = None

for force in system_full.getForces():

    if isinstance(
        force,
        mm.NonbondedForce,
    ):
        original_nb = force
        break


if original_nb is None:
    raise RuntimeError(
        "No OpenMM NonbondedForce found."
    )


print(
    f"Nonbonded particles: "
    f"{original_nb.getNumParticles()}"
)

print(
    f"Nonbonded exceptions: "
    f"{original_nb.getNumExceptions()}"
)


# ============================================================
# VALIDATE THAT NO GRAPHENE-LIGAND CROSS EXCEPTIONS EXIST
#
# Intramolecular 1-2/1-3/1-4 exceptions are expected within
# graphene or within ligand, but there should be no exclusions
# between the separate graphene and ligand components.
# ============================================================

graphene_set = set(
    GRAPHENE_INDICES.tolist()
)

ligand_set = set(
    LIGAND_INDICES.tolist()
)

cross_exceptions = []


for exception_index in range(
    original_nb.getNumExceptions()
):

    (
        atom1,
        atom2,
        charge_prod,
        sigma,
        epsilon,
    ) = original_nb.getExceptionParameters(
        exception_index
    )

    i = int(atom1)
    j = int(atom2)

    is_cross = (
        (
            i in graphene_set
            and
            j in ligand_set
        )
        or
        (
            j in graphene_set
            and
            i in ligand_set
        )
    )

    if is_cross:
        cross_exceptions.append(
            (
                exception_index,
                i,
                j,
            )
        )


if cross_exceptions:
    raise RuntimeError(
        "Unexpected graphene-ligand nonbonded "
        f"exceptions found: {cross_exceptions[:10]}"
    )


print(
    "PASS: no graphene-ligand cross exceptions"
)


# ============================================================
# BUILD EXACT CROSS-INTERACTION FORCE
#
# OpenMM NonbondedForce uses:
#
# Coulomb:
# k_e * q1*q2 / r
#
# Lennard-Jones:
# sigma = (sigma1 + sigma2)/2
# epsilon = sqrt(epsilon1*epsilon2)
#
# E = 4 epsilon [
#       (sigma/r)^12
#       -
#       (sigma/r)^6
#     ]
#
# This reproduces the graphene-ligand cross interaction using
# the particle parameters from the real combined FF XML.
# ============================================================

COULOMB_CONSTANT = 138.935456


def build_cross_system(
    group_a,
    group_b,
):
    """
    Build a minimal OpenMM System containing only
    nonbonded interactions between group_a and group_b.
    """

    system = System()

    for particle_index in range(
        original_nb.getNumParticles()
    ):

        mass = system_full.getParticleMass(
            particle_index
        )

        system.addParticle(
            mass
        )

    expression = (
        "coulomb + lj;"
        "coulomb = ke*q1*q2/r;"
        "lj = 4*epsilon*"
        "(sr12-sr6);"
        "sr12 = sr6*sr6;"
        "sr6 = (sigma/r)^6;"
        "sigma = 0.5*(sigma1+sigma2);"
        "epsilon = sqrt(epsilon1*epsilon2);"
        f"ke = {COULOMB_CONSTANT}"
    )

    force = CustomNonbondedForce(
        expression
    )

    force.addPerParticleParameter(
        "q"
    )

    force.addPerParticleParameter(
        "sigma"
    )

    force.addPerParticleParameter(
        "epsilon"
    )

    for particle_index in range(
        original_nb.getNumParticles()
    ):

        (
            charge,
            sigma,
            epsilon,
        ) = original_nb.getParticleParameters(
            particle_index
        )

        force.addParticle(
            [
                charge.value_in_unit(
                    unit.elementary_charge
                ),
                sigma.value_in_unit(
                    unit.nanometer
                ),
                epsilon.value_in_unit(
                    unit.kilojoule_per_mole
                ),
            ]
        )

    force.setNonbondedMethod(
        CustomNonbondedForce.NoCutoff
    )

    force.addInteractionGroup(
        set(
            int(i)
            for i in group_a
        ),
        set(
            int(i)
            for i in group_b
        ),
    )

    system.addForce(
        force
    )

    return system


# ============================================================
# TWO REAL INTERACTION SYSTEMS
#
# 1. Entire Pyrene-PEG5-Propargyl vs graphene
# 2. Aromatic pyrene atoms vs graphene
#
# The second curve is NOT a separately parameterized molecule.
# It is the pyrene aromatic contribution within the actual
# Pyrene-PEG5-Propargyl force-field model.
# ============================================================

full_interaction_system = (
    build_cross_system(
        GRAPHENE_INDICES,
        LIGAND_INDICES,
    )
)

pyrene_interaction_system = (
    build_cross_system(
        GRAPHENE_INDICES,
        PYRENE_GLOBAL_INDICES,
    )
)


# ============================================================
# PLATFORM
# ============================================================

available_platforms = [
    Platform.getPlatform(i).getName()
    for i in range(
        Platform.getNumPlatforms()
    )
]

print(
    "Available platforms:",
    available_platforms,
)


if "CUDA" in available_platforms:

    platform = Platform.getPlatformByName(
        "CUDA"
    )

    platform_properties = {
        "Precision": "mixed"
    }

elif "OpenCL" in available_platforms:

    platform = Platform.getPlatformByName(
        "OpenCL"
    )

    platform_properties = {}

else:

    platform = Platform.getPlatformByName(
        "CPU"
    )

    platform_properties = {}


print(
    f"Using platform: "
    f"{platform.getName()}"
)


# ============================================================
# CREATE CONTEXTS
# ============================================================

integrator_full = LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1.0 / unit.picosecond,
    1.0 * unit.femtosecond,
)

integrator_pyrene = LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1.0 / unit.picosecond,
    1.0 * unit.femtosecond,
)


context_full = mm.Context(
    full_interaction_system,
    integrator_full,
    platform,
    platform_properties,
)

context_pyrene = mm.Context(
    pyrene_interaction_system,
    integrator_pyrene,
    platform,
    platform_properties,
)


# ============================================================
# INITIAL HEIGHT
# ============================================================

graphene_mean_z_nm = float(
    positions_nm[
        :N_GRAPHENE_CARBONS,
        2
    ].mean()
)

pyrene_mean_z_nm = float(
    positions_nm[
        PYRENE_GLOBAL_INDICES,
        2
    ].mean()
)

initial_distance_nm = (
    pyrene_mean_z_nm
    -
    graphene_mean_z_nm
)

initial_distance_A = (
    initial_distance_nm
    * 10.0
)


print()
print(
    f"500 ps pyrene-plane distance: "
    f"{initial_distance_A:.4f} A"
)


# ============================================================
# RIGID VERTICAL TRANSLATION
# ============================================================

def positions_at_distance(
    target_distance_A,
):
    """
    Rigidly translate the entire ligand in z so that:

        mean(pyrene aromatic z)
        -
        mean(graphene carbon z)

    equals target_distance_A.

    Internal ligand geometry is NOT changed.
    """

    target_nm = (
        target_distance_A
        * 0.1
    )

    dz_nm = (
        target_nm
        -
        initial_distance_nm
    )

    scan_positions = np.array(
        positions_nm,
        copy=True,
    )

    scan_positions[
        LIGAND_START:LIGAND_END,
        2
    ] += dz_nm

    return scan_positions


# ============================================================
# ENERGY FUNCTION
# ============================================================

def get_energy_kjmol(
    context,
    xyz_nm,
):

    context.setPositions(
        xyz_nm
        * unit.nanometer
    )

    state = context.getState(
        getEnergy=True
    )

    return state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


# ============================================================
# FAR-DISTANCE ZERO REFERENCE
# ============================================================

reference_positions = positions_at_distance(
    REFERENCE_DISTANCE_A
)

reference_full = get_energy_kjmol(
    context_full,
    reference_positions,
)

reference_pyrene = get_energy_kjmol(
    context_pyrene,
    reference_positions,
)


print()
print(
    f"Zero reference distance: "
    f"{REFERENCE_DISTANCE_A:.1f} A"
)

print(
    f"Full ligand interaction at reference: "
    f"{reference_full:.6f} kJ/mol"
)

print(
    f"Pyrene contribution at reference: "
    f"{reference_pyrene:.6f} kJ/mol"
)


# ============================================================
# RUN SCAN
# ============================================================

distances_A = np.arange(
    DISTANCE_MIN_A,
    DISTANCE_MAX_A
    +
    0.5 * DISTANCE_STEP_A,
    DISTANCE_STEP_A,
)


full_energy = []
pyrene_energy = []


print()
print("=" * 72)
print("SCANNING")
print("=" * 72)


for scan_index, distance_A in enumerate(
    distances_A
):

    scan_positions = positions_at_distance(
        distance_A
    )

    e_full = get_energy_kjmol(
        context_full,
        scan_positions,
    )

    e_pyrene = get_energy_kjmol(
        context_pyrene,
        scan_positions,
    )

    # Shift so interaction approaches zero at far separation.
    e_full -= reference_full
    e_pyrene -= reference_pyrene

    full_energy.append(
        e_full
    )

    pyrene_energy.append(
        e_pyrene
    )

    print(
        f"{distance_A:6.2f} A   "
        f"Full: {e_full:12.4f} kJ/mol   "
        f"Pyrene: {e_pyrene:12.4f} kJ/mol"
    )


full_energy = np.asarray(
    full_energy
)

pyrene_energy = np.asarray(
    pyrene_energy
)


# ============================================================
# FIND MINIMA FROM ACTUAL CALCULATED CURVES
# ============================================================

full_min_index = int(
    np.argmin(
        full_energy
    )
)

pyrene_min_index = int(
    np.argmin(
        pyrene_energy
    )
)


full_min_distance_A = float(
    distances_A[
        full_min_index
    ]
)

full_min_energy = float(
    full_energy[
        full_min_index
    ]
)


pyrene_min_distance_A = float(
    distances_A[
        pyrene_min_index
    ]
)

pyrene_min_energy = float(
    pyrene_energy[
        pyrene_min_index
    ]
)


# ============================================================
# SAVE REAL NUMERICAL DATA
# ============================================================

output_array = np.column_stack(
    [
        distances_A,
        full_energy,
        pyrene_energy,
    ]
)


header = (
    "distance_A,"
    "full_ligand_graphene_interaction_kJ_mol,"
    "pyrene_aromatic_graphene_interaction_kJ_mol"
)


np.savetxt(
    CSV_OUT,
    output_array,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.8f",
)


# ============================================================
# PLOT
# ============================================================

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


fig, ax = plt.subplots(
    figsize=(10.5, 6.5)
)


ax.plot(
    distances_A,
    pyrene_energy,
    linewidth=2.2,
    label="Pyrene aromatic contribution",
)


ax.plot(
    distances_A,
    full_energy,
    linewidth=2.2,
    label="Pyrene-PEG5-Propargyl",
)


# Zero energy
ax.axhline(
    0.0,
    linewidth=1.1,
)


# Actual calculated minimum
ax.axvline(
    full_min_distance_A,
    linestyle="--",
    linewidth=1.7,
    alpha=0.8,
)


ax.scatter(
    [full_min_distance_A],
    [full_min_energy],
    s=80,
    zorder=5,
)


ax.annotate(
    (
        "Calculated minimum\n"
        f"d = {full_min_distance_A:.2f} A\n"
        f"E = {full_min_energy:.1f} kJ/mol"
    ),
    xy=(
        full_min_distance_A,
        full_min_energy,
    ),
    xytext=(
        full_min_distance_A + 0.65,
        full_min_energy + 30,
    ),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.2,
    },
)


# MD-observed mean distance
MD_MEAN_DISTANCE_A = 3.3322


ax.axvline(
    MD_MEAN_DISTANCE_A,
    linestyle=":",
    linewidth=1.8,
    alpha=0.8,
    label=(
        "300 K MD mean = "
        f"{MD_MEAN_DISTANCE_A:.2f} A"
    ),
)


ax.set_xlabel(
    "Pyrene aromatic-plane distance from graphene (A)"
)

ax.set_ylabel(
    "Relative graphene interaction energy (kJ/mol)"
)

ax.set_title(
    "Rigid Vertical Interaction-Energy Scan: "
    "GAFF2 Ligand vs. IFF Graphene"
)


ax.grid(
    alpha=0.25,
)


ax.legend(
    frameon=False,
)


fig.tight_layout()


fig.savefig(
    FIGURE_OUT,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 72)
print("SCAN RESULTS")
print("=" * 72)

print()

print(
    "FULL PYRENE-PEG5-PROPARGYL"
)

print(
    f"  Minimum distance: "
    f"{full_min_distance_A:.3f} A"
)

print(
    f"  Interaction energy: "
    f"{full_min_energy:.3f} kJ/mol"
)

print()

print(
    "PYRENE AROMATIC CONTRIBUTION"
)

print(
    f"  Minimum distance: "
    f"{pyrene_min_distance_A:.3f} A"
)

print(
    f"  Interaction energy: "
    f"{pyrene_min_energy:.3f} kJ/mol"
)

print()

print(
    f"Observed MD mean distance: "
    f"{MD_MEAN_DISTANCE_A:.4f} A"
)

print()

print(
    "IMPORTANT:"
)

print(
    "This is a rigid-coordinate interaction-energy scan, "
    "not a free-energy profile."
)

print(
    "The ligand conformation is taken from the 500 ps "
    "trajectory frame and translated only in z."
)

print()

print(
    f"CSV:    {CSV_OUT}"
)

print(
    f"Figure: {FIGURE_OUT}"
)

print()

print(
    "INTERACTION_ENERGY_SCAN_PASS"
)
