from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openmm as mm
from openmm import unit


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

MOTION_CSV = Path(
    "analysis/pyrene_motion_110to500ps.csv"
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
    / "07_interaction_energy_scan.png"
)


# ============================================================
# KNOWN SYSTEM INDEXING
# ============================================================

# Graphene carbon cores
N_GRAPHENE_CARBONS = 1250

# Full IFF graphene:
# 0-1249    carbon cores
# 1250-2499 upper pi particles
# 2500-3749 lower pi particles
GRAPHENE_START = 0
GRAPHENE_END = 3750

# Pyrene-PEG5-propargyl:
# 3750-3819
LIGAND_START = 3750
LIGAND_END = 3820

# Exact 16 aromatic pyrene atoms previously validated
# from the MOL2 aromatic-atom detector.
PYRENE_LOCAL = np.arange(
    2,
    18,
    dtype=int,
)

PYRENE_GLOBAL = (
    LIGAND_START
    +
    PYRENE_LOCAL
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
# SCAN SETTINGS
# ============================================================

DISTANCE_MIN_A = 2.50
DISTANCE_MAX_A = 8.00
DISTANCE_STEP_A = 0.05

# Far-away reference used to define E = 0.
REFERENCE_DISTANCE_A = 50.0


# ============================================================
# CHECK INPUT FILES
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    MOTION_CSV,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# LOAD MD OBSERVATION
# ============================================================

motion = pd.read_csv(
    MOTION_CSV
)

md_height_A = (
    motion["pyrene_height_nm"]
    .to_numpy(dtype=float)
    * 10.0
)

MD_MEAN_A = float(
    md_height_A.mean()
)

MD_STD_A = float(
    md_height_A.std()
)


# ============================================================
# LOAD VACUUM FORCE FIELD
# ============================================================

system_full = mm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text()
)

if system_full.getNumParticles() != 3820:
    raise RuntimeError(
        "Expected 3820 particles in vacuum assembly, "
        f"found {system_full.getNumParticles()}."
    )


# ============================================================
# FIND NONBONDED FORCE
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
        "No NonbondedForce found in system XML."
    )


if original_nb.getNumParticles() != 3820:
    raise RuntimeError(
        "Nonbonded particle count mismatch."
    )


# ============================================================
# VERIFY NO GRAPHENE-LIGAND CROSS EXCEPTIONS
# ============================================================

graphene_set = set(
    GRAPHENE_INDICES.tolist()
)

ligand_set = set(
    LIGAND_INDICES.tolist()
)

cross_exceptions = []

for k in range(
    original_nb.getNumExceptions()
):

    (
        atom1,
        atom2,
        charge_prod,
        sigma,
        epsilon,
    ) = original_nb.getExceptionParameters(k)

    i = int(atom1)
    j = int(atom2)

    cross = (
        (
            i in graphene_set
            and j in ligand_set
        )
        or
        (
            j in graphene_set
            and i in ligand_set
        )
    )

    if cross:
        cross_exceptions.append(
            (k, i, j)
        )


if cross_exceptions:
    raise RuntimeError(
        "Unexpected graphene-ligand cross exceptions: "
        f"{cross_exceptions[:10]}"
    )


# ============================================================
# LOAD 500 ps STRUCTURE
# ============================================================

all_positions_nm = np.load(
    POSITIONS_FILE
)

if all_positions_nm.shape[0] < 3820:
    raise RuntimeError(
        "Position array contains fewer than 3820 particles."
    )

positions_nm = np.array(
    all_positions_nm[:3820],
    dtype=float,
    copy=True,
)


# ============================================================
# INITIAL 500 ps SEPARATION
# ============================================================

graphene_mean_z_nm = float(
    positions_nm[
        :N_GRAPHENE_CARBONS,
        2
    ].mean()
)

pyrene_mean_z_nm = float(
    positions_nm[
        PYRENE_GLOBAL,
        2
    ].mean()
)

initial_height_nm = (
    pyrene_mean_z_nm
    -
    graphene_mean_z_nm
)

initial_height_A = (
    initial_height_nm
    * 10.0
)


# ============================================================
# BUILD EXACT CROSS-INTERACTION SYSTEM
# ============================================================

def build_cross_system(
    group_a,
    group_b,
):
    system = mm.System()

    for i in range(
        system_full.getNumParticles()
    ):
        system.addParticle(
            system_full.getParticleMass(i)
        )

    # q: elementary charge
    # sigma: nm
    # epsilon: kJ/mol
    #
    # Coulomb constant:
    # 138.935456 kJ mol^-1 nm e^-2
    expression = (
        "138.935456*q1*q2/r"
        " + "
        "4*eps*((sig/r)^12-(sig/r)^6);"
        "sig=0.5*(sigma1+sigma2);"
        "eps=sqrt(epsilon1*epsilon2)"
    )

    force = mm.CustomNonbondedForce(
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

    for i in range(
        original_nb.getNumParticles()
    ):

        (
            charge,
            sigma,
            epsilon,
        ) = original_nb.getParticleParameters(i)

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
        mm.CustomNonbondedForce.NoCutoff
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


full_cross_system = build_cross_system(
    GRAPHENE_INDICES,
    LIGAND_INDICES,
)

pyrene_cross_system = build_cross_system(
    GRAPHENE_INDICES,
    PYRENE_GLOBAL,
)


# ============================================================
# USE CPU
#
# This scan is small. CPU avoids involving the GPU while
# giving deterministic force-field energy evaluations.
# ============================================================

platform_names = [
    mm.Platform.getPlatform(i).getName()
    for i in range(
        mm.Platform.getNumPlatforms()
    )
]

if "CPU" in platform_names:
    platform = mm.Platform.getPlatformByName(
        "CPU"
    )
else:
    platform = mm.Platform.getPlatformByName(
        "Reference"
    )


integrator_full = mm.VerletIntegrator(
    1.0 * unit.femtosecond
)

integrator_pyrene = mm.VerletIntegrator(
    1.0 * unit.femtosecond
)


context_full = mm.Context(
    full_cross_system,
    integrator_full,
    platform,
)

context_pyrene = mm.Context(
    pyrene_cross_system,
    integrator_pyrene,
    platform,
)


# ============================================================
# RIGID VERTICAL TRANSLATION
# ============================================================

def make_positions(
    target_distance_A,
):
    target_distance_nm = (
        target_distance_A
        * 0.1
    )

    dz_nm = (
        target_distance_nm
        -
        initial_height_nm
    )

    xyz = np.array(
        positions_nm,
        copy=True,
    )

    # Move the ENTIRE ligand rigidly.
    xyz[
        LIGAND_START:LIGAND_END,
        2
    ] += dz_nm

    return xyz


# ============================================================
# ENERGY EVALUATION
# ============================================================

def interaction_energy(
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

    return (
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


# ============================================================
# FAR-SEPARATION ZERO
# ============================================================

far_positions = make_positions(
    REFERENCE_DISTANCE_A
)

reference_full = interaction_energy(
    context_full,
    far_positions,
)

reference_pyrene = interaction_energy(
    context_pyrene,
    far_positions,
)


# ============================================================
# DISTANCE GRID
# ============================================================

distances_A = np.arange(
    DISTANCE_MIN_A,
    DISTANCE_MAX_A
    +
    0.5 * DISTANCE_STEP_A,
    DISTANCE_STEP_A,
)


full_energies = []
pyrene_energies = []


print()
print("=" * 76)
print("PYRENE / GRAPHENE RIGID INTERACTION-ENERGY SCAN")
print("=" * 76)

print(
    f"Platform:                {platform.getName()}"
)

print(
    f"Vacuum particles:        "
    f"{system_full.getNumParticles()}"
)

print(
    f"Cross exceptions:        "
    f"{len(cross_exceptions)}"
)

print(
    f"500 ps separation:       "
    f"{initial_height_A:.4f} Angstrom"
)

print(
    f"110-500 ps MD mean:      "
    f"{MD_MEAN_A:.4f} +/- "
    f"{MD_STD_A:.4f} Angstrom"
)

print(
    f"Energy zero reference:   "
    f"{REFERENCE_DISTANCE_A:.1f} Angstrom"
)

print()
print("Starting scan...")
print()


for number, distance_A in enumerate(
    distances_A,
    start=1,
):

    xyz = make_positions(
        distance_A
    )

    full_E = (
        interaction_energy(
            context_full,
            xyz,
        )
        -
        reference_full
    )

    pyrene_E = (
        interaction_energy(
            context_pyrene,
            xyz,
        )
        -
        reference_pyrene
    )

    full_energies.append(
        full_E
    )

    pyrene_energies.append(
        pyrene_E
    )

    if (
        number == 1
        or
        number % 10 == 0
        or
        number == len(distances_A)
    ):
        print(
            f"{number:3d}/"
            f"{len(distances_A)}   "
            f"d={distance_A:5.2f} A   "
            f"Full={full_E:10.3f}   "
            f"Pyrene={pyrene_E:10.3f}"
        )


full_energies = np.asarray(
    full_energies,
    dtype=float,
)

pyrene_energies = np.asarray(
    pyrene_energies,
    dtype=float,
)


# ============================================================
# FIND ACTUAL CALCULATED MINIMA
# ============================================================

full_min_idx = int(
    np.argmin(
        full_energies
    )
)

pyrene_min_idx = int(
    np.argmin(
        pyrene_energies
    )
)


full_min_distance_A = float(
    distances_A[
        full_min_idx
    ]
)

full_min_energy = float(
    full_energies[
        full_min_idx
    ]
)


pyrene_min_distance_A = float(
    distances_A[
        pyrene_min_idx
    ]
)

pyrene_min_energy = float(
    pyrene_energies[
        pyrene_min_idx
    ]
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

output = np.column_stack(
    [
        distances_A,
        full_energies,
        pyrene_energies,
    ]
)

np.savetxt(
    CSV_OUT,
    output,
    delimiter=",",
    header=(
        "distance_A,"
        "full_ligand_graphene_kJ_mol,"
        "pyrene_aromatic_graphene_kJ_mol"
    ),
    comments="",
    fmt="%.8f",
)


# ============================================================
# PLOT
# ============================================================

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


fig, ax = plt.subplots(
    figsize=(11, 6.7)
)


# Pyrene aromatic contribution
ax.plot(
    distances_A,
    pyrene_energies,
    marker="o",
    markevery=5,
    markersize=4,
    linewidth=2.0,
    label="Pyrene aromatic contribution",
)


# Entire linker
ax.plot(
    distances_A,
    full_energies,
    marker="s",
    markevery=5,
    markersize=4,
    linewidth=2.0,
    label="Full Pyrene-PEG5-Propargyl",
)


# E = 0
ax.axhline(
    0.0,
    linewidth=1.1,
)


# Calculated minimum
ax.axvline(
    full_min_distance_A,
    linestyle="--",
    linewidth=1.8,
    label=(
        "Calculated full-ligand minimum = "
        f"{full_min_distance_A:.2f} Angstrom"
    ),
)


# MD-observed region
ax.axvspan(
    MD_MEAN_A - MD_STD_A,
    MD_MEAN_A + MD_STD_A,
    alpha=0.15,
    label=(
        "300 K MD mean +/- 1 SD"
    ),
)


ax.axvline(
    MD_MEAN_A,
    linestyle=":",
    linewidth=2.0,
    label=(
        "300 K MD mean = "
        f"{MD_MEAN_A:.2f} Angstrom"
    ),
)


# Mark energy minimum
ax.scatter(
    [
        full_min_distance_A
    ],
    [
        full_min_energy
    ],
    s=85,
    zorder=5,
)


ax.annotate(
    (
        f"Energy minimum\n"
        f"d = {full_min_distance_A:.2f} Angstrom\n"
        f"E = {full_min_energy:.1f} kJ/mol"
    ),
    xy=(
        full_min_distance_A,
        full_min_energy,
    ),
    xytext=(
        full_min_distance_A + 0.7,
        full_min_energy + 0.25
        * (
            max(full_energies.max(), 1.0)
            -
            full_min_energy
        ),
    ),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.2,
    },
)


ax.set_xlabel(
    r"Pyrene aromatic-plane distance from graphene ($\AA$)"
)

ax.set_ylabel(
    "Relative interaction energy (kJ/mol)"
)

ax.set_title(
    "Interaction-Energy Scan: "
    "GAFF2 Pyrene-PEG5 vs. IFF Graphene"
)

ax.grid(
    linestyle="--",
    alpha=0.3,
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
print("=" * 76)
print("SCAN RESULTS")
print("=" * 76)

print()
print("FULL PYRENE-PEG5-PROPARGYL")

print(
    f"  Minimum distance:    "
    f"{full_min_distance_A:.3f} Angstrom"
)

print(
    f"  Interaction energy:  "
    f"{full_min_energy:.3f} kJ/mol"
)

print()

print("PYRENE AROMATIC CONTRIBUTION")

print(
    f"  Minimum distance:    "
    f"{pyrene_min_distance_A:.3f} Angstrom"
)

print(
    f"  Interaction energy:  "
    f"{pyrene_min_energy:.3f} kJ/mol"
)

print()

print("INDEPENDENT 300 K MD RESULT")

print(
    f"  Mean distance:       "
    f"{MD_MEAN_A:.3f} +/- "
    f"{MD_STD_A:.3f} Angstrom"
)

print(
    f"  Final 500 ps frame:  "
    f"{initial_height_A:.3f} Angstrom"
)

print()

print(
    "NOTE: This is a rigid vertical interaction-energy scan."
)

print(
    "It is NOT an adsorption free-energy calculation."
)

print(
    "The 500 ps ligand conformation is translated in z "
    "without changing its internal geometry."
)

print()

print(
    f"Raw CSV: {CSV_OUT}"
)

print(
    f"Figure:  {FIGURE_OUT}"
)

print()

print(
    "INTERACTION_ENERGY_SCAN_PASS"
)
