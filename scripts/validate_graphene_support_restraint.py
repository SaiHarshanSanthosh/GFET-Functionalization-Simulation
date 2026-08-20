from pathlib import Path
import json

import numpy as np
from openmm import openmm, unit


# ============================================================
# CONFIGURATION
# ============================================================

N_CARBON = 1250

# Moderate support restraint.
# Units: kJ / (mol nm^2)
#
# This restrains graphene vertically without freezing
# its in-plane motion.
KZ = 1000.0

GRAPHENE_SYSTEM_XML = Path(
    "parameters/iff/graphene_iff_bonded_oop.xml"
)

SUPPORTED_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/graphene_support_restraint_validation.json"
)


# ============================================================
# 1. LOAD GRAPHENE SYSTEM
# ============================================================

with open(GRAPHENE_SYSTEM_XML, "r") as f:
    system = openmm.XmlSerializer.deserialize(f.read())

n_system = system.getNumParticles()

print("GRAPHENE SUPPORT RESTRAINT VALIDATION")
print("-------------------------------------")
print("Particles in graphene test system:", n_system)

expected_graphene_particles = 3 * N_CARBON

if n_system != expected_graphene_particles:
    raise RuntimeError(
        f"Expected {expected_graphene_particles} graphene particles, "
        f"found {n_system}."
    )


# ============================================================
# 2. LOAD SUPPORTED GEOMETRY
# ============================================================

supported_positions = np.load(SUPPORTED_POSITIONS)

print("Particles in supported solute:", len(supported_positions))

if len(supported_positions) < expected_graphene_particles:
    raise RuntimeError(
        "Supported coordinate file contains too few particles."
    )

# Only the graphene particles are needed for this standalone test.
positions_nm = supported_positions[:expected_graphene_particles].copy()

carbon_z = positions_nm[:N_CARBON, 2]

print()
print("GRAPHENE CARBON Z RANGE")
print("-----------------------")
print(f"Minimum: {carbon_z.min():.6f} nm")
print(f"Maximum: {carbon_z.max():.6f} nm")
print(f"Mean:    {carbon_z.mean():.6f} nm")


# ============================================================
# 3. BUILD VERTICAL SUPPORT RESTRAINT
#
# E = 1/2 * kz * dz^2
#
# Only graphene CARBON CORES are restrained.
#
# Pyrene, PEG5, water, and future biomolecules are NOT
# restrained by this force.
# ============================================================

support_force = openmm.CustomExternalForce(
    "0.5*kz*periodicdistance(0,0,z,0,0,z0)^2"
)

support_force.setName("GrapheneSupportZRestraint")

support_force.addGlobalParameter(
    "kz",
    KZ
)

support_force.addPerParticleParameter("z0")

# Separate force group lets us inspect ONLY the support energy.
SUPPORT_FORCE_GROUP = 5
support_force.setForceGroup(SUPPORT_FORCE_GROUP)

for i in range(N_CARBON):
    z0 = float(positions_nm[i, 2])

    support_force.addParticle(
        i,
        [z0]
    )

system.addForce(support_force)

print()
print("SUPPORT RESTRAINT")
print("-----------------")
print("Restrained particles:", support_force.getNumParticles())
print(f"Force constant: {KZ:.1f} kJ/(mol nm^2)")
print("Direction: z only")
print("Pyrene-PEG5 restrained: NO")


# ============================================================
# 4. CREATE OPENMM CONTEXT
# ============================================================

positions_openmm = unit.Quantity(
    [
        openmm.Vec3(float(x), float(y), float(z))
        for x, y, z in positions_nm
    ],
    unit.nanometer
)

integrator = openmm.VerletIntegrator(
    0.001 * unit.picoseconds
)

try:
    platform = openmm.Platform.getPlatformByName("CUDA")
except Exception:
    platform = openmm.Platform.getPlatformByName("CPU")

context = openmm.Context(
    system,
    integrator,
    platform
)

context.setPositions(positions_openmm)

print()
print("OpenMM platform:", platform.getName())


# ============================================================
# 5. SUPPORT ENERGY AT EQUILIBRIUM POSITION
# ============================================================

state = context.getState(
    getEnergy=True,
    groups=1 << SUPPORT_FORCE_GROUP
)

initial_energy = state.getPotentialEnergy().value_in_unit(
    unit.kilojoule_per_mole
)

print()
print("SUPPORT ENERGY TEST")
print("-------------------")
print(
    f"Energy at supported geometry: "
    f"{initial_energy:.8f} kJ/mol"
)


# ============================================================
# 6. ARTIFICIALLY LIFT ONE CARBON BY 0.10 nm
#
# This is NOT MD.
#
# We deliberately move one graphene carbon by 1 Angstrom
# to verify that the support force pushes it back.
# ============================================================

TEST_DISPLACEMENT_NM = 0.10
TEST_CARBON = 0

distorted = positions_nm.copy()

distorted[TEST_CARBON, 2] += TEST_DISPLACEMENT_NM

distorted_openmm = unit.Quantity(
    [
        openmm.Vec3(float(x), float(y), float(z))
        for x, y, z in distorted
    ],
    unit.nanometer
)

context.setPositions(distorted_openmm)

state = context.getState(
    getEnergy=True,
    groups=1 << SUPPORT_FORCE_GROUP
)

distorted_energy = state.getPotentialEnergy().value_in_unit(
    unit.kilojoule_per_mole
)

expected_energy = (
    0.5
    * KZ
    * TEST_DISPLACEMENT_NM**2
)

print(
    f"Energy after lifting one carbon by "
    f"{TEST_DISPLACEMENT_NM:.2f} nm: "
    f"{distorted_energy:.8f} kJ/mol"
)

print(
    f"Expected harmonic energy: "
    f"{expected_energy:.8f} kJ/mol"
)


# ============================================================
# 7. VALIDATE
# ============================================================

if abs(initial_energy) > 1e-5:
    raise RuntimeError(
        "Support energy should be approximately zero "
        "at the reference geometry."
    )

if not np.isclose(
    distorted_energy,
    expected_energy,
    rtol=1e-5,
    atol=1e-5
):
    raise RuntimeError(
        "Support restraint energy does not match "
        "the expected harmonic energy."
    )


# ============================================================
# 8. SAVE VALIDATION RECORD
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True
)

metadata = {
    "n_graphene_carbons": N_CARBON,
    "n_graphene_particles_total": expected_graphene_particles,
    "restrained_particle_start": 0,
    "restrained_particle_end": N_CARBON - 1,
    "restraint_direction": "z_only",
    "kz_kj_mol_nm2": KZ,
    "mean_graphene_z_nm": float(carbon_z.mean()),
    "test_displacement_nm": TEST_DISPLACEMENT_NM,
    "expected_test_energy_kj_mol": expected_energy,
    "measured_test_energy_kj_mol": distorted_energy,
    "pyrene_peg5_restrained": False,
    "md_run": False,
    "status": "PASS",
}

with open(OUTPUT_METADATA, "w") as f:
    json.dump(metadata, f, indent=2)

print()
print("SAVED")
print("-----")
print("Metadata:", OUTPUT_METADATA)

print()
print("GRAPHENE SUPPORT RESTRAINT: PASS")
print()
print("No MD has been run.")
print("No Pyrene-PEG5 atoms were restrained.")
print("No water or ions were modified.")
