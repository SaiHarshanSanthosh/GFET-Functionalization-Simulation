from pathlib import Path
import json
import numpy as np
from openmm import openmm, unit

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_XML = ROOT / "parameters/combined/ff3_explicit_dpbs_opc_yb_replicaB.xml"
POSITIONS = ROOT / "parameters/combined/ff3_explicit_dpbs_opc_yb_replicaB_production_300K_10ns_positions_nm.npy"
VELOCITIES = ROOT / "parameters/combined/ff3_explicit_dpbs_opc_yb_replicaB_production_300K_10ns_velocities_nm_per_ps.npy"
DESIGN = ROOT / "analysis/ff3_explicit_dpbs_opc_yb_replicaB_na_screening_probe_design.json"

N_GRAPHENE_CARBONS = 1250
FORCE_GROUP = 30
K_RESTRAINT = 500.0  # kJ/mol/nm^2

for path in [SYSTEM_XML, POSITIONS, VELOCITIES, DESIGN]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

design = json.loads(DESIGN.read_text(encoding="utf-8"))

probe_index = int(
    design["probe"]["system_particle_index"]
)

positions_nm = np.load(POSITIONS)
velocities_nm_ps = np.load(VELOCITIES)

with SYSTEM_XML.open("r", encoding="utf-8") as f:
    system = openmm.XmlSerializer.deserialize(f.read())

if positions_nm.shape != (system.getNumParticles(), 3):
    raise RuntimeError("Position/System particle mismatch.")

if velocities_nm_ps.shape != (system.getNumParticles(), 3):
    raise RuntimeError("Velocity/System particle mismatch.")

# ------------------------------------------------------------
# Verify force-group slot is free
# ------------------------------------------------------------

used_groups = [
    system.getForce(i).getForceGroup()
    for i in range(system.getNumForces())
]

if FORCE_GROUP in used_groups:
    raise RuntimeError(
        f"Force group {FORCE_GROUP} is already occupied."
    )

force_names = [
    system.getForce(i).getName() or ""
    for i in range(system.getNumForces())
]

if not any("Yeh-Berkowitz" in name for name in force_names):
    raise RuntimeError("Yeh-Berkowitz force not found.")

if not any("GrapheneSupport" in name for name in force_names):
    raise RuntimeError("Graphene support force not found.")

# ------------------------------------------------------------
# Current geometry
# ------------------------------------------------------------

graphene_mean_z = float(
    np.mean(
        positions_nm[:N_GRAPHENE_CARBONS, 2]
    )
)

probe_z = float(
    positions_nm[probe_index, 2]
)

current_height = (
    probe_z
    -
    graphene_mean_z
)

# ------------------------------------------------------------
# Add harmonic z restraint:
#
# U = 1/2 k (z_probe - z_graphene_centroid - z_target)^2
# ------------------------------------------------------------

restraint = openmm.CustomCentroidBondForce(
    2,
    "0.5*k_probe*(z1-z2-z_target)^2"
)

restraint.setName(
    "NaScreeningProbeZRestraint"
)

restraint.setForceGroup(
    FORCE_GROUP
)

restraint.addGlobalParameter(
    "k_probe",
    K_RESTRAINT
)

restraint.addGlobalParameter(
    "z_target",
    current_height
)

probe_group = restraint.addGroup(
    [probe_index],
    [1.0]
)

graphene_group = restraint.addGroup(
    list(range(N_GRAPHENE_CARBONS)),
    [1.0] * N_GRAPHENE_CARBONS
)

restraint.addBond(
    [probe_group, graphene_group],
    []
)

system.addForce(restraint)

# ------------------------------------------------------------
# Create Context. NO dynamics are run.
# ------------------------------------------------------------

integrator = openmm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.001 * unit.picoseconds
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
    {"Precision": "mixed"}
)

context.setPositions(
    positions_nm * unit.nanometer
)

context.setVelocities(
    velocities_nm_ps
    * unit.nanometer
    / unit.picosecond
)

# ------------------------------------------------------------
# Test 1: target = current height.
# Restraint energy should be ~0.
# ------------------------------------------------------------

state = context.getState(
    getEnergy=True,
    groups={FORCE_GROUP}
)

E_zero = (
    state.getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)

# ------------------------------------------------------------
# Test 2: mathematically change target to 0.5 nm.
# Still NO MD.
# ------------------------------------------------------------

test_target = 0.5

context.setParameter(
    "z_target",
    test_target
)

state = context.getState(
    getEnergy=True,
    groups={FORCE_GROUP}
)

E_test = (
    state.getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)

expected = (
    0.5
    *
    K_RESTRAINT
    *
    (current_height - test_target)**2
)

error = abs(
    E_test - expected
)

relative_error = (
    error
    /
    max(abs(expected), 1.0)
)

if abs(E_zero) > 1e-4:
    raise RuntimeError(
        f"Zero-offset restraint energy is {E_zero} kJ/mol."
    )

if relative_error > 1e-5:
    raise RuntimeError(
        "OpenMM restraint energy does not match "
        "the analytical harmonic expression."
    )

print("=" * 76)
print("Na+ SCREENING-PROBE RESTRAINT PREFLIGHT")
print("=" * 76)

print()
print(f"Probe system particle:       {probe_index}")
print(f"Graphene carbons:            0-{N_GRAPHENE_CARBONS-1}")
print(f"Graphene mean z:             {graphene_mean_z:.6f} nm")
print(f"Probe z:                     {probe_z:.6f} nm")
print(f"Current probe height:        {current_height:.6f} nm")

print()
print(f"Restraint k:                 {K_RESTRAINT:.1f} kJ/mol/nm^2")
print(f"Force group:                 {FORCE_GROUP}")
print(f"Energy at current height:    {E_zero:.9f} kJ/mol")

print()
print("0.5 nm target dry test:")
print(f"  OpenMM energy:             {E_test:.6f} kJ/mol")
print(f"  Analytical energy:         {expected:.6f} kJ/mol")
print(f"  Relative error:            {relative_error:.3e}")

print()
print("MD steps executed:           0")
print()
print("Na+ SCREENING-PROBE RESTRAINT PREFLIGHT: PASS")
