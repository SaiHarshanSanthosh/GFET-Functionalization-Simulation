from ase.build import graphene
from ase.neighborlist import neighbor_list

import numpy as np

from itertools import combinations
from pathlib import Path

from openmm import openmm, unit


# ============================================================
# 1. BUILD GRAPHENE LATTICE
# ============================================================

sheet = graphene(
    formula="C2",
    a=2.46,
    size=(25, 50, 1),
    vacuum=20.0
)

# Periodic in the graphene plane, nonperiodic conceptually in z.
# OpenMM still receives a finite z box vector.
sheet.pbc = (True, True, False)

carbon_positions = sheet.get_positions()  # Angstrom
cell = sheet.cell.array                   # Angstrom

n_carbon = len(carbon_positions)


# ============================================================
# 2. ADD IFF PI PSEUDO-PARTICLES
#
# 2017 graphitic IFF representation:
#
# cg1 carbon core:
#     mass   = 10.011 amu
#     charge = +0.2 e
#
# each carbon has TWO cge pi pseudo-particles:
#     mass   = 1.000 amu
#     charge = -0.1 e
#
# equilibrium cg1-cge distance:
#     0.65 Angstrom
#
# NOTE:
# Charges are not yet used in this bonded-only model.
# They will be added during the nonbonded-force step.
# ============================================================

pi_distance = 0.65  # Angstrom

# Unit vector normal to graphene plane
normal = np.cross(cell[0], cell[1])
normal = normal / np.linalg.norm(normal)

pi_above = carbon_positions + pi_distance * normal
pi_below = carbon_positions - pi_distance * normal


# Particle indexing:
#
# carbon i:
#     0 ... n_carbon-1
#
# pi above carbon i:
#     n_carbon + i
#
# pi below carbon i:
#     2*n_carbon + i

all_positions = np.vstack(
    [
        carbon_positions,
        pi_above,
        pi_below
    ]
)

n_pi = 2 * n_carbon
n_total = 3 * n_carbon


# ============================================================
# 3. FIND GRAPHENE C-C BONDS
#
# Graphene nearest-neighbor distance is about 1.42 Angstrom.
#
# A 1.60 Angstrom cutoff includes nearest neighbors while
# excluding the next-nearest neighbor shell.
# ============================================================

i_list, j_list, shifts = neighbor_list(
    "ijS",
    sheet,
    cutoff=1.60,
    self_interaction=False
)

# Dictionary:
#
#     (lower_atom_index, higher_atom_index)
#         ->
#     periodic image shift
#
# This removes duplicate i-j / j-i neighbor-list entries.

cc_bond_records = {}

for i, j, shift in zip(i_list, j_list, shifts):

    i = int(i)
    j = int(j)

    if i < j:

        key = (i, j)

        canonical_shift = np.array(
            shift,
            dtype=int
        )

    elif j < i:

        key = (j, i)

        canonical_shift = -np.array(
            shift,
            dtype=int
        )

    else:
        continue

    cc_bond_records[key] = canonical_shift


cc_bonds = sorted(
    cc_bond_records.keys()
)


# ============================================================
# 4. VERIFY GRAPHENE CONNECTIVITY
# ============================================================

neighbors = {
    i: set()
    for i in range(n_carbon)
}

for i, j in cc_bonds:

    neighbors[i].add(j)
    neighbors[j].add(i)


degrees = [
    len(neighbors[i])
    for i in range(n_carbon)
]


print("Carbon atoms:", n_carbon)
print("Pi pseudo-particles:", n_pi)
print("Total particles:", n_total)

print()

print(
    "C-C bonds:",
    len(cc_bonds)
)

print(
    "Carbon neighbor count:",
    min(degrees),
    "to",
    max(degrees)
)


# Every carbon in infinite periodic graphene should have
# exactly three carbon neighbors.

if not all(d == 3 for d in degrees):

    raise RuntimeError(
        "Graphene connectivity failed: "
        "every carbon must have exactly 3 carbon neighbors."
    )


# Expected number of C-C bonds:
#
# 1250 atoms * 3 neighbors / 2 = 1875

expected_cc_bonds = (
    n_carbon * 3
) // 2

if len(cc_bonds) != expected_cc_bonds:

    raise RuntimeError(
        f"Expected {expected_cc_bonds} C-C bonds, "
        f"found {len(cc_bonds)}."
    )


# ============================================================
# 5. VERIFY C-C BOND LENGTHS INCLUDING PERIODIC EDGES
# ============================================================

cc_lengths = []

for (i, j), shift in cc_bond_records.items():

    displacement = (
        carbon_positions[j]
        + np.dot(shift, cell)
        - carbon_positions[i]
    )

    distance = np.linalg.norm(
        displacement
    )

    cc_lengths.append(distance)


print(
    "C-C bond length range:",
    f"{min(cc_lengths):.6f}",
    "to",
    f"{max(cc_lengths):.6f}",
    "Angstrom"
)


# ============================================================
# 6. CREATE OPENMM SYSTEM
# ============================================================

system = openmm.System()


# IFF redistributes the normal carbon mass:
#
# cg1 = 10.011 amu
# cge = 1.000 amu each
#
# Total:
#
# 10.011 + 1 + 1 = 12.011 amu

for _ in range(n_carbon):

    system.addParticle(
        10.011 * unit.dalton
    )


for _ in range(n_pi):

    system.addParticle(
        1.000 * unit.dalton
    )


if system.getNumParticles() != n_total:

    raise RuntimeError(
        "OpenMM particle count does not match expected IFF particle count."
    )


# ============================================================
# 7. SET PERIODIC SIMULATION BOX
#
# ASE coordinates are Angstrom.
# OpenMM uses nanometers internally.
#
# 1 Angstrom = 0.1 nm
# ============================================================

a = (
    openmm.Vec3(
        *(cell[0] / 10.0)
    )
    * unit.nanometer
)

# The 25x50 ASE lattice has b_x = -a_x, which is a valid
# lattice basis but is not in OpenMM's required reduced form.
#
# Use the equivalent periodic lattice vector:
#
#     b_reduced = b + a
#
# This changes the lattice basis only, not the physical sheet.
b_reduced_angstrom = (
    cell[1]
    +
    cell[0]
)

b = (
    openmm.Vec3(
        *(b_reduced_angstrom / 10.0)
    )
    * unit.nanometer
)

c = (
    openmm.Vec3(
        *(cell[2] / 10.0)
    )
    * unit.nanometer
)

system.setDefaultPeriodicBoxVectors(
    a,
    b,
    c
)


# ============================================================
# 8. IFF BOND TERMS
#
# CHARMM-style bond energy:
#
#     E = K * (r-r0)^2
#
#
# cg1-cg1:
#
#     r0 = 1.42 Angstrom
#     K  = 480 kcal/(mol Angstrom^2)
#
#
# cg1-cge:
#
#     r0 = 0.65 Angstrom
#     K  = 250 kcal/(mol Angstrom^2)
#
#
# We use CustomBondForce so that the published energy
# expression is implemented directly rather than relying
# on OpenMM's built-in harmonic 1/2 factor convention.
# ============================================================

bond_force = openmm.CustomBondForce(
    "K*(r-r0)^2"
)

bond_force.addPerBondParameter(
    "r0"
)

bond_force.addPerBondParameter(
    "K"
)

bond_force.setUsesPeriodicBoundaryConditions(
    True
)

# Force group 0 = bonds
bond_force.setForceGroup(0)


# Convert:
#
# kcal -> kJ:
#     multiply by 4.184
#
# Angstrom^-2 -> nm^-2:
#     multiply by 100

K_CC = (
    480.0
    * 4.184
    * 100.0
)

K_CPI = (
    250.0
    * 4.184
    * 100.0
)

R_CC = 1.42 / 10.0
R_CPI = 0.65 / 10.0


# ----------------------------
# C-C bonds
# ----------------------------

for i, j in cc_bonds:

    bond_force.addBond(
        i,
        j,
        [
            R_CC,
            K_CC
        ]
    )


# ----------------------------
# C-pi bonds
# ----------------------------

for i in range(n_carbon):

    pi_up = n_carbon + i
    pi_down = 2 * n_carbon + i

    bond_force.addBond(
        i,
        pi_up,
        [
            R_CPI,
            K_CPI
        ]
    )

    bond_force.addBond(
        i,
        pi_down,
        [
            R_CPI,
            K_CPI
        ]
    )


system.addForce(
    bond_force
)


# ============================================================
# 9. IFF ANGLE TERMS
#
# CHARMM-style angle energy:
#
#     E = Ktheta * (theta-theta0)^2
#
#
# cg1-cg1-cg1:
#
#     theta0 = 120 degrees
#     K      = 90 kcal/(mol rad^2)
#
#
# cg1-cg1-cge:
#
#     theta0 = 90 degrees
#     K      = 50 kcal/(mol rad^2)
#
#
# cge-cg1-cge:
#
#     theta0 = 180 degrees
#     K      = 50 kcal/(mol rad^2)
# ============================================================

angle_force = openmm.CustomAngleForce(
    "K*(theta-theta0)^2"
)

angle_force.addPerAngleParameter(
    "theta0"
)

angle_force.addPerAngleParameter(
    "K"
)

angle_force.setUsesPeriodicBoundaryConditions(
    True
)

# Force group 1 = angles
angle_force.setForceGroup(1)


K_CCC = 90.0 * 4.184
K_CCPI = 50.0 * 4.184
K_PICPI = 50.0 * 4.184

THETA_CCC = np.deg2rad(
    120.0
)

THETA_CCPI = np.deg2rad(
    90.0
)

THETA_PICPI = np.deg2rad(
    180.0
)


n_ccc_angles = 0
n_ccpi_angles = 0
n_picpi_angles = 0


for center in range(n_carbon):

    carbon_neighbors = sorted(
        neighbors[center]
    )

    pi_up = n_carbon + center
    pi_down = 2 * n_carbon + center

    # --------------------------------------------------------
    # C-C-C angles
    #
    # Three neighbors around each carbon produce:
    #
    # C(3,2) = 3 angles per carbon
    # --------------------------------------------------------

    for atom1, atom3 in combinations(
        carbon_neighbors,
        2
    ):

        angle_force.addAngle(
            atom1,
            center,
            atom3,
            [
                THETA_CCC,
                K_CCC
            ]
        )

        n_ccc_angles += 1

    # --------------------------------------------------------
    # C-C-pi angles
    #
    # Each central carbon has:
    #
    # 3 carbon neighbors
    # x
    # 2 pi sites
    #
    # = 6 C-C-pi angles
    # --------------------------------------------------------

    for neighbor in carbon_neighbors:

        angle_force.addAngle(
            neighbor,
            center,
            pi_up,
            [
                THETA_CCPI,
                K_CCPI
            ]
        )

        angle_force.addAngle(
            neighbor,
            center,
            pi_down,
            [
                THETA_CCPI,
                K_CCPI
            ]
        )

        n_ccpi_angles += 2

    # --------------------------------------------------------
    # pi-C-pi angle
    #
    # One per carbon.
    # --------------------------------------------------------

    angle_force.addAngle(
        pi_up,
        center,
        pi_down,
        [
            THETA_PICPI,
            K_PICPI
        ]
    )

    n_picpi_angles += 1


system.addForce(
    angle_force
)


# ============================================================
# 10. IFF OUT-OF-PLANE / IMPROPER TERM
#
# Graphitic cg1-cg1-cg1-cg1:
#
#     Kchi  = 0.37 kcal/mol
#     n     = 2
#     phase = 180 degrees
#
#
# Periodic energy:
#
#     E = Kchi * [1 + cos(n*theta - phase)]
#
#
# IMPORTANT:
#
# For the CHARMM/CVFF improper convention being reproduced,
# the CENTRAL / symmetry carbon is listed FIRST:
#
#     center - neighbor1 - neighbor2 - neighbor3
#
#
# cge-containing out-of-plane terms have zero force and
# therefore are not added.
# ============================================================

oop_force = openmm.PeriodicTorsionForce()

oop_force.setUsesPeriodicBoundaryConditions(
    True
)

# Force group 2 = out-of-plane terms
oop_force.setForceGroup(2)


K_OOP = (
    0.37
    * 4.184
)

OOP_PERIODICITY = 2

OOP_PHASE = np.deg2rad(
    180.0
)


n_oop = 0


for center in range(n_carbon):

    nbrs = sorted(
        neighbors[center]
    )

    if len(nbrs) != 3:

        raise RuntimeError(
            f"Carbon {center} has "
            f"{len(nbrs)} neighbors; expected 3."
        )

    n1, n2, n3 = nbrs

    # CHARMM/CVFF improper ordering:
    #
    # CENTER - neighbor - neighbor - neighbor

    oop_force.addTorsion(
        center,
        n1,
        n2,
        n3,
        OOP_PERIODICITY,
        OOP_PHASE,
        K_OOP
    )

    n_oop += 1


system.addForce(
    oop_force
)


# ============================================================
# 11. VALIDATE TOPOLOGY COUNTS
# ============================================================

n_cpi_bonds = (
    2 * n_carbon
)

total_bonds = (
    len(cc_bonds)
    + n_cpi_bonds
)

total_angles = (
    n_ccc_angles
    + n_ccpi_angles
    + n_picpi_angles
)


print()

print(
    "IFF BONDED TOPOLOGY"
)

print(
    "-------------------"
)

print(
    "C-C bonds:",
    len(cc_bonds)
)

print(
    "C-pi bonds:",
    n_cpi_bonds
)

print(
    "Total bonds:",
    total_bonds
)

print()

print(
    "C-C-C angles:",
    n_ccc_angles
)

print(
    "C-C-pi angles:",
    n_ccpi_angles
)

print(
    "pi-C-pi angles:",
    n_picpi_angles
)

print(
    "Total angles:",
    total_angles
)

print()

print(
    "Out-of-plane impropers:",
    n_oop
)


# Explicit expected counts for this topology

expected_cpi_bonds = (
    2 * n_carbon
)

expected_ccc_angles = (
    3 * n_carbon
)

expected_ccpi_angles = (
    6 * n_carbon
)

expected_picpi_angles = (
    n_carbon
)

expected_oop = (
    n_carbon
)


if n_cpi_bonds != expected_cpi_bonds:

    raise RuntimeError(
        "Incorrect number of C-pi bonds."
    )


if n_ccc_angles != expected_ccc_angles:

    raise RuntimeError(
        "Incorrect number of C-C-C angles."
    )


if n_ccpi_angles != expected_ccpi_angles:

    raise RuntimeError(
        "Incorrect number of C-C-pi angles."
    )


if n_picpi_angles != expected_picpi_angles:

    raise RuntimeError(
        "Incorrect number of pi-C-pi angles."
    )


if n_oop != expected_oop:

    raise RuntimeError(
        "Incorrect number of out-of-plane impropers."
    )


# ============================================================
# 12. CONVERT POSITIONS TO OPENMM UNITS
# ============================================================

positions_nm = (
    all_positions
    / 10.0
)

positions_openmm = unit.Quantity(
    [
        openmm.Vec3(
            x,
            y,
            z
        )
        for x, y, z
        in positions_nm
    ],
    unit.nanometer
)


# ============================================================
# 13. CREATE OPENMM CONTEXT
#
# THIS IS STILL NOT MD.
#
# We are only evaluating the force-field energy at the
# starting geometry.
# ============================================================

integrator = openmm.VerletIntegrator(
    0.001
    * unit.picoseconds
)

platform = (
    openmm.Platform
    .getPlatformByName("CUDA")
)

context = openmm.Context(
    system,
    integrator,
    platform
)

context.setPositions(
    positions_openmm
)


# ============================================================
# 14. EVALUATE TOTAL BONDED ENERGY
# ============================================================

state = context.getState(
    getEnergy=True
)

total_initial_energy = (
    state.getPotentialEnergy()
)


print()

print(
    "Initial total bonded potential energy:",
    total_initial_energy
)

print(
    "OpenMM platform:",
    platform.getName()
)


# ============================================================
# 15. MEASURE ENERGY COMPONENTS SEPARATELY
#
# Group 0 = bonds
# Group 1 = angles
# Group 2 = out-of-plane
# ============================================================

bond_state = context.getState(
    getEnergy=True,
    groups=1 << 0
)

angle_state = context.getState(
    getEnergy=True,
    groups=1 << 1
)

oop_state = context.getState(
    getEnergy=True,
    groups=1 << 2
)


bond_energy = (
    bond_state.getPotentialEnergy()
)

angle_energy = (
    angle_state.getPotentialEnergy()
)

planar_oop_energy = (
    oop_state.getPotentialEnergy()
)


print()

print(
    "Bond energy:",
    bond_energy
)

print(
    "Angle energy:",
    angle_energy
)

print(
    "Planar out-of-plane energy:",
    planar_oop_energy
)


# ============================================================
# 16. OUT-OF-PLANE VALIDATION TEST
#
# Artificially move ONE graphene carbon upward by 0.10 A.
#
# THIS IS NOT A SIMULATION.
#
# It is a controlled unit test:
#
# flat graphene
#      versus
# deliberately distorted graphene
#
# The distorted system must have greater OOP energy.
# ============================================================

distorted_positions = (
    positions_nm.copy()
)

test_carbon = 0

# Move carbon upward by 0.10 Angstrom.
#
# 0.10 A = 0.010 nm

distorted_positions[
    test_carbon,
    2
] += 0.10 / 10.0


distorted_openmm = unit.Quantity(
    [
        openmm.Vec3(
            x,
            y,
            z
        )
        for x, y, z
        in distorted_positions
    ],
    unit.nanometer
)


context.setPositions(
    distorted_openmm
)


distorted_state = context.getState(
    getEnergy=True,
    groups=1 << 2
)

distorted_oop_energy = (
    distorted_state
    .getPotentialEnergy()
)


print()

print(
    "Distorted out-of-plane energy:",
    distorted_oop_energy
)


# Convert both to the same numeric units before comparing.

planar_value = (
    planar_oop_energy
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)

distorted_value = (
    distorted_oop_energy
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


if distorted_value <= planar_value:

    raise RuntimeError(
        "Out-of-plane validation failed: "
        "moving a graphene carbon out of the plane "
        "did not increase the OOP energy."
    )


print(
    "Out-of-plane validation: PASS"
)


# Restore original geometry.

context.setPositions(
    positions_openmm
)


# ============================================================
# 17. SAVE OPENMM SYSTEM
#
# This file contains:
#
#     masses
#     C-C bonds
#     C-pi bonds
#     angles
#     out-of-plane terms
#
# It DOES NOT yet contain:
#
#     electrostatic interactions
#     Lennard-Jones interactions
#
# Therefore it is NOT yet the complete IFF graphene model.
# ============================================================

Path(
    "parameters/iff"
).mkdir(
    parents=True,
    exist_ok=True
)


output_xml = (
    "parameters/iff/"
    "graphene_iff_bonded_oop_25x50.xml"
)


with open(
    output_xml,
    "w"
) as f:

    f.write(
        openmm.XmlSerializer.serialize(
            system
        )
    )


print()

print(
    "Saved bonded + OOP OpenMM system:",
    output_xml
)

print()

print(
    "IMPORTANT: This is NOT yet the complete IFF force field."
)

print(
    "Electrostatics and Lennard-Jones terms still need to be added."
)
