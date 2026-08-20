from pathlib import Path
import json

import numpy as np

from openmm import openmm, unit, app
from openmm.app import element


# ============================================================
# COMPLETE SUPPORTED SOLVATED SYSTEM ASSEMBLY
#
# SYSTEM:
#
#   IFF graphene
#   +
#   neutral GAFF2 Pyrene-PEG5
#   +
#   6065 retained rigid OPC waters
#   +
#   18 Na+ and 18 Cl- ions
#   +
#   graphene z-support restraint
#
#   No runtime water/ion wall is present.
#
# IMPORTANT:
#
#   NO MD IS RUN.
#   NO MINIMIZATION IS RUN.
#
# This script only:
#
#   1. assembles the complete OpenMM System
#   2. validates OPC virtual sites/constraints
#   3. evaluates starting potential energy
#   4. evaluates starting atomic forces
#   5. saves the assembled checkpoint system
#
# ============================================================


# ============================================================
# 1. CONSTANTS
# ============================================================

N_CARBON = 1250
N_GRAPHENE = 3750
N_LIGAND = 70
N_SOLUTE = N_GRAPHENE + N_LIGAND

SITES_PER_WATER = 4

# PBS ionic-strength surrogate:
# replace whole OPC waters with monatomic ions.
N_NA = 18
N_CL = 18

ION_SELECTION_SEED = 20260820
MIN_SOLUTE_DIST_NM = 0.50
MIN_ION_DIST_NM = 0.50


# OPC saved ordering:
#
#   0 = O
#   1 = H1
#   2 = H2
#   3 = M virtual charge site
#
OPC_O = 0
OPC_H1 = 1
OPC_H2 = 2
OPC_M = 3


# ============================================================
# 2. INPUT FILES
# ============================================================

BASE_SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_vacuum_assembly_nocutoff.xml"
)

SUPPORTED_SOLUTE_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

OPC_WATER_POSITIONS = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_one_sided_exact_nm.npy"
)

PERIODIC_BOX_VECTORS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUTPUT_SYSTEM_XML = Path(
    "parameters/combined/"
    "ff3_pbs_surrogate_opc_yb.xml"
)

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "ff3_pbs_surrogate_opc_yb_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "ff3_pbs_surrogate_opc_yb_assembly_check.json"
)


# ============================================================
# 4. SUPPORTED SLAB GEOMETRY
# ============================================================

LOWER_WATER_BOUNDARY_NM = 0.100
UPPER_WATER_BOUNDARY_NM = 6.800

FINAL_BOX_Z_NM = 12.0

Z_MID_NM = (
    LOWER_WATER_BOUNDARY_NM
    + UPPER_WATER_BOUNDARY_NM
) / 2.0

HALF_WIDTH_NM = (
    UPPER_WATER_BOUNDARY_NM
    - LOWER_WATER_BOUNDARY_NM
) / 2.0


# ============================================================
# 5. EXTERNAL RESTRAINTS
# ============================================================

K_GRAPHENE = 1000.0
# kJ/(mol nm^2)



# ============================================================
# 6. NONBONDED SETTINGS
#
# For this supported slab checkpoint we use PME electrostatics.
#
# NOTE:
# Standard OpenMM PME is still a 3D-periodic electrostatic
# treatment. The large z vacuum gap reduces direct interaction
# between slab copies but is not a formal 2D slab correction.
#
# We will test sensitivity to this later before production.
# ============================================================

NONBONDED_CUTOFF_NM = 1.0

EWALD_ERROR_TOLERANCE = 5.0e-4


# ============================================================
# 7. FORCE GROUPS
#
# Allows us to inspect energy contributions independently.
# ============================================================

GROUP_GRAPHENE_BONDS = 0
GROUP_GRAPHENE_ANGLES = 1
GROUP_GRAPHENE_OOP = 2

GROUP_LIGAND_BONDS = 3
GROUP_LIGAND_ANGLES = 4

GROUP_SUPPORT = 5

GROUP_LIGAND_TORSIONS = 7
GROUP_NONBONDED = 8

GROUP_WATER_BONDS = 9
GROUP_WATER_ANGLES = 10

# Keep YB isolated in the same force group used by the
# independently validated wetting implementation.
GROUP_YB = 29

# OpenMM Coulomb constant:
# kJ mol^-1 nm e^-2
K_E = 138.935456


# ============================================================
# 8. VERIFY FILES
# ============================================================

required_files = [
    BASE_SYSTEM_XML,
    SUPPORTED_SOLUTE_POSITIONS,
    OPC_WATER_POSITIONS,
    PERIODIC_BOX_VECTORS,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 9. LOAD BASE GRAPHENE + LIGAND SYSTEM
# ============================================================

print()
print("=" * 72)
print("FULL SUPPORTED SOLVATED SYSTEM ASSEMBLY")
print("=" * 72)

print()
print("Loading base graphene + Pyrene-PEG5 System...")


with open(
    BASE_SYSTEM_XML,
    "r",
) as f:

    system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


if system.getNumParticles() != N_SOLUTE:

    raise RuntimeError(
        "Base system particle count mismatch. "
        f"Expected {N_SOLUTE}, "
        f"found {system.getNumParticles()}."
    )


print(
    "Base particles:",
    system.getNumParticles(),
)

print(
    "Base forces:",
    system.getNumForces(),
)

print(
    "Base constraints:",
    system.getNumConstraints(),
)


# ============================================================
# 10. INSPECT BASE FORCES
# ============================================================

print()
print("BASE FORCE LIST")
print("---------------")


for i in range(
    system.getNumForces()
):

    force = system.getForce(i)

    print(
        f"[{i}] "
        f"{force.__class__.__name__}"
        f" | {force.getName()}"
    )


# We expect the previously assembled vacuum system:
#
#   0 graphene bonds
#   1 graphene angles
#   2 graphene OOP
#   3 GAFF2 ligand bonds
#   4 GAFF2 ligand angles
#   5 GAFF2 ligand torsions
#   6 unified nonbonded
#
if system.getNumForces() != 7:

    raise RuntimeError(
        "Expected 7 forces in base vacuum system. "
        f"Found {system.getNumForces()}."
    )


# ============================================================
# 11. ASSIGN FORCE GROUPS TO BASE SYSTEM
# ============================================================

system.getForce(0).setForceGroup(
    GROUP_GRAPHENE_BONDS
)

system.getForce(1).setForceGroup(
    GROUP_GRAPHENE_ANGLES
)

system.getForce(2).setForceGroup(
    GROUP_GRAPHENE_OOP
)

system.getForce(3).setForceGroup(
    GROUP_LIGAND_BONDS
)

system.getForce(4).setForceGroup(
    GROUP_LIGAND_ANGLES
)

system.getForce(5).setForceGroup(
    GROUP_LIGAND_TORSIONS
)

system.getForce(6).setForceGroup(
    GROUP_NONBONDED
)


# ============================================================
# 12. LOCATE UNIFIED NONBONDED FORCE
# ============================================================

nonbonded_forces = [
    force
    for force in system.getForces()
    if isinstance(
        force,
        openmm.NonbondedForce,
    )
]


if len(nonbonded_forces) != 1:

    raise RuntimeError(
        "Expected exactly one unified NonbondedForce "
        f"in base system; found {len(nonbonded_forces)}."
    )


nb = nonbonded_forces[0]


if nb.getNumParticles() != N_SOLUTE:

    raise RuntimeError(
        "Base NonbondedForce particle count mismatch."
    )


base_exception_count = (
    nb.getNumExceptions()
)


print()
print("BASE NONBONDED FORCE")
print("--------------------")

print(
    "Particles:",
    nb.getNumParticles(),
)

print(
    "Existing exceptions:",
    base_exception_count,
)

print(
    "Name:",
    nb.getName(),
)


# ============================================================
# 13. LOAD POSITIONS
# ============================================================

solute_positions = np.load(
    SUPPORTED_SOLUTE_POSITIONS
)

water_positions = np.load(
    OPC_WATER_POSITIONS
)

box_vectors = np.load(
    PERIODIC_BOX_VECTORS
)


if solute_positions.shape != (
    N_SOLUTE,
    3,
):

    raise RuntimeError(
        "Supported solute coordinate shape mismatch. "
        f"Found {solute_positions.shape}."
    )


if (
    water_positions.ndim != 3
    or
    water_positions.shape[1:] != (
        SITES_PER_WATER,
        3,
    )
):

    raise RuntimeError(
        "OPC water coordinate shape mismatch. "
        "Expected (N_waters, 4, 3), "
        f"found {water_positions.shape}."
    )


# Derive all solvent/system counts from the actual cleaned
# OPC coordinate array rather than hardcoding the water count.
N_WATERS = int(
    water_positions.shape[0]
)

N_WATER_SITES = (
    N_WATERS
    * SITES_PER_WATER
)

N_TOTAL_EXPECTED = (
    N_SOLUTE
    + N_WATER_SITES
)


if box_vectors.shape != (
    3,
    3,
):

    raise RuntimeError(
        "Periodic box-vector shape mismatch. "
        f"Found {box_vectors.shape}."
    )


# The stored box vectors describe the original 7 nm
# construction geometry.  Build one authoritative final
# slab-PME box by preserving x/y and extending z to 12 nm.
final_box_vectors = np.array(
    box_vectors,
    dtype=float,
    copy=True,
)

final_box_vectors[2] = np.array(
    [
        0.0,
        0.0,
        FINAL_BOX_Z_NM,
    ],
    dtype=float,
)


# Triclinic minimum-image geometry.
#
# final_box_vectors stores a, b, c as ROWS, so Cartesian
# coordinates/displacements satisfy:
#
#     r_cart = r_frac @ final_box_vectors
#
# and therefore:
#
#     r_frac = r_cart @ inverse_box
#
inverse_final_box_vectors = np.linalg.inv(
    final_box_vectors
)


def minimum_image_displacement(delta_nm):

    delta_nm = np.asarray(
        delta_nm,
        dtype=float,
    )

    fractional = (
        delta_nm
        @ inverse_final_box_vectors
    )

    fractional -= np.rint(
        fractional
    )

    return (
        fractional
        @ final_box_vectors
    )


if not np.isfinite(
    solute_positions
).all():

    raise RuntimeError(
        "Solute coordinates contain NaN/Inf."
    )


if not np.isfinite(
    water_positions
).all():

    raise RuntimeError(
        "Water coordinates contain NaN/Inf."
    )


print()
print("COORDINATES")
print("-----------")

print(
    "Solute:",
    solute_positions.shape,
)

print(
    "OPC water:",
    water_positions.shape,
)


# ============================================================
# 13A. DETERMINISTIC ION-REPLACEMENT WATER SELECTION
#
# Select whole OPC waters whose O coordinates will become the
# initial Na+ / Cl- coordinates.  No System particles are
# modified here.
# ============================================================

water_oxygen_positions = (
    water_positions[
        :,
        OPC_O,
        :
    ]
)


min_water_solute_distance_nm = np.empty(
    water_oxygen_positions.shape[0],
    dtype=float,
)


for water_index, oxygen_position in enumerate(
    water_oxygen_positions
):

    displacement = (
        solute_positions
        - oxygen_position
    )

    displacement_mic = (
        minimum_image_displacement(
            displacement
        )
    )

    min_water_solute_distance_nm[
        water_index
    ] = float(
        np.min(
            np.linalg.norm(
                displacement_mic,
                axis=1,
            )
        )
    )


ion_candidate_water_indices = np.where(
    min_water_solute_distance_nm
    >= MIN_SOLUTE_DIST_NM
)[0]


rng = np.random.default_rng(
    ION_SELECTION_SEED
)

candidate_order = rng.permutation(
    ion_candidate_water_indices
)


selected_replacement_water_indices = []


for candidate_index in candidate_order:

    candidate_position = (
        water_oxygen_positions[
            candidate_index
        ]
    )

    if selected_replacement_water_indices:

        selected_positions = (
            water_oxygen_positions[
                np.asarray(
                    selected_replacement_water_indices,
                    dtype=int,
                )
            ]
        )

        ion_displacement = (
            selected_positions
            - candidate_position
        )

        ion_displacement_mic = (
            minimum_image_displacement(
                ion_displacement
            )
        )

        min_candidate_ion_distance_nm = float(
            np.min(
                np.linalg.norm(
                    ion_displacement_mic,
                    axis=1,
                )
            )
        )

        if (
            min_candidate_ion_distance_nm
            < MIN_ION_DIST_NM
        ):
            continue

    selected_replacement_water_indices.append(
        int(candidate_index)
    )

    if len(
        selected_replacement_water_indices
    ) == (
        N_NA
        + N_CL
    ):
        break


if len(
    selected_replacement_water_indices
) != (
    N_NA
    + N_CL
):

    raise RuntimeError(
        "Could not find enough valid ion-replacement "
        "water sites."
    )


na_replacement_water_indices = (
    selected_replacement_water_indices[
        :N_NA
    ]
)

cl_replacement_water_indices = (
    selected_replacement_water_indices[
        N_NA:
        N_NA + N_CL
    ]
)


# Lock against the previously validated deterministic preview.
expected_na_replacement_water_indices = [
    2389, 1878, 2738, 1075, 3368, 222,
    5396, 2965, 5828, 2918, 5761, 2607,
    5641, 5255, 5021, 663, 740, 4125,
]

expected_cl_replacement_water_indices = [
    2099, 1789, 4853, 3999, 2915, 2966,
    5151, 1101, 2780, 823, 3933, 634,
    1039, 5995, 2478, 3702, 1739, 4928,
]


if (
    na_replacement_water_indices
    != expected_na_replacement_water_indices
):

    raise RuntimeError(
        "Deterministic Na replacement-water selection "
        "does not match the validated preview."
    )


if (
    cl_replacement_water_indices
    != expected_cl_replacement_water_indices
):

    raise RuntimeError(
        "Deterministic Cl replacement-water selection "
        "does not match the validated preview."
    )


# Build coordinate bookkeeping only.
#
# Preserve the original source-water ordering for every
# retained OPC molecule.  The selected whole waters disappear;
# their O coordinates become the initial monatomic-ion
# coordinates.
replacement_water_indices_np = np.asarray(
    selected_replacement_water_indices,
    dtype=int,
)


if len(
    np.unique(
        replacement_water_indices_np
    )
) != (
    N_NA
    + N_CL
):

    raise RuntimeError(
        "Replacement-water indices are not unique."
    )


if (
    replacement_water_indices_np.min() < 0
    or
    replacement_water_indices_np.max() >= N_WATERS
):

    raise RuntimeError(
        "Replacement-water index is out of range."
    )


retain_water_mask = np.ones(
    N_WATERS,
    dtype=bool,
)

retain_water_mask[
    replacement_water_indices_np
] = False


retained_water_source_indices = np.where(
    retain_water_mask
)[0]


retained_water_positions = (
    water_positions[
        retained_water_source_indices
    ]
)


na_positions = (
    water_oxygen_positions[
        np.asarray(
            na_replacement_water_indices,
            dtype=int,
        )
    ]
)


cl_positions = (
    water_oxygen_positions[
        np.asarray(
            cl_replacement_water_indices,
            dtype=int,
        )
    ]
)


expected_retained_water_count = (
    N_WATERS
    - N_NA
    - N_CL
)


if retained_water_positions.shape != (
    expected_retained_water_count,
    SITES_PER_WATER,
    3,
):

    raise RuntimeError(
        "Retained OPC water coordinate shape mismatch. "
        f"Found {retained_water_positions.shape}."
    )


if na_positions.shape != (
    N_NA,
    3,
):

    raise RuntimeError(
        "Na coordinate shape mismatch."
    )


if cl_positions.shape != (
    N_CL,
    3,
):

    raise RuntimeError(
        "Cl coordinate shape mismatch."
    )


# Explicit target-system counts.
#
# N_WATERS / N_WATER_SITES above deliberately describe the
# original 6101-water source array.  Target-system counts use
# the explicit N_RETAINED_* / N_IONS variables below.
N_REPLACED_WATERS = (
    N_NA
    + N_CL
)

N_RETAINED_WATERS = int(
    retained_water_positions.shape[0]
)

N_RETAINED_WATER_SITES = (
    N_RETAINED_WATERS
    * SITES_PER_WATER
)

N_IONS = (
    N_NA
    + N_CL
)

N_TOTAL_IONIZED_EXPECTED = (
    N_SOLUTE
    + N_RETAINED_WATER_SITES
    + N_IONS
)


if N_REPLACED_WATERS != 36:

    raise RuntimeError(
        "Unexpected replaced-water count."
    )


if N_RETAINED_WATERS != 6065:

    raise RuntimeError(
        "Unexpected retained-water count."
    )


if N_TOTAL_IONIZED_EXPECTED != 28116:

    raise RuntimeError(
        "Unexpected ionized-system particle count."
    )


print()
print("ION-REPLACEMENT WATER SELECTION")
print("-------------------------------")

print(
    "Solute-clear candidates:",
    len(ion_candidate_water_indices),
)

print(
    "Na replacement water indices:",
    na_replacement_water_indices,
)

print(
    "Cl replacement water indices:",
    cl_replacement_water_indices,
)


# ============================================================
# 14. BUILD ONE OPC WATER REFERENCE SYSTEM
#
# Instead of hard-coding OPC masses, charges, constraints,
# and virtual-site weights, we ask OpenMM's amber19/opc.xml
# force field to construct one reference water.
#
# Then we duplicate that exact definition for all
# 6065 retained OPC waters.
# ============================================================

print()
print("BUILDING OPC REFERENCE WATER")
print("----------------------------")


reference_topology = app.Topology()

chain = reference_topology.addChain()

residue = reference_topology.addResidue(
    "HOH",
    chain,
)


atom_o = reference_topology.addAtom(
    "O",
    element.oxygen,
    residue,
)

atom_h1 = reference_topology.addAtom(
    "H1",
    element.hydrogen,
    residue,
)

atom_h2 = reference_topology.addAtom(
    "H2",
    element.hydrogen,
    residue,
)


reference_topology.addBond(
    atom_o,
    atom_h1,
)

reference_topology.addBond(
    atom_o,
    atom_h2,
)


reference_positions = unit.Quantity(
    [
        openmm.Vec3(
            *water_positions[
                0,
                OPC_O,
                :
            ]
        ),
        openmm.Vec3(
            *water_positions[
                0,
                OPC_H1,
                :
            ]
        ),
        openmm.Vec3(
            *water_positions[
                0,
                OPC_H2,
                :
            ]
        ),
    ],
    unit.nanometer,
)


opc_forcefield = app.ForceField(
    "amber19/opc.xml"
)


# ============================================================
# 14A. BUILD MONATOMIC ION REFERENCE SYSTEMS
#
# Derive Na+ and Cl- directly from the same amber19/opc.xml
# force field used for OPC water.  Do not hard-code ion
# masses, charges, or Lennard-Jones parameters.
# ============================================================

def build_monatomic_ion_reference(
    residue_name,
    atom_name,
    element_object,
):

    topology = app.Topology()
    chain = topology.addChain()

    residue = topology.addResidue(
        residue_name,
        chain,
    )

    topology.addAtom(
        atom_name,
        element_object,
        residue,
    )

    positions = unit.Quantity(
        [
            openmm.Vec3(
                0.0,
                0.0,
                0.0,
            )
        ],
        unit.nanometer,
    )

    modeller = app.Modeller(
        topology,
        positions,
    )

    ion_system = opc_forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )

    if ion_system.getNumParticles() != 1:

        raise RuntimeError(
            f"{residue_name}: expected exactly one particle; "
            f"found {ion_system.getNumParticles()}."
        )

    if ion_system.getNumConstraints() != 0:

        raise RuntimeError(
            f"{residue_name}: monatomic ion unexpectedly "
            "contains constraints."
        )

    if ion_system.isVirtualSite(0):

        raise RuntimeError(
            f"{residue_name}: monatomic ion unexpectedly "
            "defined as a virtual site."
        )

    nb_forces = [
        force
        for force in ion_system.getForces()
        if isinstance(
            force,
            openmm.NonbondedForce,
        )
    ]

    if len(nb_forces) != 1:

        raise RuntimeError(
            f"{residue_name}: expected exactly one "
            "NonbondedForce; "
            f"found {len(nb_forces)}."
        )

    ion_nb = nb_forces[0]

    if ion_nb.getNumParticles() != 1:

        raise RuntimeError(
            f"{residue_name}: ion NonbondedForce particle "
            "count mismatch."
        )

    if ion_nb.getNumExceptions() != 0:

        raise RuntimeError(
            f"{residue_name}: monatomic ion unexpectedly "
            "contains nonbonded exceptions."
        )

    mass = ion_system.getParticleMass(0)

    charge, sigma, epsilon = (
        ion_nb.getParticleParameters(0)
    )

    return {
        "system": ion_system,
        "nonbonded": ion_nb,
        "mass": mass,
        "charge": charge,
        "sigma": sigma,
        "epsilon": epsilon,
    }


na_reference = build_monatomic_ion_reference(
    "NA",
    "NA",
    element.sodium,
)

cl_reference = build_monatomic_ion_reference(
    "CL",
    "CL",
    element.chlorine,
)


print()
print("ION REFERENCE PARAMETERS")
print("------------------------")

for ion_name, ion_reference in [
    ("NA", na_reference),
    ("CL", cl_reference),
]:

    print(
        f"{ion_name}: "
        f"mass="
        f"{ion_reference['mass'].value_in_unit(unit.dalton):.8f} Da  "
        f"q="
        f"{ion_reference['charge'].value_in_unit(unit.elementary_charge):+.8f} e  "
        f"sigma="
        f"{ion_reference['sigma'].value_in_unit(unit.nanometer):.8f} nm  "
        f"epsilon="
        f"{ion_reference['epsilon'].value_in_unit(unit.kilojoule_per_mole):.8f} "
        "kJ/mol"
    )


# Exact monovalent charges are required for the PBS surrogate.
na_charge_e = (
    na_reference["charge"]
    .value_in_unit(
        unit.elementary_charge
    )
)

cl_charge_e = (
    cl_reference["charge"]
    .value_in_unit(
        unit.elementary_charge
    )
)

if abs(
    na_charge_e - 1.0
) > 1e-8:

    raise RuntimeError(
        f"Unexpected Na charge: {na_charge_e:+.12f} e."
    )

if abs(
    cl_charge_e + 1.0
) > 1e-8:

    raise RuntimeError(
        f"Unexpected Cl charge: {cl_charge_e:+.12f} e."
    )


reference_modeller = app.Modeller(
    reference_topology,
    reference_positions,
)


# Adds the OPC M-site to the topology.
reference_modeller.addExtraParticles(
    opc_forcefield
)


reference_atoms = list(
    reference_modeller.topology.atoms()
)


print(
    "Reference topology particles:",
    len(reference_atoms),
)

print(
    "Reference particle names:",
    [
        atom.name
        for atom in reference_atoms
    ],
)


if len(reference_atoms) != 4:

    raise RuntimeError(
        "OPC reference topology should have exactly "
        f"4 sites; found {len(reference_atoms)}."
    )


# ============================================================
# 15. CREATE OPC REFERENCE SYSTEM
# ============================================================

reference_system = (
    opc_forcefield.createSystem(
        reference_modeller.topology,

        nonbondedMethod=app.NoCutoff,

        constraints=None,

        rigidWater=True,

        removeCMMotion=False,
    )
)


if reference_system.getNumParticles() != 4:

    raise RuntimeError(
        "OPC reference System does not contain 4 particles."
    )


print(
    "Reference constraints:",
    reference_system.getNumConstraints(),
)

print(
    "Reference forces:",
    reference_system.getNumForces(),
)


# ============================================================
# 16. LOCATE OPC REFERENCE NONBONDED FORCE
# ============================================================

reference_nb_forces = [
    force
    for force in reference_system.getForces()
    if isinstance(
        force,
        openmm.NonbondedForce,
    )
]


if len(reference_nb_forces) != 1:

    raise RuntimeError(
        "Expected one OPC reference NonbondedForce; "
        f"found {len(reference_nb_forces)}."
    )


reference_nb = reference_nb_forces[0]


if reference_nb.getNumParticles() != 4:

    raise RuntimeError(
        "OPC reference NonbondedForce must contain 4 sites."
    )


# ============================================================
# 17. INSPECT OPC REFERENCE SITES
# ============================================================

print()
print("OPC REFERENCE SITE PARAMETERS")
print("-----------------------------")


reference_site_parameters = []


for i in range(4):

    mass = (
        reference_system
        .getParticleMass(i)
        .value_in_unit(
            unit.dalton
        )
    )

    charge, sigma, epsilon = (
        reference_nb
        .getParticleParameters(i)
    )

    charge_e = (
        charge.value_in_unit(
            unit.elementary_charge
        )
    )

    sigma_nm = (
        sigma.value_in_unit(
            unit.nanometer
        )
    )

    epsilon_kj = (
        epsilon.value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    reference_site_parameters.append(
        {
            "mass": mass,
            "charge": charge_e,
            "sigma": sigma_nm,
            "epsilon": epsilon_kj,
        }
    )

    print(
        f"Site {i}: "
        f"mass={mass:.8f} Da  "
        f"q={charge_e:+.8f} e  "
        f"sigma={sigma_nm:.8f} nm  "
        f"epsilon={epsilon_kj:.8f} kJ/mol"
    )


# ============================================================
# 18. VALIDATE OPC VIRTUAL SITE
# ============================================================

virtual_site_indices = [
    i
    for i in range(4)
    if reference_system.isVirtualSite(i)
]


print()
print(
    "Reference virtual sites:",
    virtual_site_indices,
)


if virtual_site_indices != [OPC_M]:

    raise RuntimeError(
        "Expected OPC site 3 to be the only virtual site. "
        f"Found {virtual_site_indices}."
    )


reference_virtual_site = (
    reference_system.getVirtualSite(
        OPC_M
    )
)


if not isinstance(
    reference_virtual_site,
    openmm.ThreeParticleAverageSite,
):

    raise RuntimeError(
        "Expected OPC M site to use "
        "ThreeParticleAverageSite. "
        f"Found {type(reference_virtual_site).__name__}."
    )


virtual_particles = [
    reference_virtual_site.getParticle(i)
    for i in range(
        reference_virtual_site.getNumParticles()
    )
]


virtual_weights = [
    reference_virtual_site.getWeight(i)
    for i in range(
        reference_virtual_site.getNumParticles()
    )
]


print(
    "M-site parent particles:",
    virtual_particles,
)

print(
    "M-site weights:",
    [
        float(x)
        for x in virtual_weights
    ],
)


# ============================================================
# 19. LOCATE POSSIBLE OPC INTERNAL BONDED FORCES
#
# With rigid water these are typically absent or have no
# active terms, but we reproduce whatever OpenMM's OPC
# reference system actually contains.
# ============================================================

reference_bond_forces = [
    force
    for force in reference_system.getForces()
    if isinstance(
        force,
        openmm.HarmonicBondForce,
    )
]


reference_angle_forces = [
    force
    for force in reference_system.getForces()
    if isinstance(
        force,
        openmm.HarmonicAngleForce,
    )
]


ref_bond_term_count = sum(
    force.getNumBonds()
    for force in reference_bond_forces
)

ref_angle_term_count = sum(
    force.getNumAngles()
    for force in reference_angle_forces
)


print()
print(
    "Reference harmonic bond terms:",
    ref_bond_term_count,
)

print(
    "Reference harmonic angle terms:",
    ref_angle_term_count,
)


# Reject unexpected force types.
allowed_reference_force_types = (
    openmm.NonbondedForce,
    openmm.HarmonicBondForce,
    openmm.HarmonicAngleForce,
    openmm.CMMotionRemover,
)


for force in reference_system.getForces():

    if not isinstance(
        force,
        allowed_reference_force_types,
    ):

        raise RuntimeError(
            "Unexpected force type in OPC reference System: "
            f"{force.__class__.__name__}"
        )


# ============================================================
# 20. OPTIONAL WATER BONDED FORCE CONTAINERS
# ============================================================

water_bond_force = None
water_angle_force = None


if ref_bond_term_count > 0:

    water_bond_force = (
        openmm.HarmonicBondForce()
    )

    water_bond_force.setName(
        "OPC water internal bonds"
    )

    water_bond_force.setForceGroup(
        GROUP_WATER_BONDS
    )


if ref_angle_term_count > 0:

    water_angle_force = (
        openmm.HarmonicAngleForce()
    )

    water_angle_force.setName(
        "OPC water internal angles"
    )

    water_angle_force.setForceGroup(
        GROUP_WATER_ANGLES
    )


# ============================================================
# 21. APPEND ALL OPC WATER PARTICLES
# ============================================================

print()
print(
    f"ADDING {N_RETAINED_WATERS} RETAINED OPC WATERS"
)
print("--------------------------------")


water_start_index = (
    system.getNumParticles()
)


if water_start_index != N_SOLUTE:

    raise RuntimeError(
        "Unexpected first water index."
    )


for water_index in range(
    N_RETAINED_WATERS
):

    offset = (
        N_SOLUTE
        + water_index
        * SITES_PER_WATER
    )

    # --------------------------------------------------------
    # Add four particles and their nonbonded parameters
    # --------------------------------------------------------

    for site_index in range(
        SITES_PER_WATER
    ):

        mass = (
            reference_system
            .getParticleMass(
                site_index
            )
        )

        global_index = (
            system.addParticle(
                mass
            )
        )

        expected_index = (
            offset
            + site_index
        )

        if global_index != expected_index:

            raise RuntimeError(
                "Water particle indexing mismatch."
            )

        charge, sigma, epsilon = (
            reference_nb
            .getParticleParameters(
                site_index
            )
        )

        nb_index = nb.addParticle(
            charge,
            sigma,
            epsilon,
        )

        if nb_index != expected_index:

            raise RuntimeError(
                "Nonbonded water indexing mismatch."
            )

    # --------------------------------------------------------
    # Recreate OPC M virtual site
    # --------------------------------------------------------

    global_m = (
        offset
        + OPC_M
    )

    system.setVirtualSite(
        global_m,

        openmm.ThreeParticleAverageSite(

            offset
            + virtual_particles[0],

            offset
            + virtual_particles[1],

            offset
            + virtual_particles[2],

            virtual_weights[0],

            virtual_weights[1],

            virtual_weights[2],
        )
    )

    # --------------------------------------------------------
    # Copy rigid-water constraints
    # --------------------------------------------------------

    for constraint_index in range(
        reference_system.getNumConstraints()
    ):

        p1, p2, distance = (
            reference_system
            .getConstraintParameters(
                constraint_index
            )
        )

        system.addConstraint(

            offset + p1,

            offset + p2,

            distance,
        )

    # --------------------------------------------------------
    # Copy any reference water bond terms
    # --------------------------------------------------------

    if water_bond_force is not None:

        for ref_force in reference_bond_forces:

            for bond_index in range(
                ref_force.getNumBonds()
            ):

                p1, p2, length, k = (
                    ref_force
                    .getBondParameters(
                        bond_index
                    )
                )

                water_bond_force.addBond(

                    offset + p1,

                    offset + p2,

                    length,

                    k,
                )

    # --------------------------------------------------------
    # Copy any reference water angle terms
    # --------------------------------------------------------

    if water_angle_force is not None:

        for ref_force in reference_angle_forces:

            for angle_index in range(
                ref_force.getNumAngles()
            ):

                p1, p2, p3, theta, k = (
                    ref_force
                    .getAngleParameters(
                        angle_index
                    )
                )

                water_angle_force.addAngle(

                    offset + p1,

                    offset + p2,

                    offset + p3,

                    theta,

                    k,
                )

    # --------------------------------------------------------
    # Copy OPC intramolecular nonbonded exceptions
    # --------------------------------------------------------

    for exception_index in range(
        reference_nb.getNumExceptions()
    ):

        (
            p1,
            p2,
            charge_product,
            sigma,
            epsilon,
        ) = (
            reference_nb
            .getExceptionParameters(
                exception_index
            )
        )

        nb.addException(

            offset + p1,

            offset + p2,

            charge_product,

            sigma,

            epsilon,
        )


# ============================================================
# 21A. APPEND PBS-SURROGATE MONATOMIC IONS
#
# Final particle ordering:
#
#   solute
#   + retained OPC waters
#   + Na+
#   + Cl-
#
# Ion parameters come directly from the validated
# amber19/opc.xml monatomic reference systems.
# ============================================================

print()
print("ADDING PBS-SURROGATE IONS")
print("-------------------------")

water_end_index = (
    N_SOLUTE
    + N_RETAINED_WATER_SITES
)


if system.getNumParticles() != water_end_index:

    raise RuntimeError(
        "Unexpected particle count before ion insertion. "
        f"Expected {water_end_index}, "
        f"found {system.getNumParticles()}."
    )


if nb.getNumParticles() != water_end_index:

    raise RuntimeError(
        "Unexpected NonbondedForce particle count "
        "before ion insertion."
    )


# ------------------------------------------------------------
# Na+
# ------------------------------------------------------------

na_start_index = system.getNumParticles()


for ion_index in range(
    N_NA
):

    expected_index = (
        na_start_index
        + ion_index
    )

    global_index = system.addParticle(
        na_reference["mass"]
    )

    if global_index != expected_index:

        raise RuntimeError(
            "Na particle indexing mismatch."
        )

    nb_index = nb.addParticle(
        na_reference["charge"],
        na_reference["sigma"],
        na_reference["epsilon"],
    )

    if nb_index != expected_index:

        raise RuntimeError(
            "Na NonbondedForce indexing mismatch."
        )


na_end_index = system.getNumParticles()


# ------------------------------------------------------------
# Cl-
# ------------------------------------------------------------

cl_start_index = system.getNumParticles()


for ion_index in range(
    N_CL
):

    expected_index = (
        cl_start_index
        + ion_index
    )

    global_index = system.addParticle(
        cl_reference["mass"]
    )

    if global_index != expected_index:

        raise RuntimeError(
            "Cl particle indexing mismatch."
        )

    nb_index = nb.addParticle(
        cl_reference["charge"],
        cl_reference["sigma"],
        cl_reference["epsilon"],
    )

    if nb_index != expected_index:

        raise RuntimeError(
            "Cl NonbondedForce indexing mismatch."
        )


cl_end_index = system.getNumParticles()


# Exact intended index ranges.
if na_start_index != 28080:

    raise RuntimeError(
        f"Unexpected Na start index: {na_start_index}."
    )


if na_end_index != 28098:

    raise RuntimeError(
        f"Unexpected Na end index: {na_end_index}."
    )


if cl_start_index != 28098:

    raise RuntimeError(
        f"Unexpected Cl start index: {cl_start_index}."
    )


if cl_end_index != 28116:

    raise RuntimeError(
        f"Unexpected Cl end index: {cl_end_index}."
    )


if system.getNumParticles() != (
    N_TOTAL_IONIZED_EXPECTED
):

    raise RuntimeError(
        "Ionized System particle count mismatch "
        "immediately after ion insertion."
    )


if nb.getNumParticles() != (
    N_TOTAL_IONIZED_EXPECTED
):

    raise RuntimeError(
        "Ionized NonbondedForce particle count mismatch "
        "immediately after ion insertion."
    )


print(
    f"Na+: particles "
    f"{na_start_index}-{na_end_index - 1}"
)

print(
    f"Cl-: particles "
    f"{cl_start_index}-{cl_end_index - 1}"
)


# Add internal water forces only if OPC reference had terms.
if water_bond_force is not None:

    system.addForce(
        water_bond_force
    )


if water_angle_force is not None:

    system.addForce(
        water_angle_force
    )


# ============================================================
# 22. VERIFY FINAL PARTICLE COUNTS
# ============================================================

print()
print("PARTICLE COUNTS")
print("---------------")

print(
    "Graphene:",
    N_GRAPHENE,
)

print(
    "Pyrene-PEG5:",
    N_LIGAND,
)

print(
    "OPC waters:",
    N_RETAINED_WATERS,
)

print(
    "OPC sites:",
    N_RETAINED_WATER_SITES,
)

print(
    "Na+ ions:",
    N_NA,
)

print(
    "Cl- ions:",
    N_CL,
)

print(
    "Total particles:",
    system.getNumParticles(),
)


if system.getNumParticles() != (
    N_TOTAL_IONIZED_EXPECTED
):

    raise RuntimeError(
        "Final System particle count mismatch. "
        f"Expected {N_TOTAL_IONIZED_EXPECTED}, "
        f"found {system.getNumParticles()}."
    )


if nb.getNumParticles() != (
    N_TOTAL_IONIZED_EXPECTED
):

    raise RuntimeError(
        "Final NonbondedForce particle count mismatch."
    )


# ============================================================
# 23. VERIFY VIRTUAL SITE COUNT
# ============================================================

n_virtual_sites = sum(

    1

    for i in range(
        system.getNumParticles()
    )

    if system.isVirtualSite(i)
)


print(
    "Virtual sites:",
    n_virtual_sites,
)


if n_virtual_sites != N_RETAINED_WATERS:

    raise RuntimeError(
        "Expected one OPC virtual site per retained water. "
        f"Expected {N_RETAINED_WATERS}, "
        f"found {n_virtual_sites}."
    )


# ============================================================
# 24. VERIFY CONSTRAINT COUNT
# ============================================================

expected_added_constraints = (
    N_RETAINED_WATERS
    * reference_system.getNumConstraints()
)


print(
    "Final constraints:",
    system.getNumConstraints(),
)

print(
    "Expected water constraints:",
    expected_added_constraints,
)


if system.getNumConstraints() != (
    expected_added_constraints
):

    raise RuntimeError(
        "Final constraint count mismatch."
    )


# ============================================================
# 25. CONFIGURE PERIODIC BOX
#
# Preserve x/y vectors from the solvent-box design.
# Replace z with the validated 12 nm slab cell.
# ============================================================

a_vec = openmm.Vec3(
    *final_box_vectors[0]
) * unit.nanometer

b_vec = openmm.Vec3(
    *final_box_vectors[1]
) * unit.nanometer

c_vec = openmm.Vec3(
    *final_box_vectors[2]
) * unit.nanometer


system.setDefaultPeriodicBoxVectors(
    a_vec,
    b_vec,
    c_vec,
)


print()
print("PERIODIC BOX")
print("------------")

print(
    "a:",
    final_box_vectors[0],
    "nm",
)

print(
    "b:",
    final_box_vectors[1],
    "nm",
)

print(
    "c:",
    final_box_vectors[2],
    "nm",
)


# ============================================================
# 26. CONFIGURE UNIFIED NONBONDED PHYSICS
#
# The old vacuum assembly used NoCutoff for validation.
#
# The complete solvated system is periodic, so convert the
# existing unified IFF + GAFF2 + OPC force to PME.
# ============================================================

nb.setNonbondedMethod(
    openmm.NonbondedForce.PME
)

nb.setCutoffDistance(
    NONBONDED_CUTOFF_NM
    * unit.nanometer
)

nb.setEwaldErrorTolerance(
    EWALD_ERROR_TOLERANCE
)


# A homogeneous-fluid LJ tail correction is not ideal for this
# highly inhomogeneous water/vacuum/graphene slab.
nb.setUseDispersionCorrection(
    False
)


nb.setUseSwitchingFunction(
    False
)


nb.setForceGroup(
    GROUP_NONBONDED
)


print()
print("NONBONDED SETTINGS")
print("------------------")

print(
    "Method: PME"
)

print(
    f"Cutoff: "
    f"{NONBONDED_CUTOFF_NM:.3f} nm"
)

print(
    f"Ewald tolerance: "
    f"{EWALD_ERROR_TOLERANCE:.2e}"
)

print(
    "LJ dispersion correction: OFF"
)


# ============================================================
# 27. GRAPHENE SUPPORT RESTRAINT
#
# Only graphene carbon cores:
#
#   0 ... 1249
#
# No Pyrene-PEG5 restraint.
# ============================================================

# Preserve the validated FF-3 supported-graphene Hamiltonian.
#
# The coordinates may contain thermal graphene fluctuations,
# but the support reference plane remains the experimentally
# defined/validated value used by FF-3.
graphene_z0_nm = 0.500000

graphene_z_spread_nm = float(
    np.max(
        solute_positions[
            :N_CARBON,
            2,
        ]
    )
    -
    np.min(
        solute_positions[
            :N_CARBON,
            2,
        ]
    )
)

support_force = openmm.CustomExternalForce(
    "0.5*kz*periodicdistance(0,0,z,0,0,z0)^2"
)

support_force.setName(
    "GrapheneSupportZRestraint_global_z0"
)

support_force.addGlobalParameter(
    "kz",
    K_GRAPHENE,
)

support_force.addGlobalParameter(
    "z0",
    graphene_z0_nm,
)

support_force.setForceGroup(
    GROUP_SUPPORT
)

for i in range(
    N_CARBON
):

    support_force.addParticle(
        i,
        [],
    )

system.addForce(
    support_force
)


# ============================================================
# 28. DYNAMIC YEH-BERKOWITZ SLAB CORRECTION
#
#   Mz = sum_i q_i z_i
#
#   Ucorr = 2*pi*k_e*Mz^2 / V
#
# Charges are read directly from the FINAL unified
# NonbondedForce.  Therefore graphene, Pyrene-PEG5, and OPC
# charged sites all participate automatically.
# ============================================================

if nb.getNumParticles() != system.getNumParticles():

    raise RuntimeError(
        "Unified NonbondedForce particle count does not match "
        "the assembled System."
    )


# Final slab-box volume from the authoritative production
# box matrix, not from the old 7 nm construction box.
final_box_volume_nm3 = float(
    abs(
        np.linalg.det(
            final_box_vectors
        )
    )
)


if (
    not np.isfinite(
        final_box_volume_nm3
    )
    or
    final_box_volume_nm3 <= 0.0
):

    raise RuntimeError(
        "Invalid final periodic box volume."
    )


yb_coefficient = (
    2.0
    * np.pi
    * K_E
    / final_box_volume_nm3
)


# Build the instantaneous Mz collective variable.
mz_force = openmm.CustomExternalForce(
    "q*z"
)

mz_force.addPerParticleParameter(
    "q"
)


charged_particle_count = 0
total_charge_e = 0.0


for i in range(
    nb.getNumParticles()
):

    charge, sigma, epsilon = (
        nb.getParticleParameters(
            i
        )
    )

    q_e = float(
        charge.value_in_unit(
            unit.elementary_charge
        )
    )

    total_charge_e += q_e

    # Zero-charge LJ sites do not contribute to Mz.
    if abs(q_e) > 1.0e-12:

        mz_force.addParticle(
            i,
            [
                q_e
            ],
        )

        charged_particle_count += 1


# This particular FF-3 system is intended to be neutral.
# YB for a charged slab would require additional care.
if abs(total_charge_e) > 1.0e-6:

    raise RuntimeError(
        "FF-3 system is not neutral: "
        f"total charge = {total_charge_e:.12e} e"
    )


yb_force = openmm.CustomCVForce(
    "yb_coeff*Mz^2"
)

yb_force.addCollectiveVariable(
    "Mz",
    mz_force,
)

yb_force.addGlobalParameter(
    "yb_coeff",
    yb_coefficient,
)

yb_force.setForceGroup(
    GROUP_YB
)

yb_force.setName(
    "Yeh-Berkowitz slab correction"
)

system.addForce(
    yb_force
)


print()
print("YEH-BERKOWITZ SLAB CORRECTION")
print("-----------------------------")

print(
    f"Final box volume: "
    f"{final_box_volume_nm3:.9f} nm^3"
)

print(
    f"YB coefficient: "
    f"{yb_coefficient:.12f}"
)

print(
    f"Total charge: "
    f"{total_charge_e:.12e} e"
)

print(
    "Charged particles in Mz CV:",
    charged_particle_count,
)


# ============================================================
# 29. WATER OXYGEN INDICES
#
# DIAGNOSTICS ONLY.
#
# No force, wall, restraint, or potential is applied to water.
# These indices are retained only for geometry checks.
# ============================================================

water_oxygen_indices = []

for water_index in range(
    N_RETAINED_WATERS
):

    oxygen_index = (
        N_SOLUTE
        + water_index
        * SITES_PER_WATER
        + OPC_O
    )

    water_oxygen_indices.append(
        oxygen_index
    )


na_particle_indices = np.arange(
    na_start_index,
    na_end_index,
    dtype=int,
)

cl_particle_indices = np.arange(
    cl_start_index,
    cl_end_index,
    dtype=int,
)


if len(na_particle_indices) != N_NA:

    raise RuntimeError(
        "Na particle-index count mismatch."
    )


if len(cl_particle_indices) != N_CL:

    raise RuntimeError(
        "Cl particle-index count mismatch."
    )


# ============================================================
# 29. TOTAL CHARGE
# ============================================================

total_charge = 0.0


for i in range(
    nb.getNumParticles()
):

    charge, _, _ = (
        nb.getParticleParameters(i)
    )

    total_charge += (
        charge.value_in_unit(
            unit.elementary_charge
        )
    )


print()
print("TOTAL SYSTEM CHARGE")
print("-------------------")

print(
    f"{total_charge:+.12f} e"
)


if abs(total_charge) > 1e-5:

    raise RuntimeError(
        "System should be neutral but total charge is "
        f"{total_charge:+.12f} e."
    )


# ============================================================
# 30. COMBINE STARTING POSITIONS
# ============================================================

retained_water_flat = (
    retained_water_positions.reshape(
        N_RETAINED_WATER_SITES,
        3,
    )
)


combined_positions = np.vstack(
    [
        solute_positions,
        retained_water_flat,
        na_positions,
        cl_positions,
    ]
)


if combined_positions.shape != (
    N_TOTAL_IONIZED_EXPECTED,
    3,
):

    raise RuntimeError(
        "Combined coordinate shape mismatch."
    )


positions_openmm = unit.Quantity(
    [
        openmm.Vec3(
            float(x),
            float(y),
            float(z),
        )

        for x, y, z
        in combined_positions
    ],

    unit.nanometer,
)


# ============================================================
# 31. CREATE OPENMM CONTEXT
#
# THIS DOES NOT RUN MD.
# ============================================================

print()
print("CREATING OPENMM CONTEXT")
print("-----------------------")


integrator = openmm.VerletIntegrator(
    0.001
    * unit.picoseconds
)


try:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CUDA"
        )
    )

except Exception:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CPU"
        )
    )


properties = {}


if platform.getName() == "CUDA":

    properties = {
        "Precision": "mixed",
    }


context = openmm.Context(
    system,
    integrator,
    platform,
    properties,
)


context.setPositions(
    positions_openmm
)


print(
    "Platform:",
    platform.getName(),
)


# ============================================================
# 32. COMPUTE OPC VIRTUAL SITES
#
# OpenMM recomputes every M site directly from O/H/H.
# ============================================================

context.computeVirtualSites()


virtual_state = context.getState(
    getPositions=True
)


virtual_positions = (
    virtual_state
    .getPositions(
        asNumpy=True
    )
    .value_in_unit(
        unit.nanometer
    )
)


m_indices = np.asarray(
    [
        N_SOLUTE
        + i * 4
        + OPC_M

        for i in range(
            N_RETAINED_WATERS
        )
    ],

    dtype=int,
)


saved_m_positions = (
    combined_positions[
        m_indices
    ]
)


computed_m_positions = (
    virtual_positions[
        m_indices
    ]
)


m_site_error = np.linalg.norm(

    computed_m_positions
    - saved_m_positions,

    axis=1,
)


max_m_site_error = float(
    np.max(
        m_site_error
    )
)


print()
print("OPC VIRTUAL-SITE CHECK")
print("----------------------")

print(
    f"Maximum saved-vs-computed M-site difference: "
    f"{max_m_site_error:.10f} nm"
)


# 1e-4 nm = 0.001 Angstrom.
if max_m_site_error > 1e-4:

    raise RuntimeError(
        "Saved OPC M-site positions do not agree with "
        "OpenMM OPC virtual-site geometry."
    )


print(
    "OPC virtual-site geometry: PASS"
)


# ============================================================
# 33. APPLY RIGID-WATER CONSTRAINTS
#
# This is NOT minimization.
#
# It only projects the starting water geometry exactly onto
# the OPC rigid geometry.
# ============================================================

positions_before_constraints = (
    virtual_positions.copy()
)


context.applyConstraints(
    1e-8
)

context.computeVirtualSites()


constrained_state = context.getState(
    getPositions=True
)


constrained_positions = (
    constrained_state
    .getPositions(
        asNumpy=True
    )
    .value_in_unit(
        unit.nanometer
    )
)


constraint_shift = np.linalg.norm(

    constrained_positions
    - positions_before_constraints,

    axis=1,
)


# Ignore massless virtual sites when assessing actual atomic
# coordinate displacement.
massive_mask = np.asarray(
    [
        system.getParticleMass(i)
        .value_in_unit(
            unit.dalton
        ) > 0.0

        for i in range(
            system.getNumParticles()
        )
    ]
)


max_constraint_shift = float(
    np.max(
        constraint_shift[
            massive_mask
        ]
    )
)


print()
print("RIGID-WATER CONSTRAINT CHECK")
print("----------------------------")

print(
    f"Maximum massive-particle adjustment: "
    f"{max_constraint_shift:.10f} nm"
)


if max_constraint_shift > 5e-4:

    raise RuntimeError(
        "Applying OPC constraints required an unexpectedly "
        "large coordinate change."
    )


print(
    "Rigid-water starting geometry: PASS"
)


# ============================================================
# 34. VERIFY WATER OXYGENS REMAIN INSIDE SLAB
# ============================================================

water_oxygen_indices_np = np.asarray(
    water_oxygen_indices,
    dtype=int,
)


oxygen_z = (
    constrained_positions[
        water_oxygen_indices_np,
        2,
    ]
)


print()
print("CONSTRAINED WATER OXYGEN RANGE")
print("------------------------------")

print(
    f"Minimum z: "
    f"{oxygen_z.min():.6f} nm"
)

print(
    f"Maximum z: "
    f"{oxygen_z.max():.6f} nm"
)


if oxygen_z.min() < (
    LOWER_WATER_BOUNDARY_NM
):

    raise RuntimeError(
        "A retained water oxygen lies below the "
        "validated lower aqueous boundary after "
        "constraint projection."
    )


if oxygen_z.max() > (
    UPPER_WATER_BOUNDARY_NM
):

    raise RuntimeError(
        "A retained water oxygen lies above the "
        "validated upper aqueous boundary after "
        "constraint projection."
    )


# ============================================================
# 34A. VERIFY IONS REMAIN INSIDE AQUEOUS SLAB
#
# Diagnostics only.  No wall, restraint, or external potential
# is applied to Na+ or Cl-.
# ============================================================

na_z = (
    constrained_positions[
        na_particle_indices,
        2,
    ]
)

cl_z = (
    constrained_positions[
        cl_particle_indices,
        2,
    ]
)


print()
print("CONSTRAINED ION Z RANGES")
print("------------------------")

print(
    f"Na+ minimum z: {na_z.min():.6f} nm"
)

print(
    f"Na+ maximum z: {na_z.max():.6f} nm"
)

print(
    f"Cl- minimum z: {cl_z.min():.6f} nm"
)

print(
    f"Cl- maximum z: {cl_z.max():.6f} nm"
)


if (
    na_z.min() < LOWER_WATER_BOUNDARY_NM
    or
    cl_z.min() < LOWER_WATER_BOUNDARY_NM
):

    raise RuntimeError(
        "A Na+/Cl- ion lies below the validated "
        "lower aqueous boundary."
    )


if (
    na_z.max() > UPPER_WATER_BOUNDARY_NM
    or
    cl_z.max() > UPPER_WATER_BOUNDARY_NM
):

    raise RuntimeError(
        "A Na+/Cl- ion lies above the validated "
        "upper aqueous boundary."
    )


# ============================================================
# 34B. VERIFY FINAL ION SEPARATIONS
#
# Recompute minimum-image distances from the actual constrained
# starting coordinates that will be evaluated by OpenMM.
# ============================================================

ion_particle_indices = np.concatenate(
    [
        na_particle_indices,
        cl_particle_indices,
    ]
)

ion_positions_constrained = (
    constrained_positions[
        ion_particle_indices
    ]
)

solute_positions_constrained = (
    constrained_positions[
        :N_SOLUTE
    ]
)


minimum_ion_solute_distance_nm = float(
    "inf"
)

for ion_position in ion_positions_constrained:

    displacement = (
        solute_positions_constrained
        - ion_position
    )

    displacement_mic = (
        minimum_image_displacement(
            displacement
        )
    )

    minimum_ion_solute_distance_nm = min(
        minimum_ion_solute_distance_nm,
        float(
            np.min(
                np.linalg.norm(
                    displacement_mic,
                    axis=1,
                )
            )
        ),
    )


minimum_ion_ion_distance_nm = float(
    "inf"
)

for ion_index in range(
    len(ion_positions_constrained) - 1
):

    displacement = (
        ion_positions_constrained[
            ion_index + 1:
        ]
        - ion_positions_constrained[
            ion_index
        ]
    )

    displacement_mic = (
        minimum_image_displacement(
            displacement
        )
    )

    minimum_ion_ion_distance_nm = min(
        minimum_ion_ion_distance_nm,
        float(
            np.min(
                np.linalg.norm(
                    displacement_mic,
                    axis=1,
                )
            )
        ),
    )


print()
print("FINAL ION SEPARATION AUDIT")
print("--------------------------")

print(
    "Minimum ion-solute distance: "
    f"{minimum_ion_solute_distance_nm:.6f} nm"
)

print(
    "Minimum ion-ion distance: "
    f"{minimum_ion_ion_distance_nm:.6f} nm"
)


if (
    minimum_ion_solute_distance_nm
    < MIN_SOLUTE_DIST_NM
):

    raise RuntimeError(
        "Final ion-solute distance violates "
        "the locked minimum separation."
    )


if (
    minimum_ion_ion_distance_nm
    < MIN_ION_DIST_NM
):

    raise RuntimeError(
        "Final ion-ion distance violates "
        "the locked minimum separation."
    )


# ============================================================
# 35. ENERGY-ONLY SANITY CHECK
#
# FIRST TIME all molecular interactions are evaluated together.
#
# Still NO MD.
# ============================================================

print()
print("=" * 72)
print("FULL-SYSTEM ENERGY CHECK")
print("=" * 72)


state = context.getState(
    getEnergy=True,
    getForces=True,
)


total_energy = (
    state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


forces = (
    state
    .getForces(
        asNumpy=True
    )
    .value_in_unit(
        unit.kilojoule_per_mole
        / unit.nanometer
    )
)


if not np.isfinite(
    total_energy
):

    raise RuntimeError(
        "Total potential energy is NaN/Inf."
    )


if not np.isfinite(
    forces
).all():

    raise RuntimeError(
        "Forces contain NaN/Inf."
    )


print()
print(
    f"Total potential energy: "
    f"{total_energy:.6f} kJ/mol"
)


# ============================================================
# 36. ENERGY BY FORCE GROUP
# ============================================================

group_labels = {

    GROUP_GRAPHENE_BONDS:
        "Graphene bonds",

    GROUP_GRAPHENE_ANGLES:
        "Graphene angles",

    GROUP_GRAPHENE_OOP:
        "Graphene out-of-plane",

    GROUP_LIGAND_BONDS:
        "Pyrene-PEG5 bonds",

    GROUP_LIGAND_ANGLES:
        "Pyrene-PEG5 angles",

    GROUP_SUPPORT:
        "Graphene support",


    GROUP_LIGAND_TORSIONS:
        "Pyrene-PEG5 torsions",

    GROUP_NONBONDED:
        "Unified nonbonded",

    GROUP_WATER_BONDS:
        "OPC bond terms",

    GROUP_WATER_ANGLES:
        "OPC angle terms",

    GROUP_YB:
        "Yeh-Berkowitz slab correction",
}


group_energies = {}


print()
print("ENERGY BY FORCE GROUP")
print("---------------------")


for group, label in sorted(
    group_labels.items()
):

    group_state = context.getState(
        getEnergy=True,

        groups=(
            1
            << group
        ),
    )

    energy = (
        group_state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if not np.isfinite(
        energy
    ):

        raise RuntimeError(
            f"Nonfinite energy in group: {label}"
        )

    group_energies[
        label
    ] = float(
        energy
    )

    print(
        f"{label:<28s} "
        f"{energy:>18.6f} kJ/mol"
    )


# ============================================================
# 37. FORCE MAGNITUDES
# ============================================================

force_magnitudes = np.linalg.norm(
    forces,
    axis=1,
)


# Ignore massless virtual sites for primary max-force metric.
massive_force_magnitudes = (
    force_magnitudes.copy()
)

massive_force_magnitudes[
    ~massive_mask
] = -1.0


max_force_index = int(
    np.argmax(
        massive_force_magnitudes
    )
)


max_force = float(
    massive_force_magnitudes[
        max_force_index
    ]
)


# ============================================================
# 38. PARTICLE DESCRIPTION HELPER
# ============================================================

def describe_particle(index):

    if index < N_CARBON:

        return (
            f"graphene carbon core {index}"
        )

    if index < (
        2 * N_CARBON
    ):

        return (
            "graphene upper pi site "
            f"{index - N_CARBON}"
        )

    if index < N_GRAPHENE:

        return (
            "graphene lower pi site "
            f"{index - 2 * N_CARBON}"
        )

    if index < N_SOLUTE:

        return (
            "Pyrene-PEG5 atom "
            f"{index - N_GRAPHENE}"
        )

    if index < water_end_index:

        local = (
            index
            - N_SOLUTE
        )

        water_index = (
            local
            // SITES_PER_WATER
        )

        site_index = (
            local
            % SITES_PER_WATER
        )

        site_names = {
            0: "O",
            1: "H1",
            2: "H2",
            3: "M",
        }

        source_water_index = int(
            retained_water_source_indices[
                water_index
            ]
        )

        return (
            f"OPC retained water {water_index} "
            f"(source water {source_water_index}) "
            f"site {site_names[site_index]}"
        )

    if index < na_end_index:

        return (
            "Na+ ion "
            f"{index - na_start_index}"
        )

    if index < cl_end_index:

        return (
            "Cl- ion "
            f"{index - cl_start_index}"
        )

    return (
        f"unknown particle {index}"
    )


print()
print("MAXIMUM STARTING FORCE")
print("----------------------")

print(
    f"Particle: {max_force_index}"
)

print(
    "Identity:",
    describe_particle(
        max_force_index
    ),
)

print(
    f"Magnitude: "
    f"{max_force:.6f} "
    f"kJ/(mol nm)"
)


# ============================================================
# 39. MAX FORCE BY SYSTEM COMPONENT
# ============================================================

graphene_force_max = float(
    np.max(
        force_magnitudes[
            :N_GRAPHENE
        ]
    )
)


ligand_force_max = float(
    np.max(
        force_magnitudes[
            N_GRAPHENE:
            N_SOLUTE
        ]
    )
)


water_massive_indices = np.where(
    massive_mask[
        N_SOLUTE:
        water_end_index
    ]
)[0] + N_SOLUTE


water_force_max = float(
    np.max(
        force_magnitudes[
            water_massive_indices
        ]
    )
)


na_force_max = float(
    np.max(
        force_magnitudes[
            na_particle_indices
        ]
    )
)


cl_force_max = float(
    np.max(
        force_magnitudes[
            cl_particle_indices
        ]
    )
)


print()
print("MAX FORCE BY COMPONENT")
print("----------------------")

print(
    f"Graphene:      "
    f"{graphene_force_max:.6f} kJ/(mol nm)"
)

print(
    f"Pyrene-PEG5:   "
    f"{ligand_force_max:.6f} kJ/(mol nm)"
)

print(
    f"OPC water:     "
    f"{water_force_max:.6f} kJ/(mol nm)"
)

print(
    f"Na+:           "
    f"{na_force_max:.6f} kJ/(mol nm)"
)

print(
    f"Cl-:           "
    f"{cl_force_max:.6f} kJ/(mol nm)"
)


# ============================================================
# 40. TOP 10 MASSIVE-PARTICLE FORCES
# ============================================================

valid_indices = np.where(
    massive_mask
)[0]


sorted_force_indices = (
    valid_indices[
        np.argsort(
            force_magnitudes[
                valid_indices
            ]
        )[::-1]
    ]
)


top_indices = (
    sorted_force_indices[
        :10
    ]
)


print()
print("TOP 10 STARTING FORCES")
print("----------------------")


top_force_records = []


for rank, index in enumerate(
    top_indices,
    start=1,
):

    magnitude = float(
        force_magnitudes[
            index
        ]
    )

    description = (
        describe_particle(
            int(index)
        )
    )

    print(
        f"{rank:2d}. "
        f"particle {int(index):5d} | "
        f"{magnitude:14.6f} | "
        f"{description}"
    )

    top_force_records.append(
        {
            "rank": rank,
            "particle": int(index),
            "force_kj_mol_nm": magnitude,
            "description": description,
        }
    )


# ============================================================
# 41. STARTING-CONDITION CLASSIFICATION
#
# This is deliberately conservative.
#
# We do NOT automatically start MD regardless of result.
# ============================================================

if max_force > 1.0e7:

    force_status = (
        "CATASTROPHIC_CLASH"
    )


elif max_force > 1.0e6:

    force_status = (
        "VERY_LARGE_FORCE"
    )


elif max_force > 1.0e5:

    force_status = (
        "LARGE_FORCE_MINIMIZATION_REQUIRED"
    )


else:

    force_status = (
        "READY_FOR_CAREFUL_MINIMIZATION"
    )


print()
print("STARTING CONDITION")
print("------------------")

print(
    force_status
)


# ============================================================
# 42. SAVE CONSTRAINED STARTING POSITIONS
# ============================================================

OUTPUT_POSITIONS.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_POSITIONS,
    constrained_positions,
)


# ============================================================
# 43. SAVE COMPLETE OPENMM SYSTEM
# ============================================================

with open(
    OUTPUT_SYSTEM_XML,
    "w",
) as f:

    f.write(
        openmm.XmlSerializer.serialize(
            system
        )
    )


# ============================================================
# 44. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "graphene_particles":
        N_GRAPHENE,

    "pyrene_peg5_particles":
        N_LIGAND,

    "electrolyte_model":
        "PBS ionic-strength surrogate: NaCl in OPC water",

    "source_opc_waters":
        N_WATERS,

    "replaced_opc_waters":
        N_REPLACED_WATERS,

    "opc_waters":
        N_RETAINED_WATERS,

    "opc_sites_per_water":
        SITES_PER_WATER,

    "opc_total_sites":
        N_RETAINED_WATER_SITES,

    "na_ions":
        N_NA,

    "cl_ions":
        N_CL,

    "total_ions":
        N_IONS,

    "ion_selection_seed":
        ION_SELECTION_SEED,

    "minimum_selection_ion_solute_distance_nm":
        MIN_SOLUTE_DIST_NM,

    "minimum_selection_ion_ion_distance_nm":
        MIN_ION_DIST_NM,

    "na_replacement_water_indices":
        na_replacement_water_indices,

    "cl_replacement_water_indices":
        cl_replacement_water_indices,

    "na_particle_start_index":
        na_start_index,

    "na_particle_end_index_exclusive":
        na_end_index,

    "cl_particle_start_index":
        cl_start_index,

    "cl_particle_end_index_exclusive":
        cl_end_index,

    "final_minimum_ion_solute_distance_nm":
        minimum_ion_solute_distance_nm,

    "final_minimum_ion_ion_distance_nm":
        minimum_ion_ion_distance_nm,

    "total_particles":
        system.getNumParticles(),

    "virtual_sites":
        n_virtual_sites,

    "constraints":
        system.getNumConstraints(),

    "base_nonbonded_exceptions":
        base_exception_count,

    "final_nonbonded_exceptions":
        nb.getNumExceptions(),

    "nonbonded_method":
        "PME",

    "nonbonded_cutoff_nm":
        NONBONDED_CUTOFF_NM,

    "ewald_error_tolerance":
        EWALD_ERROR_TOLERANCE,

    "dispersion_correction":
        False,

    "box_z_nm":
        float(final_box_vectors[2, 2]),

    "final_box_volume_nm3":
        final_box_volume_nm3,

    "yeh_berkowitz_enabled":
        True,

    "yeh_berkowitz_force_group":
        GROUP_YB,

    "yeh_berkowitz_coefficient":
        yb_coefficient,

    "total_charge_e":
        total_charge_e,

    "yb_charged_particles":
        charged_particle_count,

    "lower_water_boundary_nm":
        LOWER_WATER_BOUNDARY_NM,

    "upper_water_boundary_nm":
        UPPER_WATER_BOUNDARY_NM,

    "graphene_support_k_kj_mol_nm2":
        K_GRAPHENE,


    "maximum_opc_m_site_error_nm":
        max_m_site_error,

    "maximum_constraint_adjustment_nm":
        max_constraint_shift,

    "total_potential_energy_kj_mol":
        float(total_energy),

    "energy_by_force_group_kj_mol":
        group_energies,

    "max_force_kj_mol_nm":
        max_force,

    "max_force_particle":
        max_force_index,

    "max_force_identity":
        describe_particle(
            max_force_index
        ),

    "max_graphene_force_kj_mol_nm":
        graphene_force_max,

    "max_ligand_force_kj_mol_nm":
        ligand_force_max,

    "max_water_force_kj_mol_nm":
        water_force_max,

    "max_na_force_kj_mol_nm":
        na_force_max,

    "max_cl_force_kj_mol_nm":
        cl_force_max,

    "top_10_forces":
        top_force_records,

    "starting_condition":
        force_status,

    "ions_added":
        True,

    "pyrene_peg5_restrained":
        False,

    "md_run":
        False,

    "minimization_run":
        False,

    "status":
        "ENERGY_CHECK_COMPLETE",
}


with open(
    OUTPUT_METADATA,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# 45. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "System:",
    OUTPUT_SYSTEM_XML,
)

print(
    "Positions:",
    OUTPUT_POSITIONS,
)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("FULL SUPPORTED SOLVATED ENERGY CHECK: COMPLETE")
print("=" * 72)

print()
print(
    "No MD has been run."
)

print(
    "No minimization has been run."
)

print(
    f"PBS surrogate ions present: "
    f"{N_NA} Na+ and {N_CL} Cl-."
)

print(
    "Pyrene-PEG5 remains completely unrestrained."
)

print()
print(
    "Next decision depends on the maximum starting force."
)



