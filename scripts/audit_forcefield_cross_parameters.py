from pathlib import Path
import csv
import json
import math

import numpy as np
import openmm
from openmm import unit


# ============================================================
# PURPOSE
# ============================================================
#
# Audit the ACTUAL nonbonded parameters in the production XML.
#
# Checks:
#
#   - graphene cg1 cores
#   - graphene cge virtual-pi particles
#   - OPC O/H/H/M sites
#   - ligand charge and pyrene-subset charge
#   - Lorentz-Berthelot graphene-water cross parameters
#   - graphene-water / graphene-ligand cross exceptions
#   - uniformity of all graphene and water parameters
#
# No simulation.
# No coordinates changed.
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

SYSTEM_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_supported_pme_production_safe.xml"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "forcefield_cross_parameter_audit.json"
)

LIGAND_CSV = (
    ROOT
    / "analysis"
    / "ligand_nonbonded_parameters.csv"
)


EXPECTED_PARTICLES = 28224
EXPECTED_WATERS = 6101

GRAPHENE_C = range(
    0,
    1250,
)

GRAPHENE_PI = range(
    1250,
    3750,
)

LIGAND = range(
    3750,
    3820,
)

PYRENE = range(
    3752,
    3768,
)

WATER_START = 3820


# ============================================================
# HELPERS
# ============================================================

def q_e(q):
    return float(
        q.value_in_unit(
            unit.elementary_charge
        )
    )


def sigma_nm(sigma):
    return float(
        sigma.value_in_unit(
            unit.nanometer
        )
    )


def eps_kj(epsilon):
    return float(
        epsilon.value_in_unit(
            unit.kilojoule_per_mole
        )
    )


def mass_amu(mass):
    return float(
        mass.value_in_unit(
            unit.dalton
        )
    )


def particle_parameters(nb, index):

    q, sigma, epsilon = (
        nb.getParticleParameters(
            index
        )
    )

    return {
        "q_e": q_e(q),
        "sigma_nm": sigma_nm(sigma),
        "epsilon_kJ_mol": eps_kj(epsilon),
    }


def uniform(values, atol=1.0e-10):

    values = np.asarray(
        values,
        dtype=float,
    )

    return bool(
        np.allclose(
            values,
            values[0],
            rtol=0.0,
            atol=atol,
        )
    )


def group_charge(nb, indices):

    total = 0.0

    for i in indices:
        q, sigma, epsilon = (
            nb.getParticleParameters(i)
        )

        total += q_e(q)

    return total


# ============================================================
# LOAD SYSTEM
# ============================================================

print()
print("=" * 78)
print("FORCE-FIELD CROSS-PARAMETER AUDIT")
print("=" * 78)
print()

if not SYSTEM_XML.exists():
    raise FileNotFoundError(
        SYSTEM_XML
    )


system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text(
        encoding="utf-8"
    )
)


n_particles = (
    system.getNumParticles()
)


print(
    f"OpenMM version: "
    f"{openmm.version.full_version}"
)

print(
    f"Particles:      "
    f"{n_particles}"
)


if n_particles != EXPECTED_PARTICLES:
    raise RuntimeError(
        f"Expected {EXPECTED_PARTICLES} particles, "
        f"found {n_particles}"
    )


# ============================================================
# FIND NONBONDED FORCE
# ============================================================

nb_forces = []

for force_index in range(
    system.getNumForces()
):

    force = system.getForce(
        force_index
    )

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):

        nb_forces.append(
            (
                force_index,
                force,
            )
        )


if len(nb_forces) != 1:
    raise RuntimeError(
        "Expected exactly one NonbondedForce."
    )


nb_index, nb = nb_forces[0]


print(
    f"NonbondedForce index: "
    f"{nb_index}"
)

print(
    f"Exceptions:           "
    f"{nb.getNumExceptions()}"
)


# ============================================================
# GRAPHENE UNIFORMITY
# ============================================================

core_q = []
core_sigma = []
core_eps = []
core_mass = []

for i in GRAPHENE_C:

    p = particle_parameters(
        nb,
        i,
    )

    core_q.append(
        p["q_e"]
    )

    core_sigma.append(
        p["sigma_nm"]
    )

    core_eps.append(
        p["epsilon_kJ_mol"]
    )

    core_mass.append(
        mass_amu(
            system.getParticleMass(i)
        )
    )


pi_q = []
pi_sigma = []
pi_eps = []
pi_mass = []

for i in GRAPHENE_PI:

    p = particle_parameters(
        nb,
        i,
    )

    pi_q.append(
        p["q_e"]
    )

    pi_sigma.append(
        p["sigma_nm"]
    )

    pi_eps.append(
        p["epsilon_kJ_mol"]
    )

    pi_mass.append(
        mass_amu(
            system.getParticleMass(i)
        )
    )


if not uniform(core_q):
    raise RuntimeError(
        "Graphene core charges are not uniform."
    )

if not uniform(core_sigma):
    raise RuntimeError(
        "Graphene core sigma values are not uniform."
    )

if not uniform(core_eps):
    raise RuntimeError(
        "Graphene core epsilon values are not uniform."
    )

if not uniform(core_mass):
    raise RuntimeError(
        "Graphene core masses are not uniform."
    )


if not uniform(pi_q):
    raise RuntimeError(
        "Graphene pi charges are not uniform."
    )

if not uniform(pi_eps):
    raise RuntimeError(
        "Graphene pi epsilon values are not uniform."
    )

if not uniform(pi_mass):
    raise RuntimeError(
        "Graphene pi masses are not uniform."
    )


cg1 = {
    "q_e": core_q[0],
    "sigma_nm": core_sigma[0],
    "epsilon_kJ_mol": core_eps[0],
    "mass_amu": core_mass[0],
}

cge = {
    "q_e": pi_q[0],
    "sigma_nm": pi_sigma[0],
    "epsilon_kJ_mol": pi_eps[0],
    "mass_amu": pi_mass[0],
}


# ============================================================
# FIRST OPC WATER: AUTO-IDENTIFY O/H/H/M
# ============================================================

first_water = list(
    range(
        WATER_START,
        WATER_START + 4,
    )
)


water_sites = []

for i in first_water:

    p = particle_parameters(
        nb,
        i,
    )

    water_sites.append(
        {
            "index": i,
            "mass_amu": mass_amu(
                system.getParticleMass(i)
            ),
            "virtual_site": bool(
                system.isVirtualSite(i)
            ),
            **p,
        }
    )


virtual = [
    p
    for p in water_sites
    if p["virtual_site"]
]

massive = [
    p
    for p in water_sites
    if not p["virtual_site"]
]


if len(virtual) != 1:
    raise RuntimeError(
        "Expected one OPC virtual site."
    )


M = virtual[0]


O_candidates = sorted(
    massive,
    key=lambda p: p["mass_amu"],
    reverse=True,
)


if len(O_candidates) != 3:
    raise RuntimeError(
        "Expected OPC O + H + H massive sites."
    )


O = O_candidates[0]

H1 = O_candidates[1]
H2 = O_candidates[2]


# ============================================================
# VERIFY EVERY WATER REPEATS SAME PARAMETER PATTERN
# ============================================================

number_water_particles = (
    EXPECTED_PARTICLES
    -
    WATER_START
)


if (
    number_water_particles
    !=
    EXPECTED_WATERS * 4
):

    raise RuntimeError(
        "Water particle-count mismatch."
    )


for water_number in range(
    EXPECTED_WATERS
):

    base = (
        WATER_START
        +
        4 * water_number
    )

    indices = [
        base,
        base + 1,
        base + 2,
        base + 3,
    ]


    for offset, reference_index in enumerate(
        first_water
    ):

        i = indices[offset]

        p = particle_parameters(
            nb,
            i,
        )

        ref = particle_parameters(
            nb,
            reference_index,
        )


        if abs(
            p["q_e"]
            -
            ref["q_e"]
        ) > 1.0e-10:

            raise RuntimeError(
                f"Water charge mismatch at particle {i}"
            )


        if abs(
            p["sigma_nm"]
            -
            ref["sigma_nm"]
        ) > 1.0e-10:

            raise RuntimeError(
                f"Water sigma mismatch at particle {i}"
            )


        if abs(
            p["epsilon_kJ_mol"]
            -
            ref["epsilon_kJ_mol"]
        ) > 1.0e-10:

            raise RuntimeError(
                f"Water epsilon mismatch at particle {i}"
            )


# ============================================================
# GROUP CHARGES
# ============================================================

graphene_core_charge = group_charge(
    nb,
    GRAPHENE_C,
)

graphene_pi_charge = group_charge(
    nb,
    GRAPHENE_PI,
)

graphene_total_charge = (
    graphene_core_charge
    +
    graphene_pi_charge
)

ligand_charge = group_charge(
    nb,
    LIGAND,
)

pyrene_charge = group_charge(
    nb,
    PYRENE,
)


water_charge = sum(
    site["q_e"]
    for site in water_sites
)


total_charge = group_charge(
    nb,
    range(n_particles),
)


# ============================================================
# LORENTZ-BERTHELOT CROSS INTERACTIONS
#
# OpenMM NonbondedForce:
#
# sigma_ij   = (sigma_i + sigma_j)/2
# epsilon_ij = sqrt(epsilon_i*epsilon_j)
# ============================================================

cg1_O_sigma_nm = (
    cg1["sigma_nm"]
    +
    O["sigma_nm"]
) / 2.0


cg1_O_epsilon_kj = math.sqrt(
    cg1["epsilon_kJ_mol"]
    *
    O["epsilon_kJ_mol"]
)


cge_O_epsilon_kj = math.sqrt(
    max(
        cge["epsilon_kJ_mol"],
        0.0,
    )
    *
    max(
        O["epsilon_kJ_mol"],
        0.0,
    )
)


# ============================================================
# EXPECTED REFERENCE VALUES
#
# Used only as transcription diagnostics.
# ============================================================

IFF_RMIN_NM = 0.379

IFF_EXPECTED_SIGMA_NM = (
    IFF_RMIN_NM
    /
    (
        2.0 ** (1.0 / 6.0)
    )
)

IFF_EXPECTED_EPS_KJ = (
    0.063
    *
    4.184
)


OPC_EXPECTED_SIGMA_NM = (
    3.16655
    *
    0.1
)

OPC_EXPECTED_EPS_KJ = (
    0.89036
)

OPC_EXPECTED_H_CHARGE = (
    0.6791
)

OPC_EXPECTED_M_CHARGE = (
    -2.0
    *
    OPC_EXPECTED_H_CHARGE
)


# ============================================================
# CROSS EXCEPTIONS
# ============================================================

graphene_set = set(
    range(
        0,
        3750,
    )
)

ligand_set = set(
    range(
        3750,
        3820,
    )
)

water_set = set(
    range(
        WATER_START,
        EXPECTED_PARTICLES,
    )
)


graphene_water_exceptions = 0
graphene_ligand_exceptions = 0
ligand_water_exceptions = 0


for exception_index in range(
    nb.getNumExceptions()
):

    (
        p1,
        p2,
        charge_prod,
        sigma,
        epsilon,
    ) = nb.getExceptionParameters(
        exception_index
    )

    p1 = int(p1)
    p2 = int(p2)


    if (
        (
            p1 in graphene_set
            and
            p2 in water_set
        )
        or
        (
            p2 in graphene_set
            and
            p1 in water_set
        )
    ):
        graphene_water_exceptions += 1


    if (
        (
            p1 in graphene_set
            and
            p2 in ligand_set
        )
        or
        (
            p2 in graphene_set
            and
            p1 in ligand_set
        )
    ):
        graphene_ligand_exceptions += 1


    if (
        (
            p1 in ligand_set
            and
            p2 in water_set
        )
        or
        (
            p2 in ligand_set
            and
            p1 in water_set
        )
    ):
        ligand_water_exceptions += 1


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 78)
print("GRAPHENE PARAMETERS")
print("=" * 78)

print()
print("cg1 carbon core:")

print(
    f"  mass       = "
    f"{cg1['mass_amu']:.8f} amu"
)

print(
    f"  charge     = "
    f"{cg1['q_e']:+.8f} e"
)

print(
    f"  sigma      = "
    f"{cg1['sigma_nm']:.8f} nm"
)

print(
    f"  epsilon    = "
    f"{cg1['epsilon_kJ_mol']:.8f} kJ/mol"
)


print()
print("cge virtual pi:")

print(
    f"  mass       = "
    f"{cge['mass_amu']:.8f} amu"
)

print(
    f"  charge     = "
    f"{cge['q_e']:+.8f} e"
)

print(
    f"  sigma      = "
    f"{cge['sigma_nm']:.8f} nm"
)

print(
    f"  epsilon    = "
    f"{cge['epsilon_kJ_mol']:.8f} kJ/mol"
)


print()
print("IFF transcription reference:")

print(
    f"  expected OpenMM cg1 sigma "
    f"from Rmin=3.79 A: "
    f"{IFF_EXPECTED_SIGMA_NM:.8f} nm"
)

print(
    f"  expected cg1 epsilon "
    f"from 0.063 kcal/mol: "
    f"{IFF_EXPECTED_EPS_KJ:.8f} kJ/mol"
)


print()
print("=" * 78)
print("OPC PARAMETERS")
print("=" * 78)


for label, site in [
    ("O", O),
    ("H1", H1),
    ("H2", H2),
    ("M", M),
]:

    print()
    print(
        f"{label} index {site['index']}:"
    )

    print(
        f"  mass       = "
        f"{site['mass_amu']:.8f} amu"
    )

    print(
        f"  virtual    = "
        f"{site['virtual_site']}"
    )

    print(
        f"  charge     = "
        f"{site['q_e']:+.8f} e"
    )

    print(
        f"  sigma      = "
        f"{site['sigma_nm']:.8f} nm"
    )

    print(
        f"  epsilon    = "
        f"{site['epsilon_kJ_mol']:.8f} kJ/mol"
    )


print()
print("OPC published reference:")

print(
    f"  O sigma    = "
    f"{OPC_EXPECTED_SIGMA_NM:.8f} nm"
)

print(
    f"  O epsilon  = "
    f"{OPC_EXPECTED_EPS_KJ:.8f} kJ/mol"
)

print(
    f"  H charge   = "
    f"{OPC_EXPECTED_H_CHARGE:+.8f} e"
)

print(
    f"  M charge   = "
    f"{OPC_EXPECTED_M_CHARGE:+.8f} e"
)


print()
print("=" * 78)
print("GROUP CHARGES")
print("=" * 78)

print(
    f"Graphene cores: "
    f"{graphene_core_charge:+.8f} e"
)

print(
    f"Graphene pi:    "
    f"{graphene_pi_charge:+.8f} e"
)

print(
    f"Graphene total: "
    f"{graphene_total_charge:+.8f} e"
)

print(
    f"Ligand total:   "
    f"{ligand_charge:+.8f} e"
)

print(
    f"Pyrene subset:  "
    f"{pyrene_charge:+.8f} e"
)

print(
    f"One OPC water:  "
    f"{water_charge:+.8f} e"
)

print(
    f"Whole system:   "
    f"{total_charge:+.10e} e"
)


print()
print("=" * 78)
print("GRAPHENE-WATER CROSS TERMS")
print("=" * 78)

print(
    f"cg1-O sigma:     "
    f"{cg1_O_sigma_nm:.8f} nm"
)

print(
    f"cg1-O epsilon:   "
    f"{cg1_O_epsilon_kj:.8f} kJ/mol"
)

print(
    f"cge-O epsilon:   "
    f"{cge_O_epsilon_kj:.8f} kJ/mol"
)

print()

print(
    f"cg1 x H charge product: "
    f"{cg1['q_e'] * H1['q_e']:+.8f} e^2"
)

print(
    f"cg1 x M charge product: "
    f"{cg1['q_e'] * M['q_e']:+.8f} e^2"
)

print(
    f"cge x H charge product: "
    f"{cge['q_e'] * H1['q_e']:+.8f} e^2"
)

print(
    f"cge x M charge product: "
    f"{cge['q_e'] * M['q_e']:+.8f} e^2"
)


print()
print("=" * 78)
print("CROSS EXCEPTIONS")
print("=" * 78)

print(
    f"Graphene-water: "
    f"{graphene_water_exceptions}"
)

print(
    f"Graphene-ligand: "
    f"{graphene_ligand_exceptions}"
)

print(
    f"Ligand-water:   "
    f"{ligand_water_exceptions}"
)


# ============================================================
# SANITY CONDITIONS
# ============================================================

if graphene_water_exceptions != 0:
    raise RuntimeError(
        "Unexpected graphene-water exceptions detected."
    )

if graphene_ligand_exceptions != 0:
    raise RuntimeError(
        "Unexpected graphene-ligand exceptions detected."
    )

if abs(
    graphene_total_charge
) > 1.0e-6:
    raise RuntimeError(
        "Graphene is not neutral."
    )

if abs(
    water_charge
) > 1.0e-6:
    raise RuntimeError(
        "OPC water is not neutral."
    )

if abs(
    ligand_charge
) > 1.0e-5:
    raise RuntimeError(
        "Ligand is unexpectedly non-neutral."
    )


# ============================================================
# DUMP LIGAND NONBONDED TABLE
# ============================================================

with open(
    LIGAND_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.writer(
        handle
    )

    writer.writerow(
        [
            "global_index",
            "local_ligand_index",
            "pyrene_aromatic",
            "charge_e",
            "sigma_nm",
            "epsilon_kJ_mol",
            "mass_amu",
        ]
    )


    for i in LIGAND:

        p = particle_parameters(
            nb,
            i,
        )

        writer.writerow(
            [
                i,
                i - 3750,
                int(i in set(PYRENE)),
                p["q_e"],
                p["sigma_nm"],
                p["epsilon_kJ_mol"],
                mass_amu(
                    system.getParticleMass(i)
                ),
            ]
        )


# ============================================================
# JSON
# ============================================================

results = {
    "system_xml": str(
        SYSTEM_XML
    ),

    "n_particles": n_particles,

    "graphene_cg1": cg1,

    "graphene_cge": cge,

    "opc_O": O,

    "opc_H1": H1,

    "opc_H2": H2,

    "opc_M": M,

    "charges": {
        "graphene_total_e": graphene_total_charge,
        "ligand_total_e": ligand_charge,
        "pyrene_subset_e": pyrene_charge,
        "one_water_e": water_charge,
        "system_total_e": total_charge,
    },

    "cross_parameters": {
        "cg1_O_sigma_nm": cg1_O_sigma_nm,
        "cg1_O_epsilon_kJ_mol": cg1_O_epsilon_kj,
        "cge_O_epsilon_kJ_mol": cge_O_epsilon_kj,
    },

    "cross_exceptions": {
        "graphene_water": graphene_water_exceptions,
        "graphene_ligand": graphene_ligand_exceptions,
        "ligand_water": ligand_water_exceptions,
    },

    "reference_values": {
        "iff_expected_openmm_cg1_sigma_nm": (
            IFF_EXPECTED_SIGMA_NM
        ),
        "iff_expected_cg1_epsilon_kJ_mol": (
            IFF_EXPECTED_EPS_KJ
        ),
        "opc_expected_O_sigma_nm": (
            OPC_EXPECTED_SIGMA_NM
        ),
        "opc_expected_O_epsilon_kJ_mol": (
            OPC_EXPECTED_EPS_KJ
        ),
    },
}


OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        results,
        handle,
        indent=2,
    )


print()
print(f"JSON:       {OUTPUT_JSON}")
print(f"Ligand CSV: {LIGAND_CSV}")

print()
print("FORCEFIELD_CROSS_PARAMETER_AUDIT_PASS")
print("=" * 78)