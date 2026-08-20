from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

import openmm
from openmm import unit


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

SYSTEM_XML = (
    ROOT / "parameters" / "combined"
    / "graphene_opc_wetting_cylinder.xml"
)

POSITIONS_NPY = (
    ROOT / "parameters" / "combined"
    / "graphene_opc_wetting_cylinder_positions_nm.npy"
)

BUILD_JSON = (
    ROOT / "analysis"
    / "graphene_opc_wetting_cylinder_build.json"
)

OUTPUT_JSON = (
    ROOT / "analysis"
    / "graphene_opc_wetting_cylinder_validation.json"
)

OUTPUT_FIGURE = (
    ROOT / "analysis" / "figures"
    / "17_graphene_opc_wetting_initial_geometry.png"
)


N_GRAPHENE = 3750
N_CARBON = 1250

EXPECTED_WATERS = 1642
EXPECTED_PARTICLES = (
    N_GRAPHENE
    +
    4 * EXPECTED_WATERS
)


# ============================================================
# HELPERS
# ============================================================

def vec_nm(v):
    return np.array(
        [
            v[0].value_in_unit(unit.nanometer),
            v[1].value_in_unit(unit.nanometer),
            v[2].value_in_unit(unit.nanometer),
        ],
        dtype=float,
    )


def minimum_image_xy(
    dx,
    dy,
    a,
    b,
):
    """
    Triclinic 2D minimum image.
    """

    cell = np.array(
        [
            [a[0], a[1]],
            [b[0], b[1]],
        ],
        dtype=float,
    )

    inv_cell = np.linalg.inv(
        cell
    )

    shape = np.broadcast(
        dx,
        dy,
    ).shape

    vectors = np.column_stack(
        [
            np.broadcast_to(
                dx,
                shape,
            ).ravel(),
            np.broadcast_to(
                dy,
                shape,
            ).ravel(),
        ]
    )

    frac = (
        vectors
        @
        inv_cell
    )

    frac -= np.round(
        frac
    )

    wrapped = (
        frac
        @
        cell
    )

    return (
        wrapped[:, 0].reshape(shape),
        wrapped[:, 1].reshape(shape),
    )


def minimum_oo_distance(
    oxygen_positions,
    a,
    b,
    chunk=150,
):
    """
    Minimum O-O distance using periodic xy and nonperiodic z.
    """

    n = len(
        oxygen_positions
    )

    best = np.inf


    for start in range(
        0,
        n,
        chunk,
    ):

        end = min(
            start + chunk,
            n,
        )

        p = oxygen_positions[
            start:end
        ]


        dx = (
            p[:, None, 0]
            -
            oxygen_positions[
                None, :, 0
            ]
        )

        dy = (
            p[:, None, 1]
            -
            oxygen_positions[
                None, :, 1
            ]
        )

        dz = (
            p[:, None, 2]
            -
            oxygen_positions[
                None, :, 2
            ]
        )


        dx, dy = minimum_image_xy(
            dx,
            dy,
            a,
            b,
        )


        r2 = (
            dx*dx
            +
            dy*dy
            +
            dz*dz
        )


        # Exclude self pairs.
        rows = np.arange(
            start,
            end,
        )

        local_rows = np.arange(
            end - start
        )

        r2[
            local_rows,
            rows,
        ] = np.inf


        best = min(
            best,
            float(
                np.sqrt(
                    np.min(r2)
                )
            ),
        )


    return best


def minimum_oc_distance(
    oxygen_positions,
    carbon_positions,
    a,
    b,
    chunk=100,
):
    """
    Minimum water-O to graphene-carbon distance.
    """

    best = np.inf


    for start in range(
        0,
        len(oxygen_positions),
        chunk,
    ):

        p = oxygen_positions[
            start:start+chunk
        ]


        dx = (
            p[:, None, 0]
            -
            carbon_positions[
                None, :, 0
            ]
        )

        dy = (
            p[:, None, 1]
            -
            carbon_positions[
                None, :, 1
            ]
        )

        dz = (
            p[:, None, 2]
            -
            carbon_positions[
                None, :, 2
            ]
        )


        dx, dy = minimum_image_xy(
            dx,
            dy,
            a,
            b,
        )


        r = np.sqrt(
            dx*dx
            +
            dy*dy
            +
            dz*dz
        )


        best = min(
            best,
            float(
                np.min(r)
            ),
        )


    return best


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 78)
print("GRAPHENE / OPC WETTING CYLINDER VALIDATION")
print("=" * 78)


system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text(
        encoding="utf-8"
    )
)

positions = np.load(
    POSITIONS_NPY
)

metadata = json.loads(
    BUILD_JSON.read_text(
        encoding="utf-8"
    )
)


n_particles = (
    system.getNumParticles()
)

n_virtual_sites = sum(
    system.isVirtualSite(i)
    for i in range(
        n_particles
    )
)


print()
print(
    f"Particles:      {n_particles}"
)

print(
    f"Virtual sites:  {n_virtual_sites}"
)

print(
    f"Constraints:    {system.getNumConstraints()}"
)


if n_particles != EXPECTED_PARTICLES:
    raise RuntimeError(
        "Particle count mismatch."
    )

if n_virtual_sites != EXPECTED_WATERS:
    raise RuntimeError(
        "Virtual-site count mismatch."
    )

if system.getNumConstraints() != (
    3 * EXPECTED_WATERS
):
    raise RuntimeError(
        "Constraint count mismatch."
    )


# ============================================================
# BOX
# ============================================================

a_raw, b_raw, c_raw = (
    system.getDefaultPeriodicBoxVectors()
)

a = vec_nm(
    a_raw
)

b = vec_nm(
    b_raw
)

c = vec_nm(
    c_raw
)


print()
print("Box vectors:")

print(
    f"  a = {a}"
)

print(
    f"  b = {b}"
)

print(
    f"  c = {c}"
)


# ============================================================
# SELECTIONS
# ============================================================

carbons = positions[
    :N_CARBON
]

oxygen_indices = np.arange(
    N_GRAPHENE,
    n_particles,
    4,
)

oxygens = positions[
    oxygen_indices
]


if len(oxygens) != EXPECTED_WATERS:
    raise RuntimeError(
        "Water oxygen count mismatch."
    )


carbon_plane = float(
    np.mean(
        carbons[:, 2]
    )
)


print()
print(
    f"Carbon plane:   "
    f"{carbon_plane:.6f} nm"
)

print(
    f"Water O range:  "
    f"{oxygens[:,2].min():.6f} -> "
    f"{oxygens[:,2].max():.6f} nm"
)


# ============================================================
# OVERLAP CHECKS
# ============================================================

print()
print("Checking O-O distances...")

min_oo = minimum_oo_distance(
    oxygens,
    a,
    b,
)


print(
    f"Minimum O-O distance: "
    f"{min_oo:.6f} nm "
    f"({10*min_oo:.3f} A)"
)


print()
print("Checking O-C distances...")

min_oc = minimum_oc_distance(
    oxygens,
    carbons,
    a,
    b,
)


print(
    f"Minimum O-C distance: "
    f"{min_oc:.6f} nm "
    f"({10*min_oc:.3f} A)"
)


# Conservative overlap flags.
if min_oo < 0.20:
    raise RuntimeError(
        "Severe water-water overlap detected."
    )

if min_oc < 0.20:
    raise RuntimeError(
        "Severe water-graphene overlap detected."
    )


# ============================================================
# CYLINDER GEOMETRY
# ============================================================

radius = float(
    metadata[
        "cylinder"
    ][
        "radius_nm"
    ]
)

center_y = float(
    metadata[
        "cylinder"
    ][
        "center_y_nm"
    ]
)

center_z = float(
    metadata[
        "cylinder"
    ][
        "center_z_nm"
    ]
)


ly = abs(
    b[1]
)


dy = (
    oxygens[:, 1]
    -
    center_y
)

dy -= (
    np.round(
        dy / ly
    )
    *
    ly
)


dz = (
    oxygens[:, 2]
    -
    center_z
)


radial = np.sqrt(
    dy*dy
    +
    dz*dz
)


print()
print(
    f"Initial cross-section radius:"
)

print(
    f"  maximum O radius = "
    f"{radial.max():.6f} nm"
)

print(
    f"  target radius    = "
    f"{radius:.6f} nm"
)


# ============================================================
# INITIAL OPENMM ENERGY/FORCE CHECK
# ============================================================

print()
print("Creating OpenMM Context...")


try:
    platform = (
        openmm.Platform
        .getPlatformByName(
            "CUDA"
        )
    )

    properties = {
        "Precision": "mixed"
    }

except Exception:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CPU"
        )
    )

    properties = {}


integrator = openmm.VerletIntegrator(
    0.001
    *
    unit.picoseconds
)


context = openmm.Context(
    system,
    integrator,
    platform,
    properties,
)


context.setPositions(
    positions
    *
    unit.nanometer
)


context.computeVirtualSites()


state = context.getState(
    getEnergy=True,
    getForces=True,
)


energy = float(
    state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


forces = np.asarray(
    state
    .getForces(
        asNumpy=True
    )
    .value_in_unit(
        unit.kilojoule_per_mole
        /
        unit.nanometer
    ),
    dtype=float,
)


force_magnitude = np.sqrt(
    np.sum(
        forces*forces,
        axis=1,
    )
)


max_force = float(
    np.max(
        force_magnitude
    )
)


rms_force = float(
    np.sqrt(
        np.mean(
            force_magnitude**2
        )
    )
)


print()
print(
    f"Platform:              "
    f"{platform.getName()}"
)

print(
    f"Initial potential E:   "
    f"{energy:.3f} kJ/mol"
)

print(
    f"Initial RMS |F|:       "
    f"{rms_force:.3f} kJ/mol/nm"
)

print(
    f"Initial max |F|:       "
    f"{max_force:.3f} kJ/mol/nm"
)


if not np.isfinite(
    energy
):
    raise RuntimeError(
        "Non-finite potential energy."
    )

if not np.all(
    np.isfinite(
        forces
    )
):
    raise RuntimeError(
        "Non-finite force detected."
    )


# ============================================================
# FIGURE
# ============================================================

OUTPUT_FIGURE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.5),
)


# ------------------------------------------------------------
# y-z cylinder cross-section
# ------------------------------------------------------------

ax = axes[0]


ax.scatter(
    dy,
    oxygens[:, 2],
    s=6,
    alpha=0.5,
)


theta = np.linspace(
    0,
    2*np.pi,
    500,
)


ax.plot(
    radius
    *
    np.cos(theta),

    center_z
    +
    radius
    *
    np.sin(theta),

    linestyle="--",
    linewidth=1.5,
    label="Initial selection boundary",
)


ax.axhline(
    carbon_plane,
    linewidth=2,
    label="Graphene carbon plane",
)


ax.set_xlabel(
    "y relative to cylinder center (nm)"
)

ax.set_ylabel(
    "z (nm)"
)

ax.set_title(
    "A. Water-oxygen cylinder cross-section"
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.grid(
    alpha=0.2
)

ax.legend(
    frameon=False,
)


# ------------------------------------------------------------
# x-z projection
# ------------------------------------------------------------

ax = axes[1]


ax.scatter(
    oxygens[:, 0],
    oxygens[:, 2],
    s=6,
    alpha=0.5,
)


ax.axhline(
    carbon_plane,
    linewidth=2,
    label="Graphene carbon plane",
)


ax.set_xlabel(
    "x (nm)"
)

ax.set_ylabel(
    "z (nm)"
)

ax.set_title(
    "B. Cylinder along periodic x direction"
)

ax.grid(
    alpha=0.2
)

ax.legend(
    frameon=False,
)


fig.suptitle(
    "Initial Graphene-OPC Wetting Benchmark Geometry",
    fontsize=16,
)


fig.tight_layout()


fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# SAVE
# ============================================================

results = {
    "particles": n_particles,
    "water_count": EXPECTED_WATERS,
    "virtual_sites": n_virtual_sites,
    "constraints": system.getNumConstraints(),
    "carbon_plane_nm": carbon_plane,
    "water_O_min_z_nm": float(
        oxygens[:,2].min()
    ),
    "water_O_max_z_nm": float(
        oxygens[:,2].max()
    ),
    "minimum_OO_nm": min_oo,
    "minimum_OC_nm": min_oc,
    "maximum_initial_radial_O_nm": float(
        radial.max()
    ),
    "initial_potential_energy_kJ_mol": energy,
    "initial_rms_force_kJ_mol_nm": rms_force,
    "initial_max_force_kJ_mol_nm": max_force,
    "platform": platform.getName(),
}


OUTPUT_JSON.write_text(
    json.dumps(
        results,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 78)
print("VALIDATION SUMMARY")
print("=" * 78)

print(
    f"Minimum O-O: "
    f"{10*min_oo:.3f} A"
)

print(
    f"Minimum O-C: "
    f"{10*min_oc:.3f} A"
)

print(
    f"Energy finite: "
    f"{np.isfinite(energy)}"
)

print(
    f"Forces finite: "
    f"{np.all(np.isfinite(forces))}"
)

print()
print(
    f"Figure:"
    f"\n  {OUTPUT_FIGURE}"
)

print(
    f"JSON:"
    f"\n  {OUTPUT_JSON}"
)

print()
print(
    "GRAPHENE_OPC_WETTING_INITIAL_GEOMETRY_PASS"
)

print("=" * 78)