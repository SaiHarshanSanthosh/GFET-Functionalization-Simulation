from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openmm
from openmm import unit


# ============================================================
# FF-1A
# SINGLE OPC WATER / GRAPHENE INTERACTION BENCHMARK
#
# External target:
#
# Modern many-body calculations:
# ~ -98 +/- 10 meV
# ~ -9.46 +/- 0.96 kJ/mol
#
# Most stable configuration:
# two H atoms directed toward graphene ("2-leg").
#
# This calculation:
#
# - uses ACTUAL production XML q/sigma/epsilon
# - uses ideal supported graphene coordinates
# - directly evaluates graphene-water pair interactions
# - does NOT use PME
# - does NOT use the water slab
# - does NOT run MD
#
# Graphene is tiled symmetrically in xy to approximate an
# infinite sheet.
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

GRAPHENE_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_supported_positions_nm.npy"
)

# Fallback only if the clean supported geometry file is absent.
ENDPOINT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_production_300K_5500ps_positions_nm.npy"
)

OUTPUT_CSV = (
    ROOT
    / "analysis"
    / "single_water_graphene_benchmark.csv"
)

CONVERGENCE_CSV = (
    ROOT
    / "analysis"
    / "single_water_graphene_tile_convergence.csv"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "single_water_graphene_benchmark.json"
)

OUTPUT_FIGURE = (
    ROOT
    / "analysis"
    / "figures"
    / "16_single_water_graphene_benchmark.png"
)


# ============================================================
# CONSTANTS
# ============================================================

K_E = 138.935456  # kJ mol^-1 nm e^-2

N_GRAPHENE = 3750
N_CARBON = 1250

WATER_O = 3820
WATER_H1 = 3821
WATER_H2 = 3822
WATER_M = 3823

WATER_INDICES = [
    WATER_O,
    WATER_H1,
    WATER_H2,
    WATER_M,
]

# Main production scan.
TILE_RADIUS = 3

# z = O distance above mean graphene carbon plane.
DISTANCES_A = np.arange(
    2.50,
    6.0001,
    0.05,
)

FAR_REFERENCE_A = 15.0


# Modern many-body benchmark.
BENCHMARK_ENERGY_MEV = -98.0
BENCHMARK_ENERGY_UNCERTAINTY_MEV = 10.0

MEV_TO_KJ_MOL = 0.0964853321

BENCHMARK_ENERGY_KJ_MOL = (
    BENCHMARK_ENERGY_MEV
    *
    MEV_TO_KJ_MOL
)

BENCHMARK_UNCERTAINTY_KJ_MOL = (
    BENCHMARK_ENERGY_UNCERTAINTY_MEV
    *
    MEV_TO_KJ_MOL
)


# ============================================================
# HELPERS
# ============================================================

def q_e(q):
    return float(
        q.value_in_unit(
            unit.elementary_charge
        )
    )


def sigma_nm(x):
    return float(
        x.value_in_unit(
            unit.nanometer
        )
    )


def eps_kj(x):
    return float(
        x.value_in_unit(
            unit.kilojoule_per_mole
        )
    )


def vec_nm(v):
    return np.array(
        [
            v[0].value_in_unit(unit.nanometer),
            v[1].value_in_unit(unit.nanometer),
            v[2].value_in_unit(unit.nanometer),
        ],
        dtype=float,
    )


def minimum_image_fractional_xy(
    xy,
    cell2,
):
    """
    Return xy positions represented relative to the center
    of the periodic cell, in approximately [-0.5,0.5).
    """

    inv_cell = np.linalg.inv(
        cell2
    )

    frac = (
        xy
        @
        inv_cell
    )

    frac_centered = (
        frac
        -
        np.floor(
            frac
            +
            0.5
        )
    )

    return (
        frac_centered
        @
        cell2
    )


def interaction_energy(
    graphene_xyz,
    graphene_q,
    graphene_sigma,
    graphene_eps,
    water_xyz,
    water_q,
    water_sigma,
    water_eps,
):
    """
    Direct pair interaction between tiled graphene particles
    and four OPC sites.

    Standard OpenMM Lorentz-Berthelot:
      sigma_ij = (sigma_i + sigma_j)/2
      epsilon_ij = sqrt(epsilon_i epsilon_j)
    """

    total = 0.0

    for j in range(4):

        dx = (
            graphene_xyz[:, 0]
            -
            water_xyz[j, 0]
        )

        dy = (
            graphene_xyz[:, 1]
            -
            water_xyz[j, 1]
        )

        dz = (
            graphene_xyz[:, 2]
            -
            water_xyz[j, 2]
        )

        r = np.sqrt(
            dx*dx
            +
            dy*dy
            +
            dz*dz
        )


        # Coulomb
        electrostatic = (
            K_E
            *
            graphene_q
            *
            water_q[j]
            /
            r
        )


        # Lennard-Jones
        sigma_ij = (
            graphene_sigma
            +
            water_sigma[j]
        ) / 2.0

        epsilon_ij = np.sqrt(
            np.maximum(
                graphene_eps,
                0.0,
            )
            *
            max(
                water_eps[j],
                0.0,
            )
        )


        active_lj = (
            epsilon_ij
            >
            0.0
        )

        lj = np.zeros_like(
            r
        )


        if np.any(active_lj):

            sr = (
                sigma_ij[active_lj]
                /
                r[active_lj]
            )

            sr6 = sr**6

            lj[
                active_lj
            ] = (
                4.0
                *
                epsilon_ij[active_lj]
                *
                (
                    sr6*sr6
                    -
                    sr6
                )
            )


        total += float(
            np.sum(
                electrostatic
                +
                lj
            )
        )


    return total


def rotate_z(
    xyz,
    angle_deg,
):
    angle = math.radians(
        angle_deg
    )

    c = math.cos(angle)
    s = math.sin(angle)

    R = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return (
        xyz
        @
        R.T
    )


# ============================================================
# LOAD SYSTEM
# ============================================================

print()
print("=" * 80)
print("FF-1A: SINGLE OPC WATER / GRAPHENE BENCHMARK")
print("=" * 80)


system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text(
        encoding="utf-8"
    )
)


nb = None

for i in range(
    system.getNumForces()
):

    force = system.getForce(i)

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        if nb is not None:
            raise RuntimeError(
                "Multiple NonbondedForce objects found."
            )

        nb = force


if nb is None:
    raise RuntimeError(
        "No NonbondedForce found."
    )


# ============================================================
# GRAPHENE PARAMETERS
# ============================================================

g_q = np.zeros(
    N_GRAPHENE,
    dtype=float,
)

g_sigma = np.zeros(
    N_GRAPHENE,
    dtype=float,
)

g_eps = np.zeros(
    N_GRAPHENE,
    dtype=float,
)


for i in range(
    N_GRAPHENE
):

    q, sigma, epsilon = (
        nb.getParticleParameters(i)
    )

    g_q[i] = q_e(q)
    g_sigma[i] = sigma_nm(sigma)
    g_eps[i] = eps_kj(epsilon)


# ============================================================
# WATER PARAMETERS
# ============================================================

w_q = np.zeros(
    4,
    dtype=float,
)

w_sigma = np.zeros(
    4,
    dtype=float,
)

w_eps = np.zeros(
    4,
    dtype=float,
)


for j, i in enumerate(
    WATER_INDICES
):

    q, sigma, epsilon = (
        nb.getParticleParameters(i)
    )

    w_q[j] = q_e(q)
    w_sigma[j] = sigma_nm(sigma)
    w_eps[j] = eps_kj(epsilon)


# ============================================================
# EXACT OPC GEOMETRY FROM CONSTRAINTS
# ============================================================

constraint_distances = {}


for k in range(
    system.getNumConstraints()
):

    p1, p2, d = (
        system.getConstraintParameters(k)
    )

    p1 = int(p1)
    p2 = int(p2)

    if (
        p1 in WATER_INDICES
        and
        p2 in WATER_INDICES
    ):

        constraint_distances[
            tuple(
                sorted(
                    (
                        p1,
                        p2,
                    )
                )
            )
        ] = float(
            d.value_in_unit(
                unit.nanometer
            )
        )


OH1 = constraint_distances[
    tuple(
        sorted(
            (
                WATER_O,
                WATER_H1,
            )
        )
    )
]

OH2 = constraint_distances[
    tuple(
        sorted(
            (
                WATER_O,
                WATER_H2,
            )
        )
    )
]

HH = constraint_distances[
    tuple(
        sorted(
            (
                WATER_H1,
                WATER_H2,
            )
        )
    )
]


OH = (
    OH1
    +
    OH2
) / 2.0


theta = math.acos(
    (
        2.0*OH*OH
        -
        HH*HH
    )
    /
    (
        2.0*OH*OH
    )
)


virtual_site = system.getVirtualSite(
    WATER_M
)


if not isinstance(
    virtual_site,
    openmm.ThreeParticleAverageSite,
):
    raise RuntimeError(
        "OPC M site is not a ThreeParticleAverageSite."
    )


weights = [
    float(
        virtual_site.getWeight(i)
    )
    for i in range(3)
]


print()
print("Exact OPC geometry from production System:")

print(
    f"  O-H = "
    f"{OH:.9f} nm"
)

print(
    f"  H-H = "
    f"{HH:.9f} nm"
)

print(
    f"  angle = "
    f"{math.degrees(theta):.6f} deg"
)

print(
    f"  M weights = "
    f"{weights}"
)


# ============================================================
# LOAD CLEAN GRAPHENE GEOMETRY
# ============================================================

if GRAPHENE_POSITIONS.exists():

    source_positions = np.load(
        GRAPHENE_POSITIONS
    )

    print()
    print(
        "Using clean supported graphene coordinates:"
    )

    print(
        f"  {GRAPHENE_POSITIONS}"
    )

else:

    source_positions = np.load(
        ENDPOINT_POSITIONS
    )

    print()
    print(
        "WARNING: clean supported coordinates absent."
    )

    print(
        "Using 5.5 ns graphene coordinates instead."
    )


if source_positions.shape[0] < N_GRAPHENE:
    raise RuntimeError(
        "Position file does not contain full graphene."
    )


graphene_xyz = np.asarray(
    source_positions[
        :N_GRAPHENE,
        :
    ],
    dtype=float,
).copy()


# ============================================================
# PERIODIC CELL
# ============================================================

a, b, c = (
    system
    .getDefaultPeriodicBoxVectors()
)

a_nm = vec_nm(a)
b_nm = vec_nm(b)


cell2 = np.array(
    [
        [a_nm[0], a_nm[1]],
        [b_nm[0], b_nm[1]],
    ],
    dtype=float,
)


# Put cell center at xy = 0.
graphene_xyz[
    :,
    :2,
] = minimum_image_fractional_xy(
    graphene_xyz[
        :,
        :2,
    ],
    cell2,
)


carbon_plane_nm = float(
    np.mean(
        graphene_xyz[
            :N_CARBON,
            2,
        ]
    )
)


print()
print(
    f"Mean carbon plane: "
    f"{carbon_plane_nm:.8f} nm"
)


# ============================================================
# DEFINE TOP / BRIDGE / HOLLOW SITES
# ============================================================

carbon_xy = (
    graphene_xyz[
        :N_CARBON,
        :2,
    ]
)


# Top site: carbon closest to cell center.
r2_center = np.sum(
    carbon_xy**2,
    axis=1,
)

top_index = int(
    np.argmin(
        r2_center
    )
)

top_xy = (
    carbon_xy[
        top_index
    ].copy()
)


# Bridge site: midpoint to closest bonded neighbor.
distances_from_top = np.sqrt(
    np.sum(
        (
            carbon_xy
            -
            top_xy
        )**2,
        axis=1,
    )
)

neighbor_order = np.argsort(
    distances_from_top
)

neighbor_candidates = [
    int(i)
    for i in neighbor_order
    if i != top_index
]


bridge_neighbor = (
    neighbor_candidates[0]
)

bond_length_nm = float(
    distances_from_top[
        bridge_neighbor
    ]
)


if not (
    0.13
    <
    bond_length_nm
    <
    0.15
):
    raise RuntimeError(
        "Nearest-neighbor graphene distance "
        f"looks wrong: {bond_length_nm} nm"
    )


bridge_xy = (
    0.5
    *
    (
        top_xy
        +
        carbon_xy[
            bridge_neighbor
        ]
    )
)


# ------------------------------------------------------------
# Hollow site:
#
# Find six-carbon rings in a local region around the center.
# ------------------------------------------------------------

local_indices = np.where(
    np.sqrt(
        np.sum(
            carbon_xy**2,
            axis=1,
        )
    )
    <
    0.8
)[0]


local_set = set(
    int(i)
    for i in local_indices
)


adjacency = {
    int(i): []
    for i in local_indices
}


for ii, i in enumerate(
    local_indices
):

    for j in local_indices[
        ii + 1:
    ]:

        d = float(
            np.linalg.norm(
                carbon_xy[i]
                -
                carbon_xy[j]
            )
        )

        if (
            0.13
            <
            d
            <
            0.15
        ):
            adjacency[int(i)].append(
                int(j)
            )

            adjacency[int(j)].append(
                int(i)
            )


cycles = set()


def canonical_cycle(cycle):

    cycle = list(
        cycle
    )

    variants = []

    for candidate in (
        cycle,
        list(
            reversed(
                cycle
            )
        ),
    ):

        for shift in range(6):

            rotated = (
                candidate[shift:]
                +
                candidate[:shift]
            )

            variants.append(
                tuple(
                    rotated
                )
            )

    return min(
        variants
    )


def dfs_cycle(
    start,
    current,
    path,
):

    if len(path) == 6:

        if start in adjacency[current]:
            cycles.add(
                canonical_cycle(
                    path
                )
            )

        return


    for nxt in adjacency[current]:

        if nxt == start:
            continue

        if nxt in path:
            continue

        dfs_cycle(
            start,
            nxt,
            path + [nxt],
        )


for start in local_indices:

    dfs_cycle(
        int(start),
        int(start),
        [
            int(start)
        ],
    )


if len(cycles) == 0:
    raise RuntimeError(
        "Could not identify graphene hexagons."
    )


best_cycle = None
best_center_distance = None
best_center = None


for cycle in cycles:

    xy = carbon_xy[
        list(cycle)
    ]

    center = np.mean(
        xy,
        axis=0,
    )

    radii = np.sqrt(
        np.sum(
            (
                xy
                -
                center
            )**2,
            axis=1,
        )
    )


    # Real graphene hexagon should be regular.
    if np.std(radii) > 0.01:
        continue


    center_distance = float(
        np.linalg.norm(
            center
        )
    )


    if (
        best_center_distance is None
        or
        center_distance
        <
        best_center_distance
    ):

        best_center_distance = (
            center_distance
        )

        best_cycle = (
            cycle
        )

        best_center = (
            center
        )


if best_center is None:
    raise RuntimeError(
        "No valid graphene hexagon found."
    )


hollow_xy = np.asarray(
    best_center,
    dtype=float,
)


sites = {
    "top": top_xy,
    "bridge": bridge_xy,
    "hollow": hollow_xy,
}


print()
print("Adsorption sites:")

print(
    f"  top carbon index: "
    f"{top_index}"
)

print(
    f"  bridge neighbor:  "
    f"{bridge_neighbor}"
)

print(
    f"  C-C bond length:  "
    f"{bond_length_nm:.6f} nm"
)

print(
    f"  hollow ring:      "
    f"{best_cycle}"
)


# ============================================================
# WATER ORIENTATIONS
# ============================================================

def add_m_site(
    O,
    H1,
    H2,
):
    """
    Reproduce OpenMM ThreeParticleAverageSite.

    Parent order in OPC is expected O,H1,H2.
    """

    return (
        weights[0] * O
        +
        weights[1] * H1
        +
        weights[2] * H2
    )


def make_water(
    orientation,
    O_height_nm,
    xy,
):

    O = np.array(
        [
            xy[0],
            xy[1],
            O_height_nm,
        ],
        dtype=float,
    )


    if orientation == "2-leg":

        half = (
            theta
            /
            2.0
        )

        lateral = (
            OH
            *
            math.sin(
                half
            )
        )

        down = (
            OH
            *
            math.cos(
                half
            )
        )

        H1 = O + np.array(
            [
                +lateral,
                0.0,
                -down,
            ]
        )

        H2 = O + np.array(
            [
                -lateral,
                0.0,
                -down,
            ]
        )


    elif orientation == "0-leg":

        half = (
            theta
            /
            2.0
        )

        lateral = (
            OH
            *
            math.sin(
                half
            )
        )

        up = (
            OH
            *
            math.cos(
                half
            )
        )

        H1 = O + np.array(
            [
                +lateral,
                0.0,
                +up,
            ]
        )

        H2 = O + np.array(
            [
                -lateral,
                0.0,
                +up,
            ]
        )


    elif orientation == "1-leg":

        H1 = O + np.array(
            [
                0.0,
                0.0,
                -OH,
            ]
        )

        H2 = O + np.array(
            [
                OH
                *
                math.sin(theta),

                0.0,

                -OH
                *
                math.cos(theta),
            ]
        )


    else:

        raise ValueError(
            orientation
        )


    M = add_m_site(
        O,
        H1,
        H2,
    )


    return np.vstack(
        [
            O,
            H1,
            H2,
            M,
        ]
    )


# ============================================================
# TILE GRAPHENE
# ============================================================

def make_tiled_graphene(
    radius,
):

    xyz_list = []
    q_list = []
    sigma_list = []
    eps_list = []


    for ia in range(
        -radius,
        radius + 1,
    ):

        for ib in range(
            -radius,
            radius + 1,
        ):

            shift = (
                ia * a_nm
                +
                ib * b_nm
            )

            xyz = (
                graphene_xyz
                +
                shift
            )

            xyz_list.append(
                xyz
            )

            q_list.append(
                g_q
            )

            sigma_list.append(
                g_sigma
            )

            eps_list.append(
                g_eps
            )


    return (
        np.concatenate(
            xyz_list,
            axis=0,
        ),
        np.concatenate(
            q_list,
        ),
        np.concatenate(
            sigma_list,
        ),
        np.concatenate(
            eps_list,
        ),
    )


(
    tiled_xyz,
    tiled_q,
    tiled_sigma,
    tiled_eps,
) = make_tiled_graphene(
    TILE_RADIUS
)


print()
print(
    f"Main tiling: "
    f"{2*TILE_RADIUS+1} x "
    f"{2*TILE_RADIUS+1} cells"
)

print(
    f"Tiled graphene particles: "
    f"{len(tiled_xyz)}"
)


# ============================================================
# MAIN SCAN
# ============================================================

rows = []


print()
print("=" * 80)
print("INTERACTION SCANS")
print("=" * 80)


for site_name, site_xy in sites.items():

    for orientation in [
        "0-leg",
        "1-leg",
        "2-leg",
    ]:

        # Far reference with exact same orientation/site.
        far_water = make_water(
            orientation,
            carbon_plane_nm
            +
            FAR_REFERENCE_A
            *
            0.1,
            site_xy,
        )


        far_energy = interaction_energy(
            tiled_xyz,
            tiled_q,
            tiled_sigma,
            tiled_eps,
            far_water,
            w_q,
            w_sigma,
            w_eps,
        )


        config_energies = []


        for distance_A in DISTANCES_A:

            water = make_water(
                orientation,
                carbon_plane_nm
                +
                distance_A
                *
                0.1,
                site_xy,
            )


            raw_energy = interaction_energy(
                tiled_xyz,
                tiled_q,
                tiled_sigma,
                tiled_eps,
                water,
                w_q,
                w_sigma,
                w_eps,
            )


            interaction = (
                raw_energy
                -
                far_energy
            )


            config_energies.append(
                interaction
            )


            rows.append(
                {
                    "site": site_name,
                    "orientation": orientation,
                    "O_graphene_distance_A": (
                        float(
                            distance_A
                        )
                    ),
                    "interaction_energy_kJ_mol": (
                        float(
                            interaction
                        )
                    ),
                    "tile_radius": TILE_RADIUS,
                    "far_reference_A": (
                        FAR_REFERENCE_A
                    ),
                }
            )


        config_energies = np.asarray(
            config_energies
        )


        imin = int(
            np.argmin(
                config_energies
            )
        )


        print(
            f"{site_name:7s}  "
            f"{orientation:5s}  "
            f"minimum = "
            f"{config_energies[imin]:8.4f} kJ/mol  "
            f"at "
            f"{DISTANCES_A[imin]:.2f} A"
        )


results = pd.DataFrame(
    rows
)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# GLOBAL MINIMUM
# ============================================================

best_index = int(
    results[
        "interaction_energy_kJ_mol"
    ].idxmin()
)

best = results.loc[
    best_index
]


best_site = str(
    best["site"]
)

best_orientation = str(
    best["orientation"]
)

best_distance_A = float(
    best[
        "O_graphene_distance_A"
    ]
)

best_energy = float(
    best[
        "interaction_energy_kJ_mol"
    ]
)


print()
print("=" * 80)
print("GLOBAL MINIMUM")
print("=" * 80)

print(
    f"Site:              "
    f"{best_site}"
)

print(
    f"Orientation:       "
    f"{best_orientation}"
)

print(
    f"O-surface height:  "
    f"{best_distance_A:.3f} A"
)

print(
    f"Binding energy:    "
    f"{best_energy:.6f} kJ/mol"
)

print(
    f"Binding energy:    "
    f"{best_energy / MEV_TO_KJ_MOL:.3f} meV"
)

print()
print(
    f"Modern benchmark:  "
    f"{BENCHMARK_ENERGY_KJ_MOL:.4f} "
    f"+/- "
    f"{BENCHMARK_UNCERTAINTY_KJ_MOL:.4f} "
    f"kJ/mol"
)

print(
    f"                   "
    f"{BENCHMARK_ENERGY_MEV:.1f} "
    f"+/- "
    f"{BENCHMARK_ENERGY_UNCERTAINTY_MEV:.1f} "
    f"meV"
)


energy_error = (
    best_energy
    -
    BENCHMARK_ENERGY_KJ_MOL
)


print(
    f"Energy difference: "
    f"{energy_error:+.4f} kJ/mol"
)


# ============================================================
# TILE CONVERGENCE AT GLOBAL MINIMUM
# ============================================================

print()
print("=" * 80)
print("TILE CONVERGENCE AT GLOBAL MINIMUM")
print("=" * 80)


convergence_rows = []


for radius in [
    1,
    2,
    3,
    4,
]:

    (
        gx,
        gq,
        gs,
        ge,
    ) = make_tiled_graphene(
        radius
    )


    water = make_water(
        best_orientation,
        carbon_plane_nm
        +
        best_distance_A
        *
        0.1,
        sites[
            best_site
        ],
    )


    far_water = make_water(
        best_orientation,
        carbon_plane_nm
        +
        FAR_REFERENCE_A
        *
        0.1,
        sites[
            best_site
        ],
    )


    e = interaction_energy(
        gx,
        gq,
        gs,
        ge,
        water,
        w_q,
        w_sigma,
        w_eps,
    )


    e_far = interaction_energy(
        gx,
        gq,
        gs,
        ge,
        far_water,
        w_q,
        w_sigma,
        w_eps,
    )


    binding = (
        e
        -
        e_far
    )


    convergence_rows.append(
        {
            "tile_radius": radius,
            "cells_per_side": (
                2*radius + 1
            ),
            "binding_energy_kJ_mol": (
                float(
                    binding
                )
            ),
        }
    )


    print(
        f"radius {radius}:  "
        f"{2*radius+1}x"
        f"{2*radius+1} cells  "
        f"E = {binding:.6f} kJ/mol"
    )


convergence = pd.DataFrame(
    convergence_rows
)

CONVERGENCE_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

convergence.to_csv(
    CONVERGENCE_CSV,
    index=False,
)


# ============================================================
# FIGURE
# ============================================================

OUTPUT_FIGURE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fig, ax = plt.subplots(
    figsize=(10.5, 6.5)
)


for (
    site_name,
    orientation
), group in results.groupby(
    [
        "site",
        "orientation",
    ]
):

    ax.plot(
        group[
            "O_graphene_distance_A"
        ],
        group[
            "interaction_energy_kJ_mol"
        ],
        linewidth=1.8,
        label=(
            f"{site_name}, "
            f"{orientation}"
        ),
    )


ax.axhspan(
    BENCHMARK_ENERGY_KJ_MOL
    -
    BENCHMARK_UNCERTAINTY_KJ_MOL,

    BENCHMARK_ENERGY_KJ_MOL
    +
    BENCHMARK_UNCERTAINTY_KJ_MOL,

    alpha=0.12,

    label=(
        "Many-body benchmark "
        "(-98 +/- 10 meV)"
    ),
)


ax.axhline(
    0.0,
    linewidth=1.0,
)


ax.scatter(
    [
        best_distance_A
    ],
    [
        best_energy
    ],
    s=90,
    marker="*",
    zorder=10,
)


ax.set_xlabel(
    "Water O - graphene distance (A)"
)

ax.set_ylabel(
    "Graphene-water interaction energy (kJ/mol)"
)

ax.set_title(
    "Single-Water Adsorption on Graphene: "
    "IFF + OPC Benchmark"
)

ax.grid(
    alpha=0.22
)

ax.legend(
    frameon=False,
    fontsize=8,
    ncol=2,
)


fig.tight_layout()

fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "benchmark": {
        "energy_meV": (
            BENCHMARK_ENERGY_MEV
        ),
        "uncertainty_meV": (
            BENCHMARK_ENERGY_UNCERTAINTY_MEV
        ),
        "energy_kJ_mol": (
            BENCHMARK_ENERGY_KJ_MOL
        ),
        "uncertainty_kJ_mol": (
            BENCHMARK_UNCERTAINTY_KJ_MOL
        ),
    },

    "global_minimum": {
        "site": (
            best_site
        ),
        "orientation": (
            best_orientation
        ),
        "distance_A": (
            best_distance_A
        ),
        "energy_kJ_mol": (
            best_energy
        ),
        "energy_meV": (
            best_energy
            /
            MEV_TO_KJ_MOL
        ),
        "difference_from_benchmark_kJ_mol": (
            energy_error
        ),
    },

    "tile_radius": (
        TILE_RADIUS
    ),

    "far_reference_A": (
        FAR_REFERENCE_A
    ),

    "graphene_coordinate_source": str(
        GRAPHENE_POSITIONS
        if GRAPHENE_POSITIONS.exists()
        else ENDPOINT_POSITIONS
    ),

    "note": (
        "Rigid single-water interaction scan. "
        "Not a finite-temperature adsorption free energy."
    ),
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        metadata,
        handle,
        indent=2,
    )


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    f"Scan CSV:       "
    f"{OUTPUT_CSV}"
)

print(
    f"Convergence:    "
    f"{CONVERGENCE_CSV}"
)

print(
    f"JSON:           "
    f"{OUTPUT_JSON}"
)

print(
    f"Figure:         "
    f"{OUTPUT_FIGURE}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is a rigid pair-interaction benchmark, "
    "not an adsorption free energy."
)

print(
    "The purpose is to determine whether our exact "
    "IFF-OPC cross interactions reproduce the "
    "independent water/graphene interaction scale."
)

print()
print("SINGLE_WATER_GRAPHENE_BENCHMARK_COMPLETE")
print("=" * 80)