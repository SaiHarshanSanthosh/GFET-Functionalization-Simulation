from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from openmm import openmm, unit


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
PARAMETERS = ROOT / "parameters" / "combined"

SYSTEM_XML = (
    PARAMETERS /
    "ff3_explicit_dpbs_opc_yb_replicaB.xml"
)

ASSEMBLY_METADATA = (
    ANALYSIS /
    "ff3_explicit_dpbs_opc_yb_replicaB_assembly_check.json"
)

SCREENING_TRAJECTORY = (
    ANALYSIS /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_screening_trajectory_float32.npy"
)

SCREENING_METADATA = (
    ANALYSIS /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_screening_trajectory.json"
)

WATER_TRAJECTORY = (
    ANALYSIS /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_opc_ohh_10ps_float32.npy"
)

WATER_METADATA = (
    ANALYSIS /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_opc_ohh_10ps.json"
)

PREFIX = (
    ANALYSIS /
    "ff3_explicit_dpbs_opc_yb_replicaB_0to10ns_"
    "solution_electrostatic_potential"
)

OUT_CSV = Path(str(PREFIX) + ".csv")
OUT_JSON = Path(str(PREFIX) + "_fit.json")

OUT_RHO_PNG = Path(
    str(PREFIX) + "_charge_density.png"
)

OUT_PHI_PNG = Path(
    str(PREFIX) + "_potential_vs_height.png"
)

OUT_NORM_PNG = Path(
    str(PREFIX) + "_normalized_debye_fit.png"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

DZ_NM = 0.10

Z_MIN_NM = 0.0
# Derived later from the actual periodic box length.
# We use complete 0.1 nm bins up to the top of the cell.
Z_MAX_NM = None

# Gauge/reference for electrostatic potential.
BULK_REFERENCE_LO_NM = 20.0
BULK_REFERENCE_HI_NM = 30.0

# Fixed BEFORE seeing the result.
# Excludes the strongest molecular contact layer.
FIT_LO_NM = 0.8
FIT_HI_NM = 3.0

# Same bulk region used for our previous Debye calculation.
DEBYE_BULK_LO_NM = 5.0
DEBYE_BULK_HI_NM = 30.0


EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
KB = 1.380649e-23
NA = 6.02214076e23

T_K = 300.0
EPS_R_BULK = 77.9

M_TO_NUMBER_DENSITY_NM3 = 0.602214076


# IMPORTANT:
#
# We use epsilon_r = 1 when reconstructing phi from the
# explicit molecular charge density.
#
# OPC water polarization is ALREADY represented by its
# molecular partial charges. Using epsilon_r=77.9 here would
# count water dielectric screening twice.
EPS_R_POISSON = 1.0


# ============================================================
# INPUT VALIDATION
# ============================================================

for path in [
    SYSTEM_XML,
    ASSEMBLY_METADATA,
    SCREENING_TRAJECTORY,
    SCREENING_METADATA,
    WATER_TRAJECTORY,
    WATER_METADATA,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


assembly = json.loads(
    ASSEMBLY_METADATA.read_text(
        encoding="utf-8"
    )
)

screen_meta = json.loads(
    SCREENING_METADATA.read_text(
        encoding="utf-8"
    )
)

water_meta = json.loads(
    WATER_METADATA.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# LOAD OPENMM SYSTEM AND ACTUAL PARTICLE CHARGES
# ============================================================

with SYSTEM_XML.open(
    "r",
    encoding="utf-8",
) as f:

    system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


nonbonded = []

for i in range(
    system.getNumForces()
):

    force = system.getForce(i)

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        nonbonded.append(force)


if len(nonbonded) != 1:

    raise RuntimeError(
        "Expected exactly one NonbondedForce; "
        f"found {len(nonbonded)}"
    )


nb = nonbonded[0]

n_particles = (
    system.getNumParticles()
)


charges_e = np.empty(
    n_particles,
    dtype=np.float64,
)

masses_da = np.empty(
    n_particles,
    dtype=np.float64,
)


for i in range(
    n_particles
):

    q, sigma, epsilon = (
        nb.getParticleParameters(i)
    )

    charges_e[i] = (
        q.value_in_unit(
            unit.elementary_charge
        )
    )

    masses_da[i] = (
        system.getParticleMass(i)
        .value_in_unit(
            unit.dalton
        )
    )


# ============================================================
# EXACT SYSTEM INDEX RANGES
# ============================================================

n_graphene = int(
    assembly[
        "graphene_particles"
    ]
)

n_ligand = int(
    assembly[
        "pyrene_peg5_particles"
    ]
)

n_waters = int(
    assembly[
        "opc_waters"
    ]
)

sites_per_water = int(
    assembly[
        "opc_sites_per_water"
    ]
)


if sites_per_water != 4:

    raise RuntimeError(
        "Expected four-site OPC water."
    )


ligand_start = n_graphene
ligand_stop = (
    ligand_start
    +
    n_ligand
)


water_start = ligand_stop

water_stop = (
    water_start
    +
    n_waters
    *
    sites_per_water
)


na_start = int(
    assembly[
        "na_particle_start_index"
    ]
)

na_stop = int(
    assembly[
        "na_particle_end_index_exclusive"
    ]
)

k_start = int(
    assembly[
        "k_particle_start_index"
    ]
)

k_stop = int(
    assembly[
        "k_particle_end_index_exclusive"
    ]
)

cl_start = int(
    assembly[
        "cl_particle_start_index"
    ]
)

cl_stop = int(
    assembly[
        "cl_particle_end_index_exclusive"
    ]
)

h2_start = int(
    assembly[
        "h2po4_particle_start_index"
    ]
)

h2_stop = int(
    assembly[
        "h2po4_particle_end_index_exclusive"
    ]
)

hp_start = int(
    assembly[
        "hpo4_particle_start_index"
    ]
)

hp_stop = int(
    assembly[
        "hpo4_particle_end_index_exclusive"
    ]
)


n_h2 = int(
    assembly[
        "h2po4_ions"
    ]
)

n_hp = int(
    assembly[
        "hpo4_ions"
    ]
)


if water_stop != na_start:

    raise RuntimeError(
        "Water/Na particle boundary mismatch."
    )


if hp_stop != n_particles:

    raise RuntimeError(
        "Final phosphate endpoint "
        "does not equal System particle count."
    )


ligand_q = (
    charges_e[
        ligand_start:
        ligand_stop
    ]
)

na_q = (
    charges_e[
        na_start:
        na_stop
    ]
)

k_q = (
    charges_e[
        k_start:
        k_stop
    ]
)

cl_q = (
    charges_e[
        cl_start:
        cl_stop
    ]
)


h2_q = (
    charges_e[
        h2_start:
        h2_stop
    ]
    .reshape(
        n_h2,
        7,
    )
)

hp_q = (
    charges_e[
        hp_start:
        hp_stop
    ]
    .reshape(
        n_hp,
        6,
    )
)


# ============================================================
# IDENTIFY PHOSPHORUS FROM PARTICLE MASS
# ============================================================

h2_mass = (
    masses_da[
        h2_start:
        h2_stop
    ]
    .reshape(
        n_h2,
        7,
    )
)

hp_mass = (
    masses_da[
        hp_start:
        hp_stop
    ]
    .reshape(
        n_hp,
        6,
    )
)


h2_p_local = int(
    np.argmax(
        h2_mass[0]
    )
)

hp_p_local = int(
    np.argmax(
        hp_mass[0]
    )
)


if not np.all(
    np.argmax(
        h2_mass,
        axis=1,
    )
    ==
    h2_p_local
):

    raise RuntimeError(
        "H2PO4 phosphorus index mismatch."
    )


if not np.all(
    np.argmax(
        hp_mass,
        axis=1,
    )
    ==
    hp_p_local
):

    raise RuntimeError(
        "HPO4 phosphorus index mismatch."
    )


# ============================================================
# OPC CHARGES
# ============================================================

water_q = (
    charges_e[
        water_start:
        water_stop
    ]
    .reshape(
        n_waters,
        4,
    )
)


reference_water_q = (
    water_q[0]
    .copy()
)


max_water_q_error = float(
    np.max(
        np.abs(
            water_q
            -
            reference_water_q[
                None,
                :
            ]
        )
    )
)


if max_water_q_error > 1e-10:

    raise RuntimeError(
        "Water charge pattern "
        "is not identical across OPC waters."
    )


qO, qH1, qH2, qM = (
    reference_water_q
)


opc_meta_q = (
    water_meta[
        "opc_site_charges_e"
    ]
)


for observed, expected, name in [
    (
        qO,
        float(opc_meta_q["O"]),
        "O",
    ),
    (
        qH1,
        float(opc_meta_q["H1"]),
        "H1",
    ),
    (
        qH2,
        float(opc_meta_q["H2"]),
        "H2",
    ),
    (
        qM,
        float(opc_meta_q["M"]),
        "M",
    ),
]:

    if abs(
        observed
        -
        expected
    ) > 1e-8:

        raise RuntimeError(
            f"OPC {name} charge mismatch."
        )


m_meta = (
    water_meta[
        "m_site_reconstruction"
    ]
)


wO = float(
    m_meta["wO"]
)

wH1 = float(
    m_meta["wH1"]
)

wH2 = float(
    m_meta["wH2"]
)


if abs(
    wO
    +
    wH1
    +
    wH2
    -
    1.0
) > 1e-12:

    raise RuntimeError(
        "OPC M-site weights "
        "do not sum to one."
    )


# ============================================================
# LOAD TRAJECTORIES
# ============================================================

screen = np.load(
    SCREENING_TRAJECTORY,
    mmap_mode="r",
)

water = np.load(
    WATER_TRAJECTORY,
    mmap_mode="r",
)


expected_water_shape = tuple(
    int(x)
    for x
    in water_meta[
        "matrix_shape"
    ]
)


if tuple(
    water.shape
) != expected_water_shape:

    raise RuntimeError(
        "Water trajectory shape mismatch: "
        f"{water.shape} vs "
        f"{expected_water_shape}"
    )


layout = (
    screen_meta[
        "layout"
    ]
)


def extract_screening(
    name,
    rows=None,
):

    info = layout[name]

    a = int(
        info[
            "start_column_inclusive"
        ]
    )

    b = int(
        info[
            "end_column_exclusive"
        ]
    )

    shape = tuple(
        int(x)
        for x
        in info["shape"]
    )


    if rows is None:

        data = np.asarray(
            screen[
                :,
                a:b
            ],
            dtype=np.float64,
        )

        n = (
            screen.shape[0]
        )

    else:

        data = np.asarray(
            screen[
                rows,
                a:b
            ],
            dtype=np.float64,
        )

        n = len(rows)


    return data.reshape(
        (n,)
        +
        shape
    )


screen_dt_ps = float(
    screen_meta[
        "frame_interval_ps"
    ]
)

water_dt_ps = float(
    water_meta[
        "frame_interval_ps"
    ]
)


ratio = (
    water_dt_ps
    /
    screen_dt_ps
)

ratio_int = int(
    round(ratio)
)


if abs(
    ratio
    -
    ratio_int
) > 1e-12:

    raise RuntimeError(
        "Screening/water frame "
        "intervals do not align."
    )


water_rows = (
    np.arange(
        water.shape[0],
        dtype=int,
    )
    *
    ratio_int
)


water_time_ps = (
    extract_screening(
        "production_elapsed_ps",
        water_rows,
    )[:, 0]
)


expected_time_ps = (
    np.arange(
        water.shape[0],
        dtype=float,
    )
    *
    water_dt_ps
)


if not np.allclose(
    water_time_ps,
    expected_time_ps,
    atol=1e-5,
):

    raise RuntimeError(
        "Water/screening clocks "
        "are not aligned."
    )


graphene = (
    extract_screening(
        "graphene_z_summary_nm",
        water_rows,
    )
)


# mean, min, max
graphene_zmax = (
    graphene[:, 2]
)


ligand_xyz = (
    extract_screening(
        "ligand_xyz_nm",
        water_rows,
    )
)

na_xyz = (
    extract_screening(
        "na_xyz_nm",
        water_rows,
    )
)

k_xyz = (
    extract_screening(
        "k_xyz_nm",
        water_rows,
    )
)

cl_xyz = (
    extract_screening(
        "cl_xyz_nm",
        water_rows,
    )
)


h2_xyz = (
    extract_screening(
        "h2po4_xyz_nm",
        water_rows,
    )
    .reshape(
        water.shape[0],
        n_h2,
        7,
        3,
    )
)


hp_xyz = (
    extract_screening(
        "hpo4_xyz_nm",
        water_rows,
    )
    .reshape(
        water.shape[0],
        n_hp,
        6,
        3,
    )
)


# ============================================================
# BOX / HISTOGRAM GEOMETRY
# ============================================================

box = np.asarray(
    screen_meta[
        "box_vectors_nm"
    ],
    dtype=float,
)


area_nm2 = float(
    np.linalg.norm(
        np.cross(
            box[0],
            box[1],
        )
    )
)


# Use the actual periodic z length rather than an arbitrary
# 36 nm ceiling. Keep only complete DZ_NM-wide bins.
LZ_NM = float(
    box[2, 2]
)

Z_MAX_NM = float(
    np.floor(
        LZ_NM
        /
        DZ_NM
    )
    *
    DZ_NM
)

if Z_MAX_NM <= Z_MIN_NM:
    raise RuntimeError(
        "Invalid z analysis domain."
    )

print(
    f"Periodic box Lz: {LZ_NM:.6f} nm"
)

print(
    f"Charge-density domain: "
    f"{Z_MIN_NM:.1f}-{Z_MAX_NM:.1f} nm"
)

edges_nm = np.arange(
    Z_MIN_NM,
    Z_MAX_NM + 0.5 * DZ_NM,
    DZ_NM,
)


centers_nm = (
    0.5
    *
    (
        edges_nm[:-1]
        +
        edges_nm[1:]
    )
)


n_bins = len(
    centers_nm
)


bin_volume_nm3 = (
    area_nm2
    *
    DZ_NM
)


# ============================================================
# PLANAR CHARGE DENSITY rho_q(z)
# ============================================================

rho_frames = np.empty(
    (
        water.shape[0],
        n_bins,
    ),
    dtype=np.float64,
)


outside_sites = 0
min_gap_seen = np.inf
max_gap_seen = -np.inf


def track_gap(gap):

    global outside_sites
    global min_gap_seen
    global max_gap_seen

    gap = np.asarray(
        gap,
        dtype=np.float64,
    ).ravel()

    min_gap_seen = min(
        min_gap_seen,
        float(
            np.min(gap)
        ),
    )

    max_gap_seen = max(
        max_gap_seen,
        float(
            np.max(gap)
        ),
    )

    outside_sites += int(
        np.sum(
            (gap < Z_MIN_NM)
            |
            (gap >= Z_MAX_NM)
        )
    )

    return gap


def add_weighted(
    hist,
    gap,
    charge,
):

    gap = track_gap(gap)

    charge = np.asarray(
        charge,
        dtype=np.float64,
    ).ravel()

    if len(gap) != len(charge):

        raise RuntimeError(
            "Gap/charge length mismatch."
        )

    h, _ = np.histogram(
        gap,
        bins=edges_nm,
        weights=charge,
    )

    hist += h


def add_uniform(
    hist,
    gap,
    charge,
):

    gap = track_gap(gap)

    counts, _ = np.histogram(
        gap,
        bins=edges_nm,
    )

    hist += (
        charge
        *
        counts
    )


for frame in range(
    water.shape[0]
):

    g = float(
        graphene_zmax[
            frame
        ]
    )

    charge_hist = np.zeros(
        n_bins,
        dtype=np.float64,
    )


    # ----------------------------
    # Ligand
    # ----------------------------

    add_weighted(
        charge_hist,
        (
            ligand_xyz[
                frame,
                :,
                2,
            ]
            -
            g
        ),
        ligand_q,
    )


    # ----------------------------
    # Monatomic ions
    # ----------------------------

    add_weighted(
        charge_hist,
        na_xyz[
            frame,
            :,
            2,
        ]
        -
        g,
        na_q,
    )

    add_weighted(
        charge_hist,
        k_xyz[
            frame,
            :,
            2,
        ]
        -
        g,
        k_q,
    )

    add_weighted(
        charge_hist,
        cl_xyz[
            frame,
            :,
            2,
        ]
        -
        g,
        cl_q,
    )


    # ----------------------------
    # Phosphates
    # all atomic partial charges
    # ----------------------------

    add_weighted(
        charge_hist,
        h2_xyz[
            frame,
            :,
            :,
            2,
        ].ravel()
        -
        g,
        h2_q.ravel(),
    )

    add_weighted(
        charge_hist,
        hp_xyz[
            frame,
            :,
            :,
            2,
        ].ravel()
        -
        g,
        hp_q.ravel(),
    )


    # ----------------------------
    # OPC water
    # Stored: O, H1, H2
    # Reconstruct charged M site.
    # ----------------------------

    w = np.asarray(
        water[frame],
        dtype=np.float64,
    )

    zO = w[
        :,
        0,
        2
    ]

    zH1 = w[
        :,
        1,
        2
    ]

    zH2 = w[
        :,
        2,
        2
    ]


    zM = (
        wO * zO
        +
        wH1 * zH1
        +
        wH2 * zH2
    )


    # OPC O is normally zero-charge.
    if abs(qO) > 1e-15:

        add_uniform(
            charge_hist,
            zO - g,
            qO,
        )


    add_uniform(
        charge_hist,
        zH1 - g,
        qH1,
    )

    add_uniform(
        charge_hist,
        zH2 - g,
        qH2,
    )

    add_uniform(
        charge_hist,
        zM - g,
        qM,
    )


    rho_frames[
        frame
    ] = (
        charge_hist
        /
        bin_volume_nm3
    )


    if (
        frame % 100 == 0
        or
        frame
        ==
        water.shape[0] - 1
    ):

        print(
            f"Charge-density frames: "
            f"{frame + 1}/"
            f"{water.shape[0]}",
            flush=True,
        )


if outside_sites != 0:

    raise RuntimeError(
        f"{outside_sites} charged sites "
        "fell outside the analysis region. "
        f"Observed gap range = "
        f"{min_gap_seen:.6f} to "
        f"{max_gap_seen:.6f} nm"
    )


# ============================================================
# FIVE 2 ns BLOCKS
# ============================================================

def block_masks(
    time_ps
):

    result = []

    for block in range(5):

        lo = (
            block
            *
            2000.0
        )

        hi = (
            (block + 1)
            *
            2000.0
        )


        if block < 4:

            mask = (
                (time_ps >= lo)
                &
                (time_ps < hi)
            )

        else:

            mask = (
                (time_ps >= lo)
                &
                (time_ps <= hi)
            )


        result.append(
            mask
        )


    return result


masks = block_masks(
    water_time_ps
)


rho_mean = np.mean(
    rho_frames,
    axis=0,
)


rho_blocks = np.stack(
    [
        np.mean(
            rho_frames[
                mask
            ],
            axis=0,
        )
        for mask
        in masks
    ],
    axis=0,
)


rho_block_sd = np.std(
    rho_blocks,
    axis=0,
    ddof=1,
)


# ============================================================
# POISSON:
#
# dE/dz = rho / epsilon0
# E = -dphi/dz
#
# Boundary:
# E(top vacuum) = 0
#
# Gauge:
# average phi from 20-30 nm = 0
# ============================================================

bulk_reference_mask = (
    (
        centers_nm
        >=
        BULK_REFERENCE_LO_NM
    )
    &
    (
        centers_nm
        <
        BULK_REFERENCE_HI_NM
    )
)


def poisson(
    rho_e_nm3
):

    rho_C_m3 = (
        rho_e_nm3
        *
        E_CHARGE
        *
        1e27
    )

    dz_m = (
        DZ_NM
        *
        1e-9
    )


    E_edges = np.zeros(
        n_bins + 1,
        dtype=np.float64,
    )


    for i in range(
        n_bins - 1,
        -1,
        -1,
    ):

        E_edges[i] = (
            E_edges[i + 1]
            -
            (
                rho_C_m3[i]
                *
                dz_m
                /
                (
                    EPS0
                    *
                    EPS_R_POISSON
                )
            )
        )


    phi_edges = np.zeros(
        n_bins + 1,
        dtype=np.float64,
    )


    for i in range(
        n_bins - 1,
        -1,
        -1,
    ):

        E_mid = (
            0.5
            *
            (
                E_edges[i]
                +
                E_edges[i + 1]
            )
        )


        phi_edges[i] = (
            phi_edges[i + 1]
            +
            E_mid
            *
            dz_m
        )


    phi = (
        0.5
        *
        (
            phi_edges[:-1]
            +
            phi_edges[1:]
        )
    )


    E_center = (
        0.5
        *
        (
            E_edges[:-1]
            +
            E_edges[1:]
        )
    )


    # Bulk = zero potential reference.
    phi -= np.mean(
        phi[
            bulk_reference_mask
        ]
    )


    return (
        phi,
        E_center,
        E_edges,
    )


phi_mean_V, E_mean, E_edges_mean = (
    poisson(
        rho_mean
    )
)


phi_blocks_V = []

E_blocks = []


for block_rho in rho_blocks:

    phi_b, E_b, _ = (
        poisson(
            block_rho
        )
    )

    phi_blocks_V.append(
        phi_b
    )

    E_blocks.append(
        E_b
    )


phi_blocks_V = np.stack(
    phi_blocks_V,
    axis=0,
)

E_blocks = np.stack(
    E_blocks,
    axis=0,
)


phi_block_sd_V = np.std(
    phi_blocks_V,
    axis=0,
    ddof=1,
)


# ============================================================
# INDEPENDENT BULK DEBYE LENGTH
# FROM FULL 1 ps SCREENING TRAJECTORY
# ============================================================

screen_graphene = (
    extract_screening(
        "graphene_z_summary_nm"
    )[:, 2]
)


screen_na = (
    extract_screening(
        "na_xyz_nm"
    )
)

screen_k = (
    extract_screening(
        "k_xyz_nm"
    )
)

screen_cl = (
    extract_screening(
        "cl_xyz_nm"
    )
)


screen_h2 = (
    extract_screening(
        "h2po4_xyz_nm"
    )
    .reshape(
        screen.shape[0],
        n_h2,
        7,
        3,
    )
)


screen_hp = (
    extract_screening(
        "hpo4_xyz_nm"
    )
    .reshape(
        screen.shape[0],
        n_hp,
        6,
        3,
    )
)


bulk_height_nm = (
    DEBYE_BULK_HI_NM
    -
    DEBYE_BULK_LO_NM
)


bulk_volume_nm3 = (
    area_nm2
    *
    bulk_height_nm
)


def concentration_M(
    z_nm
):

    gap = (
        z_nm
        -
        screen_graphene[
            :,
            None
        ]
    )


    count = np.sum(
        (
            gap
            >=
            DEBYE_BULK_LO_NM
        )
        &
        (
            gap
            <
            DEBYE_BULK_HI_NM
        ),
        axis=1,
    )


    return (
        count
        /
        (
            M_TO_NUMBER_DENSITY_NM3
            *
            bulk_volume_nm3
        )
    )


c_na = concentration_M(
    screen_na[
        :,
        :,
        2,
    ]
)

c_k = concentration_M(
    screen_k[
        :,
        :,
        2,
    ]
)

c_cl = concentration_M(
    screen_cl[
        :,
        :,
        2,
    ]
)

c_h2 = concentration_M(
    screen_h2[
        :,
        :,
        h2_p_local,
        2,
    ]
)

c_hp = concentration_M(
    screen_hp[
        :,
        :,
        hp_p_local,
        2,
    ]
)


I_frames = (
    0.5
    *
    (
        c_na
        +
        c_k
        +
        c_cl
        +
        c_h2
        +
        4.0
        *
        c_hp
    )
)


I_mean = float(
    np.mean(
        I_frames
    )
)


def debye_nm(
    I_M
):

    return float(
        np.sqrt(
            (
                EPS_R_BULK
                *
                EPS0
                *
                KB
                *
                T_K
            )
            /
            (
                2.0
                *
                NA
                *
                E_CHARGE**2
                *
                1000.0
                *
                I_M
            )
        )
        *
        1e9
    )


bulk_debye_nm = (
    debye_nm(
        I_mean
    )
)


# ============================================================
# FREE EXPONENTIAL FIT
#
# phi(z) = A exp(-z/lambda_eff) + C
#
# lambda is NOT fixed to the bulk Debye value.
#
# For each trial lambda, A and C are solved by linear
# least-squares. No SciPy dependency required.
# ============================================================

fit_mask = (
    (
        centers_nm
        >=
        FIT_LO_NM
    )
    &
    (
        centers_nm
        <=
        FIT_HI_NM
    )
)


z_fit = (
    centers_nm[
        fit_mask
    ]
)


def fit_exponential(
    z,
    y,
):

    z = np.asarray(
        z,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )


    best = None

    lo = 0.10
    hi = 5.00


    for refinement in range(4):

        lambdas = np.linspace(
            lo,
            hi,
            4000,
        )


        for lam in lambdas:

            x = np.exp(
                -z
                /
                lam
            )


            X = np.column_stack(
                [
                    x,
                    np.ones_like(x),
                ]
            )


            A, C = (
                np.linalg.lstsq(
                    X,
                    y,
                    rcond=None,
                )[0]
            )


            prediction = (
                A * x
                +
                C
            )


            sse = float(
                np.sum(
                    (
                        y
                        -
                        prediction
                    )**2
                )
            )


            if (
                best is None
                or
                sse
                <
                best["sse"]
            ):

                best = {
                    "lambda_nm":
                        float(lam),

                    "A_V":
                        float(A),

                    "C_V":
                        float(C),

                    "sse":
                        sse,
                }


        step = (
            (hi - lo)
            /
            (
                len(lambdas)
                -
                1
            )
        )


        lo = max(
            0.10,
            best[
                "lambda_nm"
            ]
            -
            5.0
            *
            step,
        )


        hi = min(
            5.00,
            best[
                "lambda_nm"
            ]
            +
            5.0
            *
            step,
        )


    prediction = (
        best["A_V"]
        *
        np.exp(
            -z
            /
            best[
                "lambda_nm"
            ]
        )
        +
        best["C_V"]
    )


    ss_total = float(
        np.sum(
            (
                y
                -
                np.mean(y)
            )**2
        )
    )


    if ss_total > 0:

        r2 = (
            1.0
            -
            best["sse"]
            /
            ss_total
        )

    else:

        r2 = np.nan


    best[
        "r2"
    ] = float(r2)


    return best


fit = fit_exponential(
    z_fit,
    phi_mean_V[
        fit_mask
    ],
)


block_fits = []


for block in range(5):

    block_fit = (
        fit_exponential(
            z_fit,
            phi_blocks_V[
                block,
                fit_mask
            ],
        )
    )


    block_fit[
        "block"
    ] = (
        block + 1
    )

    block_fit[
        "start_ps"
    ] = (
        block
        *
        2000.0
    )

    block_fit[
        "end_ps"
    ] = (
        (block + 1)
        *
        2000.0
    )


    block_fits.append(
        block_fit
    )


block_lambdas = np.asarray(
    [
        x[
            "lambda_nm"
        ]
        for x
        in block_fits
    ],
    dtype=float,
)


lambda_block_sd = float(
    np.std(
        block_lambdas,
        ddof=1,
    )
)


fit_curve_V = (
    fit["A_V"]
    *
    np.exp(
        -centers_nm
        /
        fit[
            "lambda_nm"
        ]
    )
    +
    fit["C_V"]
)


# ============================================================
# CHARGE SANITY CHECK
# ============================================================

graphene_charge_e = float(
    np.sum(
        charges_e[
            :
            ligand_start
        ]
    )
)


solution_charge_e = float(
    np.sum(
        charges_e[
            ligand_start:
        ]
    )
)


system_charge_e = float(
    np.sum(
        charges_e
    )
)


histogram_charge_e = float(
    np.sum(
        rho_mean
        *
        bin_volume_nm3
    )
)


# ============================================================
# SAVE CSV
# ============================================================

data = np.column_stack(
    [
        centers_nm,
        rho_mean,
        rho_block_sd,
        phi_mean_V * 1000.0,
        phi_block_sd_V * 1000.0,
        E_mean,
        fit_curve_V * 1000.0,
    ]
)


np.savetxt(
    OUT_CSV,
    data,
    delimiter=",",
    header=(
        "z_nm,"
        "rho_mean_e_per_nm3,"
        "rho_2ns_block_sd_e_per_nm3,"
        "phi_mean_mV_relative_to_bulk,"
        "phi_2ns_block_sd_mV,"
        "E_mean_V_per_m,"
        "exponential_fit_mV"
    ),
    comments="",
)


# ============================================================
# FIGURE 1:
# CHARGE DENSITY
# ============================================================

mask5 = (
    centers_nm
    <=
    5.0
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    centers_nm[
        mask5
    ],
    rho_mean[
        mask5
    ],
    label="MD mean",
)


plt.fill_between(
    centers_nm[
        mask5
    ],
    (
        rho_mean
        -
        rho_block_sd
    )[
        mask5
    ],
    (
        rho_mean
        +
        rho_block_sd
    )[
        mask5
    ],
    alpha=0.2,
    label="2 ns block SD",
)


plt.axhline(
    0.0,
    linewidth=1,
)


plt.axvline(
    bulk_debye_nm,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Bulk Debye length = "
        f"{bulk_debye_nm:.3f} nm"
    ),
)


plt.xlabel(
    "Height above max graphene C z (nm)"
)

plt.ylabel(
    "Solution charge density (e / nm³)"
)

plt.title(
    "Replica B explicit-water planar charge density"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_RHO_PNG,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 2:
# ACTUAL MD POTENTIAL HISTOGRAM
# ============================================================

mask3 = (
    centers_nm
    <=
    3.0
)


bar_x = (
    centers_nm[
        mask3
    ]
)

bar_y = (
    phi_mean_V[
        mask3
    ]
    *
    1000.0
)

bar_sd = (
    phi_block_sd_V[
        mask3
    ]
    *
    1000.0
)


plt.figure(
    figsize=(11, 6.5)
)


plt.bar(
    bar_x,
    bar_y,
    width=(
        DZ_NM
        *
        0.82
    ),
    alpha=0.55,
    label=(
        "MD mean potential "
        "(0.1 nm bins)"
    ),
)


plt.errorbar(
    bar_x,
    bar_y,
    yerr=bar_sd,
    fmt="none",
    capsize=2,
    linewidth=1,
    label="2 ns block SD",
)


plt.plot(
    centers_nm[
        fit_mask
    ],
    (
        fit_curve_V[
            fit_mask
        ]
        *
        1000.0
    ),
    linewidth=2.3,
    label=(
        "Free exponential fit: "
        f"lambda_eff = "
        f"{fit['lambda_nm']:.3f} nm, "
        f"R² = "
        f"{fit['r2']:.3f}"
    ),
)


plt.axhline(
    0.0,
    linewidth=1,
)


plt.axvline(
    bulk_debye_nm,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Bulk lambda_D = "
        f"{bulk_debye_nm:.3f} nm"
    ),
)


plt.axvspan(
    FIT_LO_NM,
    FIT_HI_NM,
    alpha=0.08,
    label=(
        f"Fixed fit window "
        f"{FIT_LO_NM:.1f}-"
        f"{FIT_HI_NM:.1f} nm"
    ),
)


plt.xlabel(
    "Height above max graphene C z (nm)"
)

plt.ylabel(
    "Mean electrostatic potential "
    "relative to bulk (mV)"
)

plt.title(
    "Replica B MD-derived solution-side potential vs height"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_PHI_PNG,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 3:
# NORMALIZED EXPONENTIAL / DEBYE-FORM VIEW
# ============================================================

A_fit = fit[
    "A_V"
]

C_fit = fit[
    "C_V"
]


if abs(
    A_fit
) < 1e-15:

    raise RuntimeError(
        "Fitted exponential amplitude "
        "is effectively zero."
    )


normalized_phi = (
    (
        phi_mean_V
        -
        C_fit
    )
    /
    A_fit
)


normalized_sd = (
    phi_block_sd_V
    /
    abs(A_fit)
)


ideal_curve = np.exp(
    -centers_nm
    /
    fit[
        "lambda_nm"
    ]
)


plt.figure(
    figsize=(11, 6.5)
)


plt.bar(
    bar_x,
    normalized_phi[
        mask3
    ],
    width=(
        DZ_NM
        *
        0.82
    ),
    alpha=0.55,
    label=(
        "MD potential normalized "
        "by fitted A and C"
    ),
)


plt.errorbar(
    bar_x,
    normalized_phi[
        mask3
    ],
    yerr=normalized_sd[
        mask3
    ],
    fmt="none",
    capsize=2,
    linewidth=1,
)


plt.plot(
    centers_nm[
        fit_mask
    ],
    ideal_curve[
        fit_mask
    ],
    linewidth=2.3,
    label=(
        "exp(-z/lambda_eff), "
        f"lambda_eff = "
        f"{fit['lambda_nm']:.3f} nm"
    ),
)


plt.axhline(
    np.exp(-1.0),
    linestyle="-.",
    linewidth=1.3,
    label="1/e",
)


plt.axvline(
    fit[
        "lambda_nm"
    ],
    linestyle=":",
    linewidth=1.5,
    label=(
        f"Fitted lambda_eff = "
        f"{fit['lambda_nm']:.3f} nm"
    ),
)


plt.axvline(
    bulk_debye_nm,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Bulk lambda_D = "
        f"{bulk_debye_nm:.3f} nm"
    ),
)


plt.axvspan(
    FIT_LO_NM,
    FIT_HI_NM,
    alpha=0.08,
)


plt.xlabel(
    "Height above max graphene C z (nm)"
)

plt.ylabel(
    "(phi(z) - C) / A"
)

plt.title(
    "Replica B normalized MD potential and exponential fit"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_NORM_PNG,
    dpi=220,
)

plt.close()


# ============================================================
# FIT QUALITY
# ============================================================

if fit[
    "r2"
] >= 0.90:

    fit_quality = "strong"

elif fit[
    "r2"
] >= 0.70:

    fit_quality = "moderate"

else:

    fit_quality = "weak"


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

summary = {

    "status":
        "PASS",

    "scope":
        (
            "Solution-side planar electrostatic potential "
            "from explicit OPC water, pyrene-PEG5 ligand, "
            "and all explicit DPBS ions."
        ),

    "graphene_reference":
        (
            "framewise maximum "
            "graphene-carbon z"
        ),

    "poisson_relative_permittivity":
        EPS_R_POISSON,

    "poisson_note":
        (
            "epsilon_r=1 because OPC molecular charges "
            "explicitly represent water polarization."
        ),

    "bin_width_nm":
        DZ_NM,

    "bulk_potential_reference_nm":
        [
            BULK_REFERENCE_LO_NM,
            BULK_REFERENCE_HI_NM,
        ],

    "fit_model":
        (
            "phi(z) = "
            "A*exp(-z/lambda_eff) + C"
        ),

    "fit_window_nm":
        [
            FIT_LO_NM,
            FIT_HI_NM,
        ],

    "fit": {

        "lambda_eff_nm":
            fit[
                "lambda_nm"
            ],

        "lambda_eff_2ns_block_sd_nm":
            lambda_block_sd,

        "A_mV":
            fit[
                "A_V"
            ]
            *
            1000.0,

        "C_mV":
            fit[
                "C_V"
            ]
            *
            1000.0,

        "r2":
            fit[
                "r2"
            ],

        "quality":
            fit_quality,

        "blocks":
            block_fits,
    },

    "bulk_electrolyte": {

        "ionic_strength_M":
            I_mean,

        "debye_length_nm":
            bulk_debye_nm,

        "temperature_K":
            T_K,

        "relative_permittivity":
            EPS_R_BULK,

        "bulk_region_nm":
            [
                DEBYE_BULK_LO_NM,
                DEBYE_BULK_HI_NM,
            ],
    },

    "phosphate_validation": {

        "H2PO4_P_local_index":
            h2_p_local,

        "HPO4_P_local_index":
            hp_p_local,

        "H2PO4_charge_e":
            float(
                np.sum(
                    h2_q[0]
                )
            ),

        "HPO4_charge_e":
            float(
                np.sum(
                    hp_q[0]
                )
            ),
    },

    "opc_validation": {

        "O_H1_H2_M_charges_e":
            reference_water_q.tolist(),

        "M_site_weights":
            [
                wO,
                wH1,
                wH2,
            ],
    },

    "charge_sanity": {

        "graphene_charge_e":
            graphene_charge_e,

        "solution_charge_e":
            solution_charge_e,

        "total_system_charge_e":
            system_charge_e,

        "histogram_solution_charge_e":
            histogram_charge_e,

        "minimum_gap_nm":
            float(
                min_gap_seen
            ),

        "maximum_gap_nm":
            float(
                max_gap_seen
            ),
    },

    "interpretation_guardrail":
        (
            "lambda_eff is a diagnostic exponential fit "
            "to the equilibrium solution-side potential. "
            "It should not automatically be called the "
            "Debye length unless the fit is strong and "
            "agrees with the independently calculated "
            "bulk Debye length."
        ),

    "outputs": {

        "csv":
            str(
                OUT_CSV
            ),

        "charge_density_png":
            str(
                OUT_RHO_PNG
            ),

        "potential_png":
            str(
                OUT_PHI_PNG
            ),

        "normalized_fit_png":
            str(
                OUT_NORM_PNG
            ),
    },
}


OUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()

print(
    "=" * 78
)

print(
    "REPLICA B EXPLICIT-WATER "
    "ELECTROSTATIC POTENTIAL ANALYSIS"
)

print(
    "=" * 78
)


print()
print(
    "Charge / topology validation:"
)

print(
    f"  System total charge:       "
    f"{system_charge_e:+.9e} e"
)

print(
    f"  Graphene charge:           "
    f"{graphene_charge_e:+.9e} e"
)

print(
    f"  Solution charge:           "
    f"{solution_charge_e:+.9e} e"
)

print(
    f"  Histogram solution charge: "
    f"{histogram_charge_e:+.9e} e"
)

print(
    f"  H2PO4 P local index:       "
    f"{h2_p_local}"
)

print(
    f"  HPO4 P local index:        "
    f"{hp_p_local}"
)


print()
print(
    "Independent bulk screening:"
)

print(
    f"  Ionic strength: "
    f"{I_mean:.6f} M"
)

print(
    f"  Bulk lambda_D:  "
    f"{bulk_debye_nm:.6f} nm"
)


print()
print(
    "MD potential exponential fit:"
)

print(
    f"  Fixed fit window: "
    f"{FIT_LO_NM:.1f}-"
    f"{FIT_HI_NM:.1f} nm"
)

print(
    f"  lambda_eff: "
    f"{fit['lambda_nm']:.6f} nm"
)

print(
    f"  block SD:   "
    f"{lambda_block_sd:.6f} nm"
)

print(
    f"  A:          "
    f"{fit['A_V'] * 1000:+.6f} mV"
)

print(
    f"  C:          "
    f"{fit['C_V'] * 1000:+.6f} mV"
)

print(
    f"  R^2:        "
    f"{fit['r2']:.6f}"
)

print(
    f"  quality:    "
    f"{fit_quality}"
)


print()
print(
    "2 ns block lambda_eff:"
)


for b in block_fits:

    print(
        f"  "
        f"{b['start_ps']/1000:.0f}-"
        f"{b['end_ps']/1000:.0f} ns: "
        f"lambda_eff="
        f"{b['lambda_nm']:.6f} nm, "
        f"R^2="
        f"{b['r2']:.6f}"
    )


print()
print(
    "Saved:"
)

print(
    " ",
    OUT_CSV,
)

print(
    " ",
    OUT_JSON,
)

print(
    " ",
    OUT_RHO_PNG,
)

print(
    " ",
    OUT_PHI_PNG,
)

print(
    " ",
    OUT_NORM_PNG,
)


print()
print(
    "REPLICA B EXPLICIT-WATER "
    "POTENTIAL ANALYSIS: PASS"
)
