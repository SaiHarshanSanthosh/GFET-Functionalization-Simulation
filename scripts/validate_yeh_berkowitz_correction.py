from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openmm
from openmm import unit


# ============================================================
# PURPOSE
# ============================================================
#
# Validate whether the observed finite-Lz dependence in the
# matched-grid 3D PME audit is quantitatively explained by the
# Yeh-Berkowitz slab dipole correction.
#
# NO dynamics.
# NO minimization.
# NO files modified except new analysis outputs.
#
# For a neutral system:
#
#   U_YB = 2*pi*k_e*Mz^2 / V
#
# where:
#
#   Mz = sum_i q_i*z_i
#   V  = A_xy * Lz
#
# Force correction:
#
#   Fz_i,YB = -4*pi*k_e*q_i*Mz / V
#
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

INPUT_CSV = (
    ROOT
    / "analysis"
    / "pme_slab_sensitivity_matched_grid_5500ps.csv"
)

SYSTEM_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_supported_pme_production_safe.xml"
)

POSITIONS_NPY = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_production_300K_5500ps_positions_nm.npy"
)

OUTPUT_CSV = (
    ROOT
    / "analysis"
    / "yeh_berkowitz_validation_5500ps.csv"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "yeh_berkowitz_validation_5500ps.json"
)

OUTPUT_FIGURE = (
    ROOT
    / "analysis"
    / "figures"
    / "15_yeh_berkowitz_validation.png"
)


# ============================================================
# CONSTANTS
# ============================================================

# OpenMM-compatible Coulomb conversion:
#
# kJ mol^-1 nm e^-2
#
K_E = 138.935456


EXPECTED_PARTICLES = 28224

LIGAND = np.arange(
    3750,
    3820,
    dtype=int,
)

PYRENE = np.arange(
    3752,
    3768,
    dtype=int,
)

REFERENCE_LZ_NM = 30.0


# ============================================================
# INPUT CHECKS
# ============================================================

print()
print("=" * 80)
print("YEH-BERKOWITZ SLAB-CORRECTION VALIDATION")
print("=" * 80)
print()


for path in [
    INPUT_CSV,
    SYSTEM_XML,
    POSITIONS_NPY,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


print("Matched-grid audit:")
print(f"  {INPUT_CSV}")

print()
print("System:")
print(f"  {SYSTEM_XML}")

print()
print("Coordinates:")
print(f"  {POSITIONS_NPY}")


# ============================================================
# LOAD MATCHED-GRID RESULTS
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

required_columns = [
    "lz_nm",
    "total_potential_kJ_mol",
    "pme_reciprocal_kJ_mol",
    "ligand_reciprocal_Fz_kJ_mol_nm",
    "pyrene_reciprocal_Fz_kJ_mol_nm",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns in matched-grid CSV: {missing}"
    )


df = (
    df
    .sort_values("lz_nm")
    .reset_index(drop=True)
)


# ============================================================
# LOAD SYSTEM / CHARGES
# ============================================================

system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text(
        encoding="utf-8"
    )
)


if system.getNumParticles() != EXPECTED_PARTICLES:
    raise RuntimeError(
        "Unexpected particle count."
    )


nonbonded = None

for i in range(
    system.getNumForces()
):

    force = system.getForce(i)

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):

        if nonbonded is not None:
            raise RuntimeError(
                "More than one NonbondedForce found."
            )

        nonbonded = force


if nonbonded is None:
    raise RuntimeError(
        "No NonbondedForce found."
    )


charges_e = np.zeros(
    EXPECTED_PARTICLES,
    dtype=float,
)


for i in range(
    EXPECTED_PARTICLES
):

    q, sigma, epsilon = (
        nonbonded.getParticleParameters(i)
    )

    charges_e[i] = float(
        q.value_in_unit(
            unit.elementary_charge
        )
    )


total_charge_e = float(
    np.sum(
        charges_e
    )
)

ligand_charge_e = float(
    np.sum(
        charges_e[LIGAND]
    )
)

pyrene_charge_e = float(
    np.sum(
        charges_e[PYRENE]
    )
)


# ============================================================
# POSITIONS / Mz
# ============================================================

positions_nm = np.load(
    POSITIONS_NPY
)


if positions_nm.shape != (
    EXPECTED_PARTICLES,
    3,
):
    raise RuntimeError(
        "Unexpected coordinate array shape."
    )


Mz_e_nm = float(
    np.sum(
        charges_e
        *
        positions_nm[:, 2]
    )
)


# ============================================================
# XY AREA
# ============================================================

a, b, c = (
    system.getDefaultPeriodicBoxVectors()
)


a_nm = np.array(
    [
        a[0].value_in_unit(unit.nanometer),
        a[1].value_in_unit(unit.nanometer),
        a[2].value_in_unit(unit.nanometer),
    ],
    dtype=float,
)

b_nm = np.array(
    [
        b[0].value_in_unit(unit.nanometer),
        b[1].value_in_unit(unit.nanometer),
        b[2].value_in_unit(unit.nanometer),
    ],
    dtype=float,
)


area_nm2 = float(
    np.linalg.norm(
        np.cross(
            a_nm,
            b_nm,
        )
    )
)


print()
print("SYSTEM ELECTROSTATIC DIAGNOSTICS")
print("--------------------------------")

print(
    f"Total charge:      "
    f"{total_charge_e:+.10e} e"
)

print(
    f"Whole ligand Q:    "
    f"{ligand_charge_e:+.8f} e"
)

print(
    f"Pyrene subset Q:   "
    f"{pyrene_charge_e:+.8f} e"
)

print(
    f"Mz:                "
    f"{Mz_e_nm:+.8f} e nm"
)

print(
    f"XY area:           "
    f"{area_nm2:.8f} nm^2"
)


if abs(total_charge_e) > 1.0e-5:
    raise RuntimeError(
        "Yeh-Berkowitz neutral-system expression "
        "should not be applied to this non-neutral system."
    )


# ============================================================
# APPLY ANALYTICAL CORRECTION
# ============================================================

yb_energy = []
yb_ligand_force = []
yb_pyrene_force = []


for _, row in df.iterrows():

    lz_nm = float(
        row["lz_nm"]
    )

    volume_nm3 = (
        area_nm2
        *
        lz_nm
    )


    # --------------------------------------------------------
    # Yeh-Berkowitz energy correction
    # --------------------------------------------------------

    U_corr = (
        2.0
        *
        math.pi
        *
        K_E
        *
        Mz_e_nm**2
        /
        volume_nm3
    )


    # --------------------------------------------------------
    # Group force correction
    #
    # Sum_i Fz_i:
    #
    #  -4*pi*k_e*Mz/V * sum_i(q_i)
    #
    # --------------------------------------------------------

    common_force_factor = (
        -4.0
        *
        math.pi
        *
        K_E
        *
        Mz_e_nm
        /
        volume_nm3
    )


    F_ligand_corr = (
        common_force_factor
        *
        ligand_charge_e
    )

    F_pyrene_corr = (
        common_force_factor
        *
        pyrene_charge_e
    )


    yb_energy.append(
        U_corr
    )

    yb_ligand_force.append(
        F_ligand_corr
    )

    yb_pyrene_force.append(
        F_pyrene_corr
    )


df[
    "yb_energy_correction_kJ_mol"
] = yb_energy

df[
    "yb_ligand_Fz_correction_kJ_mol_nm"
] = yb_ligand_force

df[
    "yb_pyrene_Fz_correction_kJ_mol_nm"
] = yb_pyrene_force


# ============================================================
# CORRECTED VALUES
# ============================================================

df[
    "corrected_total_PE_kJ_mol"
] = (
    df[
        "total_potential_kJ_mol"
    ]
    +
    df[
        "yb_energy_correction_kJ_mol"
    ]
)


df[
    "corrected_PME_recip_kJ_mol"
] = (
    df[
        "pme_reciprocal_kJ_mol"
    ]
    +
    df[
        "yb_energy_correction_kJ_mol"
    ]
)


df[
    "corrected_ligand_recip_Fz_kJ_mol_nm"
] = (
    df[
        "ligand_reciprocal_Fz_kJ_mol_nm"
    ]
    +
    df[
        "yb_ligand_Fz_correction_kJ_mol_nm"
    ]
)


df[
    "corrected_pyrene_recip_Fz_kJ_mol_nm"
] = (
    df[
        "pyrene_reciprocal_Fz_kJ_mol_nm"
    ]
    +
    df[
        "yb_pyrene_Fz_correction_kJ_mol_nm"
    ]
)


# ============================================================
# REFERENCE ROW
# ============================================================

reference_candidates = df[
    np.isclose(
        df["lz_nm"],
        REFERENCE_LZ_NM,
    )
]


if len(reference_candidates) != 1:
    raise RuntimeError(
        "Could not uniquely identify 30 nm reference row."
    )


ref = reference_candidates.iloc[0]


# ============================================================
# DELTAS RELATIVE TO 30 NM
# ============================================================

df[
    "raw_total_delta_vs_30_kJ_mol"
] = (
    df[
        "total_potential_kJ_mol"
    ]
    -
    float(
        ref[
            "total_potential_kJ_mol"
        ]
    )
)


df[
    "yb_predicted_raw_energy_delta_vs_30_kJ_mol"
] = -(
    df[
        "yb_energy_correction_kJ_mol"
    ]
    -
    float(
        ref[
            "yb_energy_correction_kJ_mol"
        ]
    )
)


df[
    "energy_prediction_residual_kJ_mol"
] = (
    df[
        "raw_total_delta_vs_30_kJ_mol"
    ]
    -
    df[
        "yb_predicted_raw_energy_delta_vs_30_kJ_mol"
    ]
)


df[
    "corrected_total_delta_vs_30_kJ_mol"
] = (
    df[
        "corrected_total_PE_kJ_mol"
    ]
    -
    float(
        ref[
            "corrected_total_PE_kJ_mol"
        ]
    )
)


# ------------------------------------------------------------
# PYRENE FORCE
# ------------------------------------------------------------

df[
    "raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"
] = (
    df[
        "pyrene_reciprocal_Fz_kJ_mol_nm"
    ]
    -
    float(
        ref[
            "pyrene_reciprocal_Fz_kJ_mol_nm"
        ]
    )
)


df[
    "yb_predicted_raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"
] = -(
    df[
        "yb_pyrene_Fz_correction_kJ_mol_nm"
    ]
    -
    float(
        ref[
            "yb_pyrene_Fz_correction_kJ_mol_nm"
        ]
    )
)


df[
    "pyrene_force_prediction_residual_kJ_mol_nm"
] = (
    df[
        "raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"
    ]
    -
    df[
        "yb_predicted_raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"
    ]
)


df[
    "corrected_pyrene_Fz_delta_vs_30_kJ_mol_nm"
] = (
    df[
        "corrected_pyrene_recip_Fz_kJ_mol_nm"
    ]
    -
    float(
        ref[
            "corrected_pyrene_recip_Fz_kJ_mol_nm"
        ]
    )
)


# ------------------------------------------------------------
# WHOLE LIGAND FORCE
# ------------------------------------------------------------

df[
    "corrected_ligand_Fz_delta_vs_30_kJ_mol_nm"
] = (
    df[
        "corrected_ligand_recip_Fz_kJ_mol_nm"
    ]
    -
    float(
        ref[
            "corrected_ligand_recip_Fz_kJ_mol_nm"
        ]
    )
)


# ============================================================
# SUMMARY METRICS
# ============================================================

raw_energy_span = float(
    df[
        "raw_total_delta_vs_30_kJ_mol"
    ].max()
    -
    df[
        "raw_total_delta_vs_30_kJ_mol"
    ].min()
)


corrected_energy_span = float(
    df[
        "corrected_total_delta_vs_30_kJ_mol"
    ].max()
    -
    df[
        "corrected_total_delta_vs_30_kJ_mol"
    ].min()
)


max_energy_residual = float(
    np.max(
        np.abs(
            df[
                "energy_prediction_residual_kJ_mol"
            ]
        )
    )
)


raw_pyrene_force_span = float(
    df[
        "raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"
    ].max()
    -
    df[
        "raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"
    ].min()
)


corrected_pyrene_force_span = float(
    df[
        "corrected_pyrene_Fz_delta_vs_30_kJ_mol_nm"
    ].max()
    -
    df[
        "corrected_pyrene_Fz_delta_vs_30_kJ_mol_nm"
    ].min()
)


max_pyrene_force_residual = float(
    np.max(
        np.abs(
            df[
                "pyrene_force_prediction_residual_kJ_mol_nm"
            ]
        )
    )
)


# ============================================================
# PRINT TABLE
# ============================================================

print()
print("=" * 115)
print("ENERGY: OBSERVED 3D-PME ARTIFACT VS YEH-BERKOWITZ PREDICTION")
print("=" * 115)

print(
    " Lz | "
    "Raw dE vs30 | "
    "YB predicted dE | "
    "Residual | "
    "YB correction | "
    "Corrected dE"
)

print("-" * 115)


for _, row in df.iterrows():

    print(
        f"{row['lz_nm']:4.0f} | "
        f"{row['raw_total_delta_vs_30_kJ_mol']:12.6f} | "
        f"{row['yb_predicted_raw_energy_delta_vs_30_kJ_mol']:15.6f} | "
        f"{row['energy_prediction_residual_kJ_mol']:8.6f} | "
        f"{row['yb_energy_correction_kJ_mol']:13.6f} | "
        f"{row['corrected_total_delta_vs_30_kJ_mol']:12.6f}"
    )


print()
print("=" * 115)
print("PYRENE Fz: OBSERVED ARTIFACT VS YEH-BERKOWITZ PREDICTION")
print("=" * 115)

print(
    " Lz | "
    "Raw dFz vs30 | "
    "YB predicted dFz | "
    "Residual | "
    "YB Fcorr | "
    "Corrected dFz"
)

print("-" * 115)


for _, row in df.iterrows():

    print(
        f"{row['lz_nm']:4.0f} | "
        f"{row['raw_pyrene_Fz_delta_vs_30_kJ_mol_nm']:13.6f} | "
        f"{row['yb_predicted_raw_pyrene_Fz_delta_vs_30_kJ_mol_nm']:16.6f} | "
        f"{row['pyrene_force_prediction_residual_kJ_mol_nm']:8.6f} | "
        f"{row['yb_pyrene_Fz_correction_kJ_mol_nm']:9.6f} | "
        f"{row['corrected_pyrene_Fz_delta_vs_30_kJ_mol_nm']:13.6f}"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FIGURE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


df.to_csv(
    OUTPUT_CSV,
    index=False,
)


metadata = {
    "method": (
        "Analytical Yeh-Berkowitz correction validation "
        "against matched-grid 3D PME box-height sensitivity."
    ),

    "formula_energy": (
        "Ucorr = 2*pi*k_e*Mz^2/V"
    ),

    "formula_force": (
        "Fz_i,corr = -4*pi*k_e*q_i*Mz/V"
    ),

    "coulomb_constant_kJ_mol_nm_e2": K_E,

    "total_charge_e": total_charge_e,

    "ligand_charge_e": ligand_charge_e,

    "pyrene_subset_charge_e": pyrene_charge_e,

    "Mz_e_nm": Mz_e_nm,

    "area_nm2": area_nm2,

    "reference_lz_nm": REFERENCE_LZ_NM,

    "raw_energy_span_kJ_mol": raw_energy_span,

    "corrected_energy_span_kJ_mol": corrected_energy_span,

    "max_energy_prediction_residual_kJ_mol": (
        max_energy_residual
    ),

    "raw_pyrene_force_span_kJ_mol_nm": (
        raw_pyrene_force_span
    ),

    "corrected_pyrene_force_span_kJ_mol_nm": (
        corrected_pyrene_force_span
    ),

    "max_pyrene_force_prediction_residual_kJ_mol_nm": (
        max_pyrene_force_residual
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


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.5),
)


# ------------------------------------------------------------
# Energy
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    df["lz_nm"],
    df["raw_total_delta_vs_30_kJ_mol"],
    marker="o",
    linewidth=2,
    label="Raw 3D PME",
)

ax.plot(
    df["lz_nm"],
    df["yb_predicted_raw_energy_delta_vs_30_kJ_mol"],
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Yeh-Berkowitz prediction",
)

ax.plot(
    df["lz_nm"],
    df["corrected_total_delta_vs_30_kJ_mol"],
    marker="^",
    linewidth=2,
    label="After correction",
)

ax.axhline(
    0.0,
    linewidth=1,
)

ax.set_xlabel(
    "Periodic box height Lz (nm)"
)

ax.set_ylabel(
    "Energy difference vs 30 nm (kJ/mol)"
)

ax.set_title(
    "A. Finite-Lz energy artifact"
)

ax.grid(alpha=0.25)
ax.legend(frameon=False)


# ------------------------------------------------------------
# Pyrene force
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    df["lz_nm"],
    df["raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"],
    marker="o",
    linewidth=2,
    label="Raw 3D PME",
)

ax.plot(
    df["lz_nm"],
    df["yb_predicted_raw_pyrene_Fz_delta_vs_30_kJ_mol_nm"],
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Yeh-Berkowitz prediction",
)

ax.plot(
    df["lz_nm"],
    df["corrected_pyrene_Fz_delta_vs_30_kJ_mol_nm"],
    marker="^",
    linewidth=2,
    label="After correction",
)

ax.axhline(
    0.0,
    linewidth=1,
)

ax.set_xlabel(
    "Periodic box height Lz (nm)"
)

ax.set_ylabel(
    "Pyrene reciprocal Fz difference "
    "(kJ/mol/nm)"
)

ax.set_title(
    "B. Finite-Lz pyrene force artifact"
)

ax.grid(alpha=0.25)
ax.legend(frameon=False)


fig.suptitle(
    "Yeh-Berkowitz Validation of the 3D PME Slab Artifact",
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
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("CORRECTION COLLAPSE SUMMARY")
print("=" * 80)

print(
    f"Raw energy span:             "
    f"{raw_energy_span:.6f} kJ/mol"
)

print(
    f"Corrected energy span:       "
    f"{corrected_energy_span:.6f} kJ/mol"
)

print(
    f"Max energy residual:         "
    f"{max_energy_residual:.6f} kJ/mol"
)

print()

print(
    f"Raw pyrene Fz span:          "
    f"{raw_pyrene_force_span:.6f} kJ/mol/nm"
)

print(
    f"Corrected pyrene Fz span:    "
    f"{corrected_pyrene_force_span:.6f} kJ/mol/nm"
)

print(
    f"Max pyrene-force residual:   "
    f"{max_pyrene_force_residual:.6f} kJ/mol/nm"
)

print()

print(
    "Whole-ligand charge:        "
    f"{ligand_charge_e:+.8f} e"
)

print(
    "Pyrene-subset charge:       "
    f"{pyrene_charge_e:+.8f} e"
)

print()

print(f"CSV:    {OUTPUT_CSV}")
print(f"JSON:   {OUTPUT_JSON}")
print(f"Figure: {OUTPUT_FIGURE}")

print()
print(
    "NOTE: This validates the analytical correction against "
    "this static matched-grid test. It is not yet a corrected "
    "production MD trajectory."
)

print()
print("YEH_BERKOWITZ_VALIDATION_COMPLETE")
print("=" * 80)