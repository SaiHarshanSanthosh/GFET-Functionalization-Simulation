from pathlib import Path
import numpy as np


INPUT = Path(
    "parameters/gaff2/pyrene_peg5_gaff2.mol2"
)

OUTPUT = Path(
    "parameters/gaff2/pyrene_peg5_gaff2_neutral.mol2"
)


# ============================================================
# READ MOL2
# ============================================================

lines = INPUT.read_text().splitlines(
    keepends=True
)

atom_line_indices = []
charges = []

in_atoms = False


for line_index, line in enumerate(lines):

    stripped = line.strip()

    if stripped == "@<TRIPOS>ATOM":
        in_atoms = True
        continue

    if (
        in_atoms
        and stripped.startswith("@<TRIPOS>")
    ):
        break

    if in_atoms and stripped:

        fields = stripped.split()

        if len(fields) < 9:
            raise RuntimeError(
                f"Unexpected MOL2 atom line:\n{line}"
            )

        atom_line_indices.append(
            line_index
        )

        charges.append(
            float(fields[8])
        )


charges = np.asarray(
    charges,
    dtype=float
)


# ============================================================
# CHECK INPUT
# ============================================================

n_atoms = len(charges)

if n_atoms != 70:
    raise RuntimeError(
        f"Expected 70 atoms, found {n_atoms}."
    )


original_total = float(
    charges.sum()
)


print(
    "Atoms:",
    n_atoms
)

print(
    f"Original total charge: "
    f"{original_total:+.12f} e"
)


# ============================================================
# NORMALIZE TO EXACT FORMAL CHARGE = 0
#
# We spread the tiny rounding residual uniformly across all
# atoms instead of dumping the entire correction onto one atom.
#
# This preserves relative charge differences as closely as
# possible.
# ============================================================

TARGET_CHARGE = 0.0

residual = (
    original_total
    - TARGET_CHARGE
)

correction_per_atom = (
    residual
    / n_atoms
)


corrected = (
    charges
    - correction_per_atom
)


print(
    f"Residual to remove: "
    f"{residual:+.12f} e"
)

print(
    f"Correction per atom: "
    f"{correction_per_atom:+.12f} e"
)


# ============================================================
# WRITE NEW MOL2
#
# Use 10 decimal places so we do not immediately recreate
# the same rounding problem.
# ============================================================

for atom_number, line_index in enumerate(
    atom_line_indices
):

    fields = (
        lines[line_index]
        .strip()
        .split()
    )

    fields[8] = (
        f"{corrected[atom_number]:.10f}"
    )

    lines[line_index] = (
        " ".join(fields)
        + "\n"
    )


OUTPUT.write_text(
    "".join(lines)
)


# ============================================================
# READ BACK AND VERIFY WHAT WAS ACTUALLY WRITTEN
# ============================================================

written_charges = []

in_atoms = False


for line in OUTPUT.read_text().splitlines():

    stripped = line.strip()

    if stripped == "@<TRIPOS>ATOM":
        in_atoms = True
        continue

    if (
        in_atoms
        and stripped.startswith("@<TRIPOS>")
    ):
        break

    if in_atoms and stripped:

        fields = stripped.split()

        written_charges.append(
            float(fields[8])
        )


written_total = float(
    np.sum(written_charges)
)


print()

print(
    f"Written total charge: "
    f"{written_total:+.12f} e"
)


if abs(written_total) > 1e-8:

    raise RuntimeError(
        "Charge normalization failed."
    )


print(
    "Charge normalization: PASS"
)

print(
    "Saved:",
    OUTPUT
)

print()

print(
    "Original MOL2 was NOT modified."
)
