from ase.build import graphene
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Rebuild our graphene sheet exactly as before
# --------------------------------------------------
sheet = graphene(
    formula="C2",
    a=2.46,
    size=(25, 25, 1),
    vacuum=20.0
)

sheet.pbc = (True, True, False)

carbon_positions = sheet.get_positions()
cell = sheet.cell.array

n_carbon = len(carbon_positions)

# --------------------------------------------------
# IFF 2017 graphitic model
#
# Each graphene carbon (cg1):
#   charge = +0.2 e
#   mass   = 10.011 amu
#
# Two pi-electron pseudo-particles (cge) per carbon:
#   charge = -0.1 e each
#   mass   = 1.000 amu each
#   equilibrium C-pi distance = 0.65 Angstrom
# --------------------------------------------------

pi_distance = 0.65  # Angstrom

# Normal vector to the graphene plane
normal = np.cross(cell[0], cell[1])
normal = normal / np.linalg.norm(normal)

pi_above = carbon_positions + pi_distance * normal
pi_below = carbon_positions - pi_distance * normal

# --------------------------------------------------
# Sanity checks
# --------------------------------------------------

n_pi = 2 * n_carbon
n_total = n_carbon + n_pi

total_charge = (
    n_carbon * (+0.2)
    + n_pi * (-0.1)
)

mass_per_graphene_carbon = (
    10.011
    + 2 * 1.000
)

print("Graphene carbon atoms:", n_carbon)
print("IFF pi-electron sites:", n_pi)
print("Total IFF particles:", n_total)

print()
print("Total graphene charge:", total_charge, "e")
print(
    "Mass per C + two pi sites:",
    mass_per_graphene_carbon,
    "amu"
)

# --------------------------------------------------
# Save a visualization-only XYZ
#
# X = pseudo-particle, NOT a real chemical element.
# This file is only for looking at the geometry.
# --------------------------------------------------

Path("structures").mkdir(exist_ok=True)

output = "structures/graphene_iff_sites.xyz"

with open(output, "w") as f:
    f.write(f"{n_total}\n")
    f.write(
        "IFF graphene: C = cg1 carbon core; "
        "X = cge virtual pi-electron pseudo-particle\n"
    )

    for x, y, z in carbon_positions:
        f.write(f"C {x:.6f} {y:.6f} {z:.6f}\n")

    for x, y, z in pi_above:
        f.write(f"X {x:.6f} {y:.6f} {z:.6f}\n")

    for x, y, z in pi_below:
        f.write(f"X {x:.6f} {y:.6f} {z:.6f}\n")

print()
print("Saved:", output)
