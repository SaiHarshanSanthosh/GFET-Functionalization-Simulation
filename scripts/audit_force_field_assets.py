from pathlib import Path
import json
import sys

import numpy as np


# ============================================================
# FORCE-FIELD ASSET AUDIT
#
# PURPOSE
# -------
# Find everything already created in this repository that could
# be needed to assemble the real:
#
#   graphene + Pyrene-PEG5 + OPC water
#
# OpenMM system.
#
# THIS SCRIPT:
#
#   - runs NO MD
#   - modifies NO files
#   - changes NO parameters
#
# It only inspects the repository.
# ============================================================


ROOT = Path(".").resolve()


# ============================================================
# 1. DIRECTORIES TO IGNORE
# ============================================================

IGNORE_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "env",
    "venv",
    ".venv",
    "site-packages",
}


# ============================================================
# 2. FILE TYPES WE CARE ABOUT
# ============================================================

INTERESTING_EXTENSIONS = {
    ".xml",
    ".mol2",
    ".frcmod",
    ".prmtop",
    ".parm7",
    ".inpcrd",
    ".rst7",
    ".lib",
    ".off",
    ".dat",
    ".json",
    ".npy",
    ".pdb",
    ".xyz",
    ".gro",
    ".top",
    ".itp",
    ".py",
    ".txt",
}


# ============================================================
# 3. KEYWORDS
# ============================================================

NAME_KEYWORDS = [
    "pyrene",
    "peg5",
    "graphene",
    "iff",
    "gaff",
    "amber",
    "opc",
    "water",
    "combined",
    "nonbond",
    "forcefield",
    "force_field",
]


CONTENT_KEYWORDS = [
    "NonbondedForce",
    "CustomNonbondedForce",
    "sigma",
    "epsilon",
    "charge",
    "GAFF",
    "GAFF2",
    "antechamber",
    "parmchk2",
    "tleap",
    "OPC",
    "cg1",
    "cge",
    "pyrene",
    "PEG5",
]


# ============================================================
# 4. HELPERS
# ============================================================

def ignored(path: Path) -> bool:
    return any(
        part in IGNORE_DIR_NAMES
        for part in path.parts
    )


def rel(path: Path) -> str:
    try:
        return str(
            path.relative_to(ROOT)
        )
    except Exception:
        return str(path)


def looks_relevant_by_name(path: Path) -> bool:
    lower = path.name.lower()

    return any(
        keyword.lower() in lower
        for keyword in NAME_KEYWORDS
    )


def safe_read_text(
    path: Path,
    max_bytes: int = 2_000_000,
):
    """
    Read small-ish text files only.

    Returns:
        text or None
    """

    try:
        if path.stat().st_size > max_bytes:
            return None

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        return None


def looks_relevant_by_content(
    path: Path,
):
    text = safe_read_text(path)

    if text is None:
        return False

    lower = text.lower()

    return any(
        keyword.lower() in lower
        for keyword in CONTENT_KEYWORDS
    )


# ============================================================
# 5. FIND ALL INTERESTING FILES
# ============================================================

all_interesting = []

for path in ROOT.rglob("*"):

    if not path.is_file():
        continue

    if ignored(path):
        continue

    if path.suffix.lower() not in INTERESTING_EXTENSIONS:
        continue

    all_interesting.append(path)


all_interesting.sort(
    key=lambda p: rel(p).lower()
)


print()
print("=" * 72)
print("FORCE-FIELD ASSET AUDIT")
print("=" * 72)

print()
print("Repository:")
print(ROOT)

print()
print(
    "Interesting files found:",
    len(all_interesting),
)


# ============================================================
# 6. FIND PROBABLY RELEVANT FILES
# ============================================================

relevant = []

for path in all_interesting:

    relevant_name = (
        looks_relevant_by_name(path)
    )

    relevant_content = False

    # Only inspect likely text-based formats.
    if path.suffix.lower() in {
        ".xml",
        ".mol2",
        ".frcmod",
        ".lib",
        ".off",
        ".dat",
        ".json",
        ".py",
        ".txt",
        ".top",
        ".itp",
    }:
        relevant_content = (
            looks_relevant_by_content(path)
        )

    if relevant_name or relevant_content:
        relevant.append(path)


print()
print("=" * 72)
print("PROBABLY RELEVANT FILES")
print("=" * 72)

if not relevant:

    print("NONE FOUND")

else:

    for path in relevant:
        print(rel(path))


# ============================================================
# 7. CHECK EXPECTED FILES FROM OUR CURRENT BUILD
# ============================================================

expected_paths = [

    Path(
        "parameters/iff/"
        "graphene_iff_bonded_oop.xml"
    ),

    Path(
        "parameters/combined/"
        "pyrene_peg5_graphene_supported_positions_nm.npy"
    ),

    Path(
        "parameters/combined/"
        "one_sided_opc_water_positions_nm.npy"
    ),

    Path(
        "analysis/"
        "graphene_support_restraint_validation.json"
    ),

    Path(
        "analysis/"
        "supported_slab_boundary_validation.json"
    ),
]


print()
print("=" * 72)
print("KNOWN CURRENT BUILD FILES")
print("=" * 72)

for path in expected_paths:

    status = (
        "FOUND"
        if path.exists()
        else "MISSING"
    )

    print(
        f"{status:8s}  {path}"
    )


# ============================================================
# 8. INSPECT NPY FILES
# ============================================================

print()
print("=" * 72)
print("NUMPY ASSET SHAPES")
print("=" * 72)

npy_files = [
    p
    for p in relevant
    if p.suffix.lower() == ".npy"
]

# Also explicitly include our known arrays.
for p in expected_paths:

    if (
        p.suffix.lower() == ".npy"
        and p.exists()
        and p not in npy_files
    ):
        npy_files.append(p)


npy_files = sorted(
    set(npy_files),
    key=lambda p: rel(p).lower(),
)


if not npy_files:

    print("No relevant .npy files found.")

else:

    for path in npy_files:

        try:

            arr = np.load(
                path,
                allow_pickle=False,
            )

            finite = (
                np.isfinite(arr).all()
                if np.issubdtype(
                    arr.dtype,
                    np.number,
                )
                else "N/A"
            )

            print()
            print(rel(path))
            print(
                "  shape:",
                arr.shape,
            )
            print(
                "  dtype:",
                arr.dtype,
            )
            print(
                "  finite:",
                finite,
            )

        except Exception as exc:

            print()
            print(rel(path))
            print(
                "  ERROR:",
                repr(exc),
            )


# ============================================================
# 9. INSPECT OPENMM XML SYSTEMS
# ============================================================

print()
print("=" * 72)
print("OPENMM XML SYSTEM INSPECTION")
print("=" * 72)


try:

    from openmm import openmm

    openmm_available = True

except Exception as exc:

    openmm_available = False

    print(
        "Could not import OpenMM:",
        repr(exc),
    )


xml_files = [
    p
    for p in relevant
    if p.suffix.lower() == ".xml"
]


if not xml_files:

    print(
        "No relevant XML files found."
    )


elif openmm_available:

    for path in xml_files:

        print()
        print("-" * 72)
        print(rel(path))

        text = safe_read_text(
            path,
            max_bytes=20_000_000,
        )

        if text is None:

            print(
                "  Too large/unreadable "
                "for XML inspection."
            )

            continue

        # ----------------------------------------------------
        # Simple string checks first
        # ----------------------------------------------------

        has_nonbonded_string = (
            "NonbondedForce"
            in text
        )

        has_custom_nonbonded_string = (
            "CustomNonbondedForce"
            in text
        )

        print(
            "  Contains NonbondedForce text:",
            has_nonbonded_string,
        )

        print(
            "  Contains CustomNonbondedForce text:",
            has_custom_nonbonded_string,
        )

        # ----------------------------------------------------
        # Try deserializing as an OpenMM System
        # ----------------------------------------------------

        try:

            obj = (
                openmm.XmlSerializer
                .deserialize(text)
            )

        except Exception as exc:

            print(
                "  Not an OpenMM serialized System "
                "or could not deserialize."
            )

            print(
                "  Deserialize message:",
                str(exc)[:300],
            )

            continue

        if not isinstance(
            obj,
            openmm.System,
        ):

            print(
                "  XML deserialized, "
                "but object is not an OpenMM System."
            )

            print(
                "  Type:",
                type(obj),
            )

            continue

        system = obj

        print(
            "  PARTICLES:",
            system.getNumParticles(),
        )

        print(
            "  CONSTRAINTS:",
            system.getNumConstraints(),
        )

        print(
            "  FORCES:",
            system.getNumForces(),
        )

        print(
            "  Force list:"
        )

        nonbonded_count = 0

        for i in range(
            system.getNumForces()
        ):

            force = system.getForce(i)

            class_name = (
                force.__class__.__name__
            )

            try:
                force_name = force.getName()
            except Exception:
                force_name = ""

            print(
                f"    [{i}] "
                f"{class_name}"
                f" | name={force_name}"
            )

            if class_name in {
                "NonbondedForce",
                "CustomNonbondedForce",
            }:
                nonbonded_count += 1

        print(
            "  Nonbonded-type forces:",
            nonbonded_count,
        )


# ============================================================
# 10. SEARCH TEXT FILES FOR IMPORTANT PARAMETER TERMS
# ============================================================

print()
print("=" * 72)
print("KEY PARAMETER / BUILD REFERENCES")
print("=" * 72)


text_extensions = {
    ".py",
    ".txt",
    ".mol2",
    ".frcmod",
    ".dat",
    ".json",
    ".top",
    ".itp",
    ".xml",
}


hits_by_file = {}


for path in relevant:

    if path.suffix.lower() not in text_extensions:
        continue

    text = safe_read_text(path)

    if text is None:
        continue

    lines = text.splitlines()

    file_hits = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        lower = line.lower()

        matching_keywords = [
            keyword
            for keyword in CONTENT_KEYWORDS
            if keyword.lower() in lower
        ]

        if matching_keywords:

            clean_line = line.strip()

            if len(clean_line) > 180:
                clean_line = (
                    clean_line[:177]
                    + "..."
                )

            file_hits.append(
                (
                    line_number,
                    matching_keywords,
                    clean_line,
                )
            )

    if file_hits:
        hits_by_file[path] = file_hits


for path, hits in hits_by_file.items():

    print()
    print("-" * 72)
    print(rel(path))

    # Avoid gigantic output.
    for (
        line_number,
        matching_keywords,
        clean_line,
    ) in hits[:30]:

        keywords_text = ",".join(
            matching_keywords
        )

        print(
            f"  L{line_number:<5d} "
            f"[{keywords_text}] "
            f"{clean_line}"
        )

    if len(hits) > 30:

        print(
            f"  ... {len(hits) - 30} "
            "additional matching lines omitted"
        )


# ============================================================
# 11. SPECIFIC AMBER / GAFF FILE INVENTORY
# ============================================================

print()
print("=" * 72)
print("AMBER / GAFF PARAMETER FILE INVENTORY")
print("=" * 72)


amber_extensions = {
    ".mol2",
    ".frcmod",
    ".prmtop",
    ".parm7",
    ".inpcrd",
    ".rst7",
    ".lib",
    ".off",
}


amber_files = [
    p
    for p in all_interesting
    if p.suffix.lower()
    in amber_extensions
]


if not amber_files:

    print(
        "NO Amber/GAFF parameter files "
        "found in repository."
    )

else:

    for path in amber_files:

        print(
            f"{path.suffix.lower():8s} "
            f"{rel(path)}"
        )


# ============================================================
# 12. CHECK FOR GRAPHENE NONBONDED EVIDENCE
# ============================================================

print()
print("=" * 72)
print("GRAPHENE NONBONDED EVIDENCE")
print("=" * 72)


graphene_candidates = []

for path in relevant:

    name_lower = path.name.lower()

    if (
        "graphene" in name_lower
        or "iff" in name_lower
    ):

        graphene_candidates.append(
            path
        )


found_graphene_nonbonded = False


for path in graphene_candidates:

    text = safe_read_text(path)

    if text is None:
        continue

    lower = text.lower()

    evidence = []

    for keyword in [
        "nonbondedforce",
        "customnonbondedforce",
        "epsilon",
        "sigma",
        "lj",
        "lennard",
        "vdw",
        "charge",
    ]:

        if keyword in lower:
            evidence.append(keyword)

    if evidence:

        found_graphene_nonbonded = True

        print()
        print(rel(path))

        print(
            "  evidence:",
            ", ".join(evidence),
        )


if not found_graphene_nonbonded:

    print(
        "NO obvious graphene nonbonded "
        "parameter evidence found."
    )


# ============================================================
# 13. SUMMARY
# ============================================================

print()
print("=" * 72)
print("AUDIT SUMMARY")
print("=" * 72)


print(
    "Relevant files:",
    len(relevant),
)

print(
    "Amber/GAFF parameter files:",
    len(amber_files),
)

print(
    "Graphene nonbonded evidence:",
    (
        "FOUND"
        if found_graphene_nonbonded
        else "NOT FOUND"
    ),
)


print()
print(
    "FORCE-FIELD ASSET AUDIT: COMPLETE"
)

print()
print(
    "No MD was run."
)

print(
    "No files were modified."
)

print(
    "No force-field parameters were changed."
)
