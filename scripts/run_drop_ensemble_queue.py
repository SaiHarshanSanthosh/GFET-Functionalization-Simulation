from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    /
    "ensemble/drop_ensemble_manifest.csv"
)

STAGE_A_SCRIPT = (
    ROOT
    /
    "scripts/run_drop_replica_stageA_20ps.py"
)

TO_5NS_SCRIPT = (
    ROOT
    /
    "scripts/run_drop_replica_continue_20ps_to_5ns.py"
)

TO_8NS_SCRIPT = (
    ROOT
    /
    "scripts/run_drop_replica_continue_5ns_to_8ns.py"
)

QUEUE_STATE = (
    ROOT
    /
    "analysis/ensemble/drop_ensemble_queue_state.json"
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dry-run",
    action="store_true",
)

args = parser.parse_args()


def dcd_frames(drop):
    path = (
        ROOT
        /
        f"trajectories/ensemble/{drop}_graphene_ligand.dcd"
    )

    if not path.exists():
        return None

    with path.open("rb") as f:
        magic = f.read(8)

        if (
            len(magic) != 8
            or
            magic[4:8] != b"CORD"
        ):
            raise RuntimeError(
                f"{drop}: invalid DCD header."
            )

        return struct.unpack(
            "<i",
            f.read(4),
        )[0]


def metadata_pass(
    path,
    expected_status,
):

    if not path.exists():
        return False

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    return (
        data.get("status")
        ==
        expected_status
    )


def stage_a_complete(drop):

    required = [
        ROOT / f"parameters/ensemble/{drop}_stageA_20ps_positions_nm.npy",
        ROOT / f"parameters/ensemble/{drop}_stageA_20ps_velocities_nm_per_ps.npy",
        ROOT / f"checkpoints/ensemble/{drop}_stageA_20ps.chk",
        ROOT / f"analysis/ensemble/{drop}_stageA_20ps_log.csv",
        ROOT / f"analysis/ensemble/{drop}_stageA_20ps.json",
        ROOT / f"trajectories/ensemble/{drop}_graphene_ligand.dcd",
    ]

    return (
        all(p.exists() for p in required)
        and
        metadata_pass(
            ROOT
            /
            f"analysis/ensemble/{drop}_stageA_20ps.json",
            f"{drop.upper()}_STAGEA_20PS_PASS",
        )
    )


def five_ns_complete(drop):

    required = [
        ROOT / f"parameters/ensemble/{drop}_5ns_positions_nm.npy",
        ROOT / f"parameters/ensemble/{drop}_5ns_velocities_nm_per_ps.npy",
        ROOT / f"checkpoints/ensemble/{drop}_5ns.chk",
        ROOT / f"analysis/ensemble/{drop}_20ps_to_5ns_log.csv",
        ROOT / f"analysis/ensemble/{drop}_5ns.json",
    ]

    return (
        all(p.exists() for p in required)
        and
        metadata_pass(
            ROOT
            /
            f"analysis/ensemble/{drop}_5ns.json",
            f"{drop.upper()}_5NS_CONTINUATION_PASS",
        )
    )


def eight_ns_complete(drop):

    required = [
        ROOT / f"parameters/ensemble/{drop}_8ns_positions_nm.npy",
        ROOT / f"parameters/ensemble/{drop}_8ns_velocities_nm_per_ps.npy",
        ROOT / f"checkpoints/ensemble/{drop}_8ns.chk",
        ROOT / f"analysis/ensemble/{drop}_5ns_to_8ns_log.csv",
        ROOT / f"analysis/ensemble/{drop}_8ns.json",
    ]

    return (
        all(p.exists() for p in required)
        and
        metadata_pass(
            ROOT
            /
            f"analysis/ensemble/{drop}_8ns.json",
            f"{drop.upper()}_8NS_CONTINUATION_PASS",
        )
    )


def save_queue_state(
    drop,
    stage,
    state,
):

    QUEUE_STATE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "timestamp":
            datetime.now().isoformat(),

        "drop_id":
            drop,

        "stage":
            stage,

        "state":
            state,
    }

    QUEUE_STATE.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_stage(
    drop,
    label,
    script,
):

    print()
    print("=" * 78)
    print(
        f"{drop.upper()} :: {label}"
    )
    print("=" * 78)

    save_queue_state(
        drop,
        label,
        "starting",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--drop-id",
            drop,
        ],
        cwd=ROOT,
        check=True,
    )

    save_queue_state(
        drop,
        label,
        "finished",
    )


# ============================================================
# Manifest
# ============================================================

with MANIFEST.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


drops = []

for row in rows:

    drop = (
        row["drop_id"]
        .strip()
        .lower()
    )

    if drop == "drop01":
        continue

    if (
        not drop.startswith("drop")
        or
        not drop[4:].isdigit()
    ):
        raise RuntimeError(
            f"Invalid drop id: {drop}"
        )

    if (
        row["status"]
        .strip()
        .lower()
        !=
        "pending"
    ):
        raise RuntimeError(
            f"{drop}: expected manifest status pending."
        )

    if abs(
        float(row["duration_ns"])
        -
        8.0
    ) > 1.0e-12:
        raise RuntimeError(
            f"{drop}: duration is not locked to 8 ns."
        )

    drops.append(drop)


drops.sort(
    key=lambda x: int(x[4:])
)


if drops != [
    f"drop{i:02d}"
    for i in range(2, 14)
]:
    raise RuntimeError(
        "Expected exactly Drop02-Drop13."
    )


# ============================================================
# Build safe execution plan
# ============================================================

plan = []

for drop in drops:

    a_done = stage_a_complete(drop)
    five_done = five_ns_complete(drop)
    eight_done = eight_ns_complete(drop)

    frames = dcd_frames(drop)

    if eight_done:

        if frames != 8001:
            raise RuntimeError(
                f"{drop}: 8 ns metadata exists but DCD "
                f"has {frames} frames instead of 8001."
            )

        plan.append(
            (drop, "COMPLETE", None)
        )

        continue


    if five_done:

        if not a_done:
            raise RuntimeError(
                f"{drop}: 5 ns complete without valid Stage A."
            )

        if frames != 5001:
            raise RuntimeError(
                f"{drop}: 5 ns state exists but DCD "
                f"has {frames} frames instead of 5001. "
                "Possible interrupted 5->8 ns continuation."
            )

        plan.append(
            (
                drop,
                "5 ns -> 8 ns",
                TO_8NS_SCRIPT,
            )
        )

        continue


    if a_done:

        if frames != 21:
            raise RuntimeError(
                f"{drop}: Stage A is complete but DCD "
                f"has {frames} frames instead of 21. "
                "Possible interrupted 20 ps->5 ns continuation; "
                "refusing to overwrite it."
            )

        plan.append(
            (
                drop,
                "20 ps -> 5 ns",
                TO_5NS_SCRIPT,
            )
        )

        plan.append(
            (
                drop,
                "5 ns -> 8 ns",
                TO_8NS_SCRIPT,
            )
        )

        continue


    if frames is not None:
        raise RuntimeError(
            f"{drop}: incomplete Stage A but a DCD already exists. "
            "Refusing to overwrite possible partial work."
        )

    plan.append(
        (
            drop,
            "Stage A 0 -> 20 ps",
            STAGE_A_SCRIPT,
        )
    )

    plan.append(
        (
            drop,
            "20 ps -> 5 ns",
            TO_5NS_SCRIPT,
        )
    )

    plan.append(
        (
            drop,
            "5 ns -> 8 ns",
            TO_8NS_SCRIPT,
        )
    )


print()
print("=" * 78)
print("DROP ENSEMBLE QUEUE PLAN")
print("=" * 78)

for drop, label, _ in plan:
    print(
        f"{drop}: {label}"
    )


if args.dry_run:

    print()
    print(
        "DRY RUN ONLY: no MD was started."
    )

    sys.exit(0)


# ============================================================
# Execute sequentially on one GPU
# ============================================================

for drop, label, script in plan:

    if label == "COMPLETE":
        print(
            f"SKIP {drop}: already complete at 8 ns."
        )
        continue

    run_stage(
        drop,
        label,
        script,
    )

    if label == "Stage A 0 -> 20 ps":

        if not stage_a_complete(drop):
            raise RuntimeError(
                f"{drop}: Stage A did not validate."
            )

        if dcd_frames(drop) != 21:
            raise RuntimeError(
                f"{drop}: Stage-A DCD frame count invalid."
            )


    elif label == "20 ps -> 5 ns":

        if not five_ns_complete(drop):
            raise RuntimeError(
                f"{drop}: 5 ns continuation did not validate."
            )

        if dcd_frames(drop) != 5001:
            raise RuntimeError(
                f"{drop}: 5 ns DCD frame count invalid."
            )


    elif label == "5 ns -> 8 ns":

        if not eight_ns_complete(drop):
            raise RuntimeError(
                f"{drop}: 8 ns continuation did not validate."
            )

        if dcd_frames(drop) != 8001:
            raise RuntimeError(
                f"{drop}: 8 ns DCD frame count invalid."
            )


print()
print("=" * 78)
print("DROP02-DROP13 QUEUE COMPLETE")
print("=" * 78)
