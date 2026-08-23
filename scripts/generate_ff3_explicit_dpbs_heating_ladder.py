from pathlib import Path

template_path = Path(r"scripts\heat_ff3_explicit_dpbs_opc_yb_100K.py")
template = template_path.read_text()

stages = [
    (150, 100, 20260821, 135.0, 165.0),
    (200, 150, 20260822, 185.0, 215.0),
    (250, 200, 20260823, 235.0, 265.0),
    (300, 250, 20260824, 285.0, 315.0),
]

for target, prior, seed, low, high in stages:
    s = template

    # Protect prior-stage references from collisions with target outputs.
    s = s.replace(
        "heated_050K_positions_nm.npy",
        "__PRIOR_POSITIONS__",
        1,
    )
    s = s.replace(
        "heated_050K_velocities_nm_per_ps.npy",
        "__PRIOR_VELOCITIES__",
        1,
    )
    s = s.replace(
        "heating_050K.json",
        "__PRIOR_METADATA__",
        1,
    )
    s = s.replace(
        '"HEATING_050K_PASS"',
        '"__PRIOR_PASS__"',
        1,
    )
    s = s.replace(
        "HEATING_050K_PASS input stage.",
        "__PRIOR_PASS_MESSAGE__",
        1,
    )

    # Target-stage physics.
    s = s.replace(
        "TARGET_TEMPERATURE_K = 100.0",
        f"TARGET_TEMPERATURE_K = {target}.0",
        1,
    )
    s = s.replace(
        "LANGEVIN_SEED = 20260820",
        f"LANGEVIN_SEED = {seed}",
        1,
    )

    # Target-stage outputs.
    s = s.replace(
        "heated_100K_positions_nm.npy",
        f"heated_{target:03d}K_positions_nm.npy",
        1,
    )
    s = s.replace(
        "heated_100K_velocities_nm_per_ps.npy",
        f"heated_{target:03d}K_velocities_nm_per_ps.npy",
        1,
    )
    s = s.replace(
        "heating_100K.chk",
        f"heating_{target:03d}K.chk",
        1,
    )
    s = s.replace(
        "heating_100K_log.csv",
        f"heating_{target:03d}K_log.csv",
        1,
    )
    s = s.replace(
        "heating_100K.json",
        f"heating_{target:03d}K.json",
        1,
    )

    # Acceptance window.
    s = s.replace(
        "85.0\n    <=\n    final_metrics",
        f"{low:.1f}\n    <=\n    final_metrics",
        1,
    )
    s = s.replace(
        "<=\n    115.0",
        f"<=\n    {high:.1f}",
        1,
    )

    # Target result status.
    s = s.replace(
        '"HEATING_100K_PASS"',
        f'"HEATING_{target:03d}K_PASS"',
        1,
    )
    s = s.replace(
        '"HEATING_100K_COMPLETE_INSPECT_WARNINGS"',
        f'"HEATING_{target:03d}K_COMPLETE_INSPECT_WARNINGS"',
        1,
    )

    # Stage labels/reporting.
    lines = s.splitlines()
    for i, line in enumerate(lines):
        if "STAGED NVT HEATING" in line and "STAGE 2" in line:
            lines[i] = (
                line.replace("STAGE 2", f"STAGE {target // 50}")
                    .replace("100 K", f"{target} K")
            )
    s = "\n".join(lines) + "\n"

    s = s.replace("#   100 K", f"#   {target} K", 1)
    s = s.replace(
        "# 10. CREATE 100 K LANGEVIN INTEGRATOR",
        f"# 10. CREATE {target} K LANGEVIN INTEGRATOR",
        1,
    )
    s = s.replace(
        "# 13. CONTINUE 50 K VELOCITIES",
        f"# 13. CONTINUE {prior} K VELOCITIES",
        1,
    )
    s = s.replace(
        "completed 50 K stage",
        f"completed {prior} K stage",
        1,
    )
    s = s.replace(
        "INITIAL STATE ENTERING 100 K STAGE",
        f"INITIAL STATE ENTERING {target} K STAGE",
        1,
    )
    s = s.replace(
        "# 17. RUN 20 ps AT 100 K",
        f"# 17. RUN 20 ps AT {target} K",
        1,
    )
    s = s.replace(
        "RUNNING 100 K NVT",
        f"RUNNING {target} K NVT",
        1,
    )
    s = s.replace(
        "100 K HEATING SUMMARY",
        f"{target} K HEATING SUMMARY",
        1,
    )
    s = s.replace(
        "100 K NVT HEATING: COMPLETE",
        f"{target} K NVT HEATING: COMPLETE",
        1,
    )
    s = s.replace(
        '"100 K"',
        f'"{target} K"',
        1,
    )

    # Restore correct prior-stage references.
    s = s.replace(
        "__PRIOR_POSITIONS__",
        f"heated_{prior:03d}K_positions_nm.npy",
        1,
    )
    s = s.replace(
        "__PRIOR_VELOCITIES__",
        f"heated_{prior:03d}K_velocities_nm_per_ps.npy",
        1,
    )
    s = s.replace(
        "__PRIOR_METADATA__",
        f"heating_{prior:03d}K.json",
        1,
    )
    s = s.replace(
        '"__PRIOR_PASS__"',
        f'"HEATING_{prior:03d}K_PASS"',
        1,
    )
    s = s.replace(
        "__PRIOR_PASS_MESSAGE__",
        f"HEATING_{prior:03d}K_PASS input stage.",
        1,
    )
    s = s.replace(
        "100 K heating requires a successful ",
        f"{target} K heating requires a successful ",
        1,
    )

    next_temp = target + 50
    s = s.replace(
        "Do not start 100 K until this output is inspected.",
        (
            f"Do not start {next_temp} K until this output is inspected."
            if target < 300
            else "Do not start production MD until this output is inspected."
        ),
        1,
    )

    # Semantic sanity checks before writing.
    required = [
        f"TARGET_TEMPERATURE_K = {target}.0",
        f"heated_{prior:03d}K_positions_nm.npy",
        f"heated_{prior:03d}K_velocities_nm_per_ps.npy",
        f"HEATING_{prior:03d}K_PASS",
        f"heated_{target:03d}K_positions_nm.npy",
        f"heated_{target:03d}K_velocities_nm_per_ps.npy",
        f"HEATING_{target:03d}K_PASS",
        f"{low:.1f}",
        f"{high:.1f}",
    ]

    for item in required:
        if item not in s:
            raise RuntimeError(
                f"{target} K missing expected content: {item}"
            )

    if "setVelocitiesToTemperature(" in s:
        raise RuntimeError(
            f"{target} K unexpectedly re-randomizes velocities"
        )

    out = Path(
        rf"scripts\heat_ff3_explicit_dpbs_opc_yb_{target}K.py"
    )

    if out.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing stage script: {out}"
        )

    out.write_text(s)
    print(f"{target} K stage generated safely")

print("EXPLICIT_DPBS_150_TO_300_STAGE_GENERATION_COMPLETE")
