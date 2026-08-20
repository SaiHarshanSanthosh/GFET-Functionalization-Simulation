from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


CSV = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_0to8ns_merged.csv"
)

JSON = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_0to8ns_adsorption_summary.json"
)

OUT = Path(
    "analysis/pubfigs_final"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Load validated analysis
# ------------------------------------------------------------

df = pd.read_csv(CSV)

with open(
    JSON,
    "r",
    encoding="utf-8",
) as f:
    summary = json.load(f)


df = df.sort_values(
    "time_ps"
).reset_index(drop=True)

df["time_ns"] = (
    df["time_ps"] / 1000.0
)


H = "pyrene_height_above_graphene_nm"
TILT = "pyrene_tilt_from_graphene_plane_deg"


stable = summary[
    "first_stable_faceon_adsorption"
]

contact = summary[
    "first_sustained_contact"
]


first_cross_ns = (
    summary["first_raw_contact_ps"]
    / 1000.0
)

contact_onset_ns = (
    contact["onset_ps"]
    / 1000.0
)

contact_confirm_ns = (
    contact["confirmation_ps"]
    / 1000.0
)

ads_onset_ns = (
    stable["onset_ps"]
    / 1000.0
)

ads_confirm_ns = (
    stable["confirmation_ps"]
    / 1000.0
)


post = df[
    df["time_ns"] >= ads_onset_ns
].copy()

pre = df[
    df["time_ns"] < ads_onset_ns
].copy()


# ------------------------------------------------------------
# Typography
# ------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def polish(ax):
    ax.grid(
        True,
        alpha=0.18,
        linewidth=0.7,
    )

    ax.tick_params(
        direction="out",
        length=4,
        width=0.9,
    )


def export(fig, stem):

    fig.savefig(
        OUT / f"{stem}.png",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
    )


# ============================================================
# FIGURE 1 — MAIN TWO-PANEL TRAJECTORY
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.4, 6.2),
    sharex=True,
)

ax_h, ax_t = axes


# --- A: Height

ax_h.plot(
    df["time_ns"],
    df[H],
    linewidth=1.05,
)

ax_h.axhline(
    0.45,
    linestyle="--",
    linewidth=1.15,
    label="Contact criterion (0.45 nm)",
)

ax_h.axhline(
    0.332,
    linestyle=":",
    linewidth=1.25,
    label="FF-2 bare-pyrene minimum (0.332 nm)",
)

ax_h.axvline(
    ads_onset_ns,
    linestyle="--",
    linewidth=1.15,
)

ax_h.axvspan(
    ads_onset_ns,
    8.0,
    alpha=0.10,
)

ax_h.set_ylabel(
    "Pyrene–graphene\nseparation (nm)"
)

ax_h.set_ylim(
    0.15,
    2.8,
)

ax_h.text(
    -0.055,
    1.015,
    "(A)",
    transform=ax_h.transAxes,
    fontsize=12,
    fontweight="bold",
    va="bottom",
    ha="left",
    clip_on=False,
)

ax_h.text(
    ads_onset_ns + 0.08,
    2.68,
    "stable adsorbed regime",
    fontsize=9,
    va="top",
)

ax_h.legend(
    loc="upper left",
    frameon=True,
)

polish(ax_h)


# --- B: Tilt

ax_t.plot(
    df["time_ns"],
    df[TILT],
    linewidth=1.0,
)

ax_t.axhline(
    30.0,
    linestyle="--",
    linewidth=1.15,
    label="Face-on criterion (30°)",
)

ax_t.axvline(
    ads_onset_ns,
    linestyle="--",
    linewidth=1.15,
    label=f"Adsorption onset ({ads_onset_ns:.3f} ns)",
)

ax_t.axvspan(
    ads_onset_ns,
    8.0,
    alpha=0.10,
)

ax_t.set_xlabel(
    "Simulation time (ns)"
)

ax_t.set_ylabel(
    "Pyrene tilt (°)"
)

ax_t.set_xlim(
    0,
    8,
)

ax_t.set_ylim(
    -2,
    95,
)

ax_t.text(
    -0.055,
    1.015,
    "(B)",
    transform=ax_t.transAxes,
    fontsize=12,
    fontweight="bold",
    va="bottom",
    ha="left",
    clip_on=False,
)

ax_t.legend(
    loc="upper left",
    frameon=True,
)

polish(ax_t)


fig.suptitle(
    "Spontaneous Pyrene–PEG5 Adsorption onto Graphene",
    fontsize=13,
    y=0.995,
)

fig.tight_layout()

export(
    fig,
    "drop01_main_adsorption_trajectory",
)

plt.close(fig)


# ============================================================
# FIGURE 2 — HEIGHT/TILT STATE SPACE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.6, 5.3)
)


ax.scatter(
    pre[H],
    pre[TILT],
    s=8,
    alpha=0.22,
    rasterized=True,
    label="Pre-adsorption",
)

ax.scatter(
    post[H],
    post[TILT],
    s=10,
    alpha=0.60,
    rasterized=True,
    label="Post-onset adsorbed state",
)


ax.axvline(
    0.45,
    linestyle="--",
    linewidth=1.15,
)

ax.axhline(
    30.0,
    linestyle="--",
    linewidth=1.15,
)


ax.text(
    0.465,
    91,
    "h = 0.45 nm",
    fontsize=8.5,
)

ax.text(
    2.38,
    31.5,
    "tilt = 30°",
    fontsize=8.5,
    ha="right",
)


ax.set_xlabel(
    "Pyrene–graphene separation (nm)"
)

ax.set_ylabel(
    "Pyrene tilt (°)"
)

ax.set_title(
    "Configurational State Space During Adsorption"
)


ax.set_xlim(
    0.25,
    2.75,
)

ax.set_ylim(
    -3,
    95,
)


ax.legend(
    loc="upper right",
    frameon=True,
)


polish(ax)

fig.tight_layout()

export(
    fig,
    "drop01_height_tilt_state_space",
)

plt.close(fig)


# ============================================================
# FIGURE 3 — CAPTURE TRANSITION
# ============================================================

zoom = df[
    (
        df["time_ns"] >= 4.60
    )
    &
    (
        df["time_ns"] <= 5.20
    )
].copy()


fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.4, 5.9),
    sharex=True,
)

ax_h, ax_t = axes


# --- Height

ax_h.plot(
    zoom["time_ns"],
    zoom[H],
    linewidth=1.2,
)

ax_h.axhline(
    0.45,
    linestyle="--",
    linewidth=1.1,
)

ax_h.axhline(
    0.332,
    linestyle=":",
    linewidth=1.1,
)

ax_h.set_ylabel(
    "Separation (nm)"
)

ax_h.text(
    0.015,
    0.93,
    "(A)",
    transform=ax_h.transAxes,
    fontweight="bold",
)

polish(ax_h)


# --- Tilt

ax_t.plot(
    zoom["time_ns"],
    zoom[TILT],
    linewidth=1.2,
)

ax_t.axhline(
    30,
    linestyle="--",
    linewidth=1.1,
)

ax_t.set_ylabel(
    "Tilt (°)"
)

ax_t.set_xlabel(
    "Simulation time (ns)"
)

ax_t.text(
    0.015,
    0.93,
    "(B)",
    transform=ax_t.transAxes,
    fontweight="bold",
)

polish(ax_t)


# Event markers on both axes

events = [
    (
        first_cross_ns,
        "Raw contact  4.961 ns",
        ":",
    ),
    (
        ads_onset_ns,
        "Adsorption onset  4.966 ns",
        "--",
    ),
    (
        contact_confirm_ns,
        "20 ps contact confirmed  4.986 ns",
        "-.",
    ),
    (
        ads_confirm_ns,
        "50 ps stable confirmed  5.016 ns",
        (0, (3, 1, 1, 1)),
    ),
]


for ax in axes:

    ax.axvspan(
        ads_onset_ns,
        5.20,
        alpha=0.07,
    )

    for x, _, style in events:

        ax.axvline(
            x,
            linewidth=1.0,
            linestyle=style,
            alpha=0.9,
        )


# Add event legend using dummy handles.
for x, label, style in events:

    ax_h.plot(
        [],
        [],
        linestyle=style,
        linewidth=1.1,
        label=label,
    )


ax_h.legend(
    loc="upper left",
    bbox_to_anchor=(0.0, -0.02),
    fontsize=8,
    frameon=True,
    ncol=2,
)


fig.suptitle(
    "Surface Capture and Transition into the Adsorbed Basin",
    fontsize=12.5,
)

fig.tight_layout()

export(
    fig,
    "drop01_adsorption_transition",
)

plt.close(fig)


print("=" * 78)
print("DROP-01 FINAL PUBLICATION FIGURES")
print("=" * 78)

for path in sorted(
    OUT.glob("*")
):
    print(path)

print()
print(
    f"First raw contact:       {first_cross_ns:.3f} ns"
)

print(
    f"Stable adsorption onset: {ads_onset_ns:.3f} ns"
)

print(
    f"20 ps confirmation:      {contact_confirm_ns:.3f} ns"
)

print(
    f"50 ps confirmation:      {ads_confirm_ns:.3f} ns"
)

print("=" * 78)
