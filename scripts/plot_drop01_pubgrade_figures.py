from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUTS
# ============================================================

MERGED_CSV = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_0to8ns_merged.csv"
)

SUMMARY_JSON = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_0to8ns_adsorption_summary.json"
)

OUTDIR = Path("analysis/pubfigs")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

if not MERGED_CSV.exists():
    raise FileNotFoundError(f"Missing merged CSV: {MERGED_CSV}")

if not SUMMARY_JSON.exists():
    raise FileNotFoundError(f"Missing summary JSON: {SUMMARY_JSON}")

df = pd.read_csv(MERGED_CSV)

with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
    summary = json.load(f)

required_cols = [
    "time_ps",
    "pyrene_height_above_graphene_nm",
    "pyrene_tilt_from_graphene_plane_deg",
]

for col in required_cols:
    if col not in df.columns:
        raise RuntimeError(f"Required column missing: {col}")

df = df.sort_values("time_ps").reset_index(drop=True)
df["time_ns"] = df["time_ps"] / 1000.0

height_col = "pyrene_height_above_graphene_nm"
tilt_col = "pyrene_tilt_from_graphene_plane_deg"

# These keys are based on your formal analysis output.
# If the JSON uses slightly different names, adjust here once.
stable_event = summary["first_stable_faceon_adsorption"]
contact_event = summary["first_sustained_contact"]

if stable_event is None:
    raise RuntimeError(
        "Formal analysis did not identify stable adsorption."
    )

if contact_event is None:
    raise RuntimeError(
        "Formal analysis did not identify sustained contact."
    )

stable_onset_ps = stable_event["onset_ps"]
stable_confirm_ps = stable_event["confirmation_ps"]
contact_onset_ps = contact_event["onset_ps"]
first_crossing_ps = summary["first_raw_contact_ps"]

stable_onset_ns = stable_onset_ps / 1000.0
stable_confirm_ns = stable_confirm_ps / 1000.0
contact_onset_ns = contact_onset_ps / 1000.0
first_crossing_ns = first_crossing_ps / 1000.0

post_ads = df[df["time_ps"] >= stable_onset_ps].copy()

if len(post_ads) == 0:
    raise RuntimeError("No post-adsorption rows found.")


# ============================================================
# PUBLICATION-STYLE MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
})


def finalize_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="out", length=5, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# ============================================================
# FIGURE 1: HEIGHT VS TIME
# ============================================================

fig = plt.figure(figsize=(8.6, 4.8))
ax = fig.add_subplot(111)

ax.plot(
    df["time_ns"],
    df[height_col],
    label="Pyrene height above graphene",
)

ax.axhline(
    0.45,
    linestyle="--",
    linewidth=1.5,
    label="Contact threshold (0.45 nm)",
)

ax.axhline(
    0.34,
    linestyle=":",
    linewidth=1.5,
    label=r"$\pi$-$\pi$ stacking floor (~0.34 nm)",
)

ax.axvline(
    stable_onset_ns,
    linestyle="--",
    linewidth=1.2,
    label=f"Stable adsorption onset ({stable_onset_ns:.3f} ns)",
)

ax.axvspan(
    stable_onset_ns,
    df["time_ns"].max(),
    alpha=0.15,
    label="Stable adsorbed regime",
)

ax.set_title("Drop-01 adsorption trajectory: pyrene height vs time")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Pyrene height above graphene (nm)")
ax.set_xlim(df["time_ns"].min(), df["time_ns"].max())

finalize_axes(ax)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(OUTDIR / "drop01_height_vs_time.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIGURE 2: TILT VS TIME
# ============================================================

fig = plt.figure(figsize=(8.6, 4.8))
ax = fig.add_subplot(111)

ax.plot(
    df["time_ns"],
    df[tilt_col],
    label="Pyrene tilt from graphene plane",
)

ax.axhline(
    30.0,
    linestyle="--",
    linewidth=1.5,
    label="Face-on criterion (30°)",
)

ax.axvline(
    stable_onset_ns,
    linestyle="--",
    linewidth=1.2,
    label=f"Stable adsorption onset ({stable_onset_ns:.3f} ns)",
)

ax.axvspan(
    stable_onset_ns,
    df["time_ns"].max(),
    alpha=0.15,
    label="Stable adsorbed regime",
)

ax.set_title("Drop-01 adsorption trajectory: pyrene tilt vs time")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Pyrene tilt from graphene plane (deg)")
ax.set_xlim(df["time_ns"].min(), df["time_ns"].max())

finalize_axes(ax)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(OUTDIR / "drop01_tilt_vs_time.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIGURE 3: HEIGHT VS TILT SCATTER
# ============================================================

fig = plt.figure(figsize=(6.6, 5.4))
ax = fig.add_subplot(111)

ax.scatter(
    df[height_col],
    df[tilt_col],
    s=10,
    alpha=0.45,
    label="All sampled frames",
)

ax.scatter(
    post_ads[height_col],
    post_ads[tilt_col],
    s=10,
    alpha=0.55,
    label="Post-onset adsorbed frames",
)

ax.axvline(
    0.45,
    linestyle="--",
    linewidth=1.3,
    label="0.45 nm threshold",
)

ax.axhline(
    30.0,
    linestyle="--",
    linewidth=1.3,
    label="30° threshold",
)

ax.set_title("Height–tilt state space for pyrene adsorption")
ax.set_xlabel("Pyrene height above graphene (nm)")
ax.set_ylabel("Pyrene tilt from graphene plane (deg)")

finalize_axes(ax)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(OUTDIR / "drop01_height_tilt_scatter.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIGURE 4: TRANSITION ZOOM
# ============================================================

# Center the zoom around adsorption onset.
zoom_start_ns = max(0.0, stable_onset_ns - 0.35)
zoom_end_ns = stable_onset_ns + 0.35

zoom = df[
    (df["time_ns"] >= zoom_start_ns)
    &
    (df["time_ns"] <= zoom_end_ns)
].copy()

fig = plt.figure(figsize=(8.6, 5.2))
ax = fig.add_subplot(111)

ax.plot(
    zoom["time_ns"],
    zoom[height_col],
    label="Height (nm)",
)

ax.axhline(
    0.45,
    linestyle="--",
    linewidth=1.4,
    label="0.45 nm threshold",
)

ax.axhline(
    0.34,
    linestyle=":",
    linewidth=1.3,
    label=r"$\pi$-$\pi$ stacking floor",
)

ax.axvline(
    first_crossing_ns,
    linestyle=":",
    linewidth=1.2,
    label=f"First h≤0.45 crossing ({first_crossing_ns:.3f} ns)",
)

ax.axvline(
    contact_onset_ns,
    linestyle="--",
    linewidth=1.2,
    label=f"Sustained-contact onset ({contact_onset_ns:.3f} ns)",
)

ax.axvline(
    stable_onset_ns,
    linestyle="--",
    linewidth=1.2,
    label=f"Stable adsorption onset ({stable_onset_ns:.3f} ns)",
)

ax.set_title("Adsorption transition window: pyrene height")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Pyrene height above graphene (nm)")
ax.set_xlim(zoom_start_ns, zoom_end_ns)

finalize_axes(ax)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(OUTDIR / "drop01_transition_zoom_height.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIGURE 5: POST-ADSORPTION HEIGHT HISTOGRAM
# ============================================================

fig = plt.figure(figsize=(6.6, 4.8))
ax = fig.add_subplot(111)

ax.hist(
    post_ads[height_col],
    bins=40,
    edgecolor="black",
    linewidth=0.7,
    label="Post-onset adsorbed frames",
)

mean_h = post_ads[height_col].mean()
median_h = post_ads[height_col].median()

ax.axvline(
    mean_h,
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {mean_h:.4f} nm",
)

ax.axvline(
    median_h,
    linestyle=":",
    linewidth=1.5,
    label=f"Median = {median_h:.4f} nm",
)

ax.axvline(
    0.34,
    linestyle="-.",
    linewidth=1.3,
    label=r"$\pi$-$\pi$ stacking floor (~0.34 nm)",
)

ax.set_title("Post-adsorption pyrene height distribution")
ax.set_xlabel("Pyrene height above graphene (nm)")
ax.set_ylabel("Count")

finalize_axes(ax)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(OUTDIR / "drop01_postads_height_hist.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# FIGURE 6: POST-ADSORPTION TILT HISTOGRAM
# ============================================================

fig = plt.figure(figsize=(6.6, 4.8))
ax = fig.add_subplot(111)

ax.hist(
    post_ads[tilt_col],
    bins=40,
    edgecolor="black",
    linewidth=0.7,
    label="Post-onset adsorbed frames",
)

mean_t = post_ads[tilt_col].mean()
median_t = post_ads[tilt_col].median()

ax.axvline(
    mean_t,
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {mean_t:.2f}°",
)

ax.axvline(
    median_t,
    linestyle=":",
    linewidth=1.5,
    label=f"Median = {median_t:.2f}°",
)

ax.axvline(
    30.0,
    linestyle="-.",
    linewidth=1.3,
    label="30° criterion",
)

ax.set_title("Post-adsorption pyrene tilt distribution")
ax.set_xlabel("Pyrene tilt from graphene plane (deg)")
ax.set_ylabel("Count")

finalize_axes(ax)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(OUTDIR / "drop01_postads_tilt_hist.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# OPTIONAL: TRANSITION ZOOM FOR TILT
# ============================================================

fig = plt.figure(figsize=(8.6, 5.2))
ax = fig.add_subplot(111)

ax.plot(
    zoom["time_ns"],
    zoom[tilt_col],
    label="Tilt (deg)",
)

ax.axhline(
    30.0,
    linestyle="--",
    linewidth=1.4,
    label="30° criterion",
)

ax.axvline(
    stable_onset_ns,
    linestyle="--",
    linewidth=1.2,
    label=f"Stable adsorption onset ({stable_onset_ns:.3f} ns)",
)

ax.set_title("Adsorption transition window: pyrene tilt")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Pyrene tilt from graphene plane (deg)")
ax.set_xlim(zoom_start_ns, zoom_end_ns)

finalize_axes(ax)
ax.legend(loc="best", frameon=True)

fig.tight_layout()
fig.savefig(OUTDIR / "drop01_transition_zoom_tilt.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 80)
print("DROP-01 PUBLICATION-GRADE FIGURE EXPORT")
print("=" * 80)
print("Input CSV:   ", MERGED_CSV)
print("Input JSON:  ", SUMMARY_JSON)
print("Output dir:  ", OUTDIR)
print()
print("Saved figures:")
for path in sorted(OUTDIR.glob("*.png")):
    print("  ", path)
print()
print(f"Stable adsorption onset:   {stable_onset_ns:.6f} ns")
print(f"Stable adsorption confirm: {stable_confirm_ns:.6f} ns")
print(f"Post-onset height mean:    {mean_h:.6f} nm")
print(f"Post-onset tilt mean:      {mean_t:.6f} deg")
print("=" * 80)
