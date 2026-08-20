from pathlib import Path

from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

FRAME_DIR = Path(
    "analysis/vmd_gif_frames"
)

OUTPUT_GIF = Path(
    "analysis/pyrene_peg5_graphene_110to500ps.gif"
)


# ============================================================
# FIND VMD RENDERS
# ============================================================

frames = sorted(
    FRAME_DIR.glob("frame_*.tga")
)

if len(frames) == 0:
    raise RuntimeError(
        f"No TGA frames found in {FRAME_DIR}"
    )


print()
print("=" * 60)
print("BUILDING PYRENE-PEG5 / GRAPHENE GIF")
print("=" * 60)

print(
    f"Frames found: {len(frames)}"
)

print(
    f"First: {frames[0]}"
)

print(
    f"Last:  {frames[-1]}"
)


# ============================================================
# TRAJECTORY TIMING
#
# VMD:
# frame 1 = 110 ps
# rendered stride = 4 trajectory frames
#
# Therefore:
# 110, 114, 118, ...
# ============================================================

START_TIME_PS = 110
TIME_STEP_PS = 4


# ============================================================
# BUILD GIF FRAMES
# ============================================================

images = []


for i, frame_file in enumerate(frames):

    time_ps = (
        START_TIME_PS
        +
        i * TIME_STEP_PS
    )

    image = (
        Image.open(frame_file)
        .convert("RGB")
    )

    draw = ImageDraw.Draw(
        image
    )


    # --------------------------------------------------------
    # TIME LABEL
    # --------------------------------------------------------

    label = (
        f"300 K MD | {time_ps} ps"
    )

    # Text box
    draw.rectangle(
        (
            20,
            18,
            210,
            48,
        ),
        fill="black",
    )

    draw.text(
        (
            30,
            27,
        ),
        label,
        fill="white",
    )


    # Convert to palette mode for GIF.
    gif_frame = image.convert(
        "P",
        palette=Image.ADAPTIVE,
        colors=256,
    )

    images.append(
        gif_frame
    )


# ============================================================
# SAVE
# ============================================================

# 60 ms/frame ~= 16.7 rendered frames/sec.
# Since each rendered frame represents 4 ps,
# the full 390 ps trajectory plays in ~5.9 seconds.

FRAME_DURATION_MS = 60


images[0].save(
    OUTPUT_GIF,
    save_all=True,
    append_images=images[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
    optimize=False,
    disposal=2,
)


print()
print("=" * 60)
print("GIF COMPLETE")
print("=" * 60)

print(
    f"Output: {OUTPUT_GIF}"
)

print(
    f"GIF frames: {len(images)}"
)

print(
    f"Playback duration: "
    f"{len(images) * FRAME_DURATION_MS / 1000:.2f} s"
)

print(
    f"Trajectory represented: "
    f"{START_TIME_PS} - "
    f"{START_TIME_PS + (len(images)-1)*TIME_STEP_PS} ps"
)

print()
print("PYRENE_GIF_PASS")