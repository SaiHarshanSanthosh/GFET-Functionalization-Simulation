# ============================================================
# Render Pyrene-PEG5 / graphene trajectory frames for GIF
#
# Assumes:
# - topology PDB + DCD are loaded into the SAME molecule
# - total frames = 392
# - frame 0 = topology PDB
# - frames 1-391 = 110-500 ps trajectory
#
# This script keeps your CURRENT VMD camera and representations.
# ============================================================

set molid [molinfo top]

set nframes [molinfo $molid get numframes]

puts ""
puts "=============================================="
puts "PYRENE GIF FRAME RENDER"
puts "=============================================="
puts "Molecule: $molid"
puts "Frames:   $nframes"

if {$nframes != 392} {
    error "Expected 392 total frames, found $nframes"
}

set outdir {C:/Users/saisa/GFET Simulation/graphene-functionalization-md/analysis/vmd_gif_frames}

file mkdir $outdir

# Orthographic projection is cleaner for the graphene surface.
display projection Orthographic

# DCD frames begin at VMD frame 1.
set first_frame 1
set last_frame 391

# Render every 4th trajectory frame.
# This gives about 98 images for the GIF.
set stride 4

set image_number 0

for {set frame $first_frame} {$frame <= $last_frame} {incr frame $stride} {

    animate goto $frame

    # Force VMD to refresh the geometry before rendering.
    display update

    set outfile [
        format "%s/frame_%04d.tga" $outdir $image_number
    ]

    puts [
        format "Rendering VMD frame %3d -> image %03d" \
        $frame \
        $image_number
    ]

    render TachyonInternal $outfile

    incr image_number
}

puts ""
puts "=============================================="
puts "VMD FRAME RENDER COMPLETE"
puts "=============================================="
puts "Rendered images: $image_number"
puts "Output directory:"
puts $outdir
puts ""