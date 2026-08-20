# ============================================================
# 5 NS PYRENE LATERAL MOTION ANALYSIS
#
# Production trajectory:
#   500 -> 5500 ps
#
# DCD reporting:
#   every 2 ps
#
# PDB frame:
#   frame 0 = topology coordinate frame (NOT analyzed)
#
# DCD frames:
#   frame 1    = 502 ps
#   ...
#   frame 2500 = 5500 ps
#
# Output distances are converted from VMD Angstrom -> nm.
# ============================================================


# ============================================================
# FILES
# ============================================================

set topology_file {C:/Users/saisa/GFET Simulation/graphene-functionalization-md/structures/pyrene_peg5_graphene_minimized_vmd_whole_with_water_6101.pdb}

set trajectory_file {C:/Users/saisa/GFET Simulation/graphene-functionalization-md/trajectories/pyrene_peg5_graphene_production_300K_500to5500ps_visible.dcd}

set output_csv {C:/Users/saisa/GFET Simulation/graphene-functionalization-md/analysis/pyrene_lateral_5ns.csv}


# ============================================================
# SYSTEM CONSTANTS
# ============================================================

set expected_atoms 19623
set expected_frames 2501

set graphene_first 0
set graphene_last 1249

# Exact visible-DCD indices for the 16 aromatic pyrene atoms.
set pyrene_indices {
1252 1253 1254 1255
1256 1257 1258 1259
1260 1261 1262 1263
1264 1265 1266 1267
}

set first_dcd_frame 1
set last_dcd_frame 2500

set start_time_ps 500.0
set frame_spacing_ps 2.0


# ============================================================
# 2D TRICLINIC MINIMUM IMAGE
#
# Cell:
#
# a = (ax, 0)
# b = (bx, by)
#
# Input/output in Angstrom.
# ============================================================

proc minimage2d {dx dy ax bx by} {

    if {$ax == 0.0 || $by == 0.0} {
        error "Invalid periodic cell dimensions"
    }

    set fy [
        expr {
            $dy / $by
        }
    ]

    set fx [
        expr {
            (
                $dx
                -
                $fy * $bx
            )
            /
            $ax
        }
    ]

    set fx [
        expr {
            $fx - round($fx)
        }
    ]

    set fy [
        expr {
            $fy - round($fy)
        }
    ]

    set dx_mi [
        expr {
            $fx * $ax
            +
            $fy * $bx
        }
    ]

    set dy_mi [
        expr {
            $fy * $by
        }
    ]

    return [
        list \
            $dx_mi \
            $dy_mi
    ]
}


# ============================================================
# PYRENE GEOMETRIC CENTER
#
# Make the 16 aromatic atoms whole in xy first.
#
# z is left unchanged because the molecule is not crossing
# the z periodic boundary.
# ============================================================

proc pyrene_center {
    selection
    ax
    bx
    by
} {

    set coords [
        $selection get {x y z}
    ]

    set first [
        lindex $coords 0
    ]

    set x0 [
        lindex $first 0
    ]

    set y0 [
        lindex $first 1
    ]

    set z0 [
        lindex $first 2
    ]

    set sum_x $x0
    set sum_y $y0
    set sum_z $z0

    set count 1


    foreach xyz [
        lrange $coords 1 end
    ] {

        set x [
            lindex $xyz 0
        ]

        set y [
            lindex $xyz 1
        ]

        set z [
            lindex $xyz 2
        ]

        set dx [
            expr {
                $x - $x0
            }
        ]

        set dy [
            expr {
                $y - $y0
            }
        ]

        lassign [
            minimage2d \
                $dx \
                $dy \
                $ax \
                $bx \
                $by
        ] dx_mi dy_mi

        set xu [
            expr {
                $x0 + $dx_mi
            }
        ]

        set yu [
            expr {
                $y0 + $dy_mi
            }
        ]

        set sum_x [
            expr {
                $sum_x + $xu
            }
        ]

        set sum_y [
            expr {
                $sum_y + $yu
            }
        ]

        set sum_z [
            expr {
                $sum_z + $z
            }
        ]

        incr count
    }


    return [
        list \
            [expr {$sum_x / double($count)}] \
            [expr {$sum_y / double($count)}] \
            [expr {$sum_z / double($count)}]
    ]
}


# ============================================================
# LOAD FRESH MOLECULE
# ============================================================

puts ""
puts "============================================================"
puts "5 NS PYRENE LATERAL MOTION ANALYSIS"
puts "============================================================"
puts ""

puts "Loading topology..."

set molid [
    mol new \
        $topology_file \
        type pdb \
        waitfor all
]

puts "Loading production DCD..."

mol addfile \
    $trajectory_file \
    type dcd \
    waitfor all \
    molid $molid


set natoms [
    molinfo $molid get numatoms
]

set nframes [
    molinfo $molid get numframes
]


puts ""
puts "Molecule: $molid"
puts "Atoms:    $natoms"
puts "Frames:   $nframes"


if {$natoms != $expected_atoms} {

    error \
        "Expected $expected_atoms atoms, found $natoms"
}


if {$nframes != $expected_frames} {

    error \
        "Expected $expected_frames frames, found $nframes"
}


# ============================================================
# SELECTIONS
# ============================================================

set graphene [
    atomselect \
        $molid \
        "index $graphene_first to $graphene_last"
]

set pyrene_selection_text \
    "index [join $pyrene_indices { }]"

set pyrene [
    atomselect \
        $molid \
        $pyrene_selection_text
]


if {[$graphene num] != 1250} {

    error \
        "Expected 1250 graphene atoms, found [$graphene num]"
}


if {[$pyrene num] != 16} {

    error \
        "Expected 16 pyrene aromatic atoms, found [$pyrene num]"
}


puts ""
puts "Graphene carbons:      [$graphene num]"
puts "Pyrene aromatic atoms: [$pyrene num]"


# ============================================================
# INITIAL FRAME = 502 ps
# ============================================================

set initial_frame $first_dcd_frame

animate goto $initial_frame

$graphene frame $initial_frame
$pyrene frame $initial_frame


# ============================================================
# PERIODIC CELL
# ============================================================

lassign [
    molinfo $molid get {a b gamma}
] a b gamma_deg


set gamma_rad [
    expr {
        $gamma_deg
        *
        acos(-1.0)
        /
        180.0
    }
]


set ax $a

set bx [
    expr {
        $b * cos($gamma_rad)
    }
]

set by [
    expr {
        $b * sin($gamma_rad)
    }
]


puts ""
puts "Periodic cell:"
puts "  ax = $ax A"
puts "  bx = $bx A"
puts "  by = $by A"


# ============================================================
# INITIAL PYRENE CENTER
# ============================================================

set initial_center [
    pyrene_center \
        $pyrene \
        $ax \
        $bx \
        $by
]

set px0 [
    lindex $initial_center 0
]

set py0 [
    lindex $initial_center 1
]


# ============================================================
# CHOOSE NEAREST GRAPHENE REFERENCE CARBON
#
# Using a graphene atom removes overall xy translation of
# the sheet from the pyrene sliding coordinate.
# ============================================================

set graphene_xy [
    $graphene get {x y}
]

set best_distance2 1.0e30

set reference_index -1

set local_index 0


foreach xy $graphene_xy {

    set gx [
        lindex $xy 0
    ]

    set gy [
        lindex $xy 1
    ]

    set dx [
        expr {
            $px0 - $gx
        }
    ]

    set dy [
        expr {
            $py0 - $gy
        }
    ]


    lassign [
        minimage2d \
            $dx \
            $dy \
            $ax \
            $bx \
            $by
    ] dx_mi dy_mi


    set d2 [
        expr {
            $dx_mi*$dx_mi
            +
            $dy_mi*$dy_mi
        }
    ]


    if {$d2 < $best_distance2} {

        set best_distance2 $d2

        set reference_index [
            expr {
                $graphene_first
                +
                $local_index
            }
        ]
    }


    incr local_index
}


if {$reference_index < 0} {

    error "Could not identify graphene reference atom"
}


set reference [
    atomselect \
        $molid \
        "index $reference_index"
]


puts ""
puts "Reference graphene carbon:"
puts "  index = $reference_index"

puts [
    format \
        "  initial xy distance = %.4f A" \
        [expr {sqrt($best_distance2)}]
]


# ============================================================
# OUTPUT
# ============================================================

set outfile [
    open \
        $output_csv \
        w
]


puts $outfile \
"time_ps,lateral_displacement_nm,lateral_x_nm,lateral_y_nm,path_length_nm,step_displacement_nm"


# ============================================================
# TRACK RELATIVE MOTION
# ============================================================

set accumulated_x_A 0.0
set accumulated_y_A 0.0

set cumulative_path_A 0.0

set previous_rel_x_A 0.0
set previous_rel_y_A 0.0

set first_sample 1

set maximum_displacement_A 0.0
set maximum_step_A 0.0

set sum_displacement2_A2 0.0
set sample_count 0


puts ""
puts "Analyzing frames..."


for {
    set frame $first_dcd_frame
} {
    $frame <= $last_dcd_frame
} {
    incr frame
} {

    animate goto $frame

    $pyrene frame $frame
    $reference frame $frame


    # --------------------------------------------
    # Cell should remain fixed in NVT, but read it
    # every frame for safety.
    # --------------------------------------------

    lassign [
        molinfo $molid get {a b gamma}
    ] a b gamma_deg


    set gamma_rad [
        expr {
            $gamma_deg
            *
            acos(-1.0)
            /
            180.0
        }
    ]


    set ax $a

    set bx [
        expr {
            $b * cos($gamma_rad)
        }
    ]

    set by [
        expr {
            $b * sin($gamma_rad)
        }
    ]


    # --------------------------------------------
    # Pyrene center
    # --------------------------------------------

    set center [
        pyrene_center \
            $pyrene \
            $ax \
            $bx \
            $by
    ]


    set px [
        lindex $center 0
    ]

    set py [
        lindex $center 1
    ]


    # --------------------------------------------
    # Graphene reference position
    # --------------------------------------------

    set ref_xyz [
        lindex \
            [$reference get {x y z}] \
            0
    ]


    set gx [
        lindex $ref_xyz 0
    ]

    set gy [
        lindex $ref_xyz 1
    ]


    # --------------------------------------------
    # Instantaneous pyrene-relative-to-graphene
    # vector, minimum imaged
    # --------------------------------------------

    set rel_dx [
        expr {
            $px - $gx
        }
    ]

    set rel_dy [
        expr {
            $py - $gy
        }
    ]


    lassign [
        minimage2d \
            $rel_dx \
            $rel_dy \
            $ax \
            $bx \
            $by
    ] rel_x_A rel_y_A


    # --------------------------------------------
    # First trajectory frame defines zero
    # --------------------------------------------

    if {$first_sample} {

        set previous_rel_x_A $rel_x_A
        set previous_rel_y_A $rel_y_A

        set step_A 0.0

        set first_sample 0

    } else {

        set delta_x [
            expr {
                $rel_x_A
                -
                $previous_rel_x_A
            }
        ]

        set delta_y [
            expr {
                $rel_y_A
                -
                $previous_rel_y_A
            }
        ]


        # Minimum-image the frame-to-frame displacement.
        lassign [
            minimage2d \
                $delta_x \
                $delta_y \
                $ax \
                $bx \
                $by
        ] delta_x_mi delta_y_mi


        set accumulated_x_A [
            expr {
                $accumulated_x_A
                +
                $delta_x_mi
            }
        ]

        set accumulated_y_A [
            expr {
                $accumulated_y_A
                +
                $delta_y_mi
            }
        ]


        set step_A [
            expr {
                sqrt(
                    $delta_x_mi*$delta_x_mi
                    +
                    $delta_y_mi*$delta_y_mi
                )
            }
        ]


        set cumulative_path_A [
            expr {
                $cumulative_path_A
                +
                $step_A
            }
        ]


        if {$step_A > $maximum_step_A} {

            set maximum_step_A $step_A
        }


        set previous_rel_x_A $rel_x_A
        set previous_rel_y_A $rel_y_A
    }


    # --------------------------------------------
    # Net displacement from first analyzed frame
    # --------------------------------------------

    set displacement_A [
        expr {
            sqrt(
                $accumulated_x_A*$accumulated_x_A
                +
                $accumulated_y_A*$accumulated_y_A
            )
        }
    ]


    if {$displacement_A > $maximum_displacement_A} {

        set maximum_displacement_A $displacement_A
    }


    set sum_displacement2_A2 [
        expr {
            $sum_displacement2_A2
            +
            $displacement_A*$displacement_A
        }
    ]


    incr sample_count


    # --------------------------------------------
    # Time mapping
    #
    # frame 1 = 502 ps
    # --------------------------------------------

    set time_ps [
        expr {
            $start_time_ps
            +
            $frame
            *
            $frame_spacing_ps
        }
    ]


    # Angstrom -> nm
    set displacement_nm [
        expr {
            0.1 * $displacement_A
        }
    ]

    set accumulated_x_nm [
        expr {
            0.1 * $accumulated_x_A
        }
    ]

    set accumulated_y_nm [
        expr {
            0.1 * $accumulated_y_A
        }
    ]

    set path_nm [
        expr {
            0.1 * $cumulative_path_A
        }
    ]

    set step_nm [
        expr {
            0.1 * $step_A
        }
    ]


    puts $outfile [
        format \
            "%.3f,%.8f,%.8f,%.8f,%.8f,%.8f" \
            $time_ps \
            $displacement_nm \
            $accumulated_x_nm \
            $accumulated_y_nm \
            $path_nm \
            $step_nm
    ]


    if {
        $frame == 1
        ||
        $frame % 250 == 0
    } {

        puts [
            format \
                "frame %4d / %4d   time=%7.1f ps   displacement=%7.4f nm   path=%8.3f nm" \
                $frame \
                $last_dcd_frame \
                $time_ps \
                $displacement_nm \
                $path_nm
        ]
    }
}


close $outfile


# ============================================================
# SUMMARY
# ============================================================

set final_displacement_A [
    expr {
        sqrt(
            $accumulated_x_A*$accumulated_x_A
            +
            $accumulated_y_A*$accumulated_y_A
        )
    }
]


set rms_displacement_A [
    expr {
        sqrt(
            $sum_displacement2_A2
            /
            double($sample_count)
        )
    }
]


puts ""
puts "============================================================"
puts "5 NS LATERAL MOTION RESULTS"
puts "============================================================"

puts ""
puts "Analyzed trajectory:"
puts "  502 -> 5500 ps"
puts "  2 ps sampling"
puts "  samples: $sample_count"

puts ""
puts "LATERAL MOTION"

puts [
    format \
        "  final net displacement: %.5f nm" \
        [expr {0.1*$final_displacement_A}]
]

puts [
    format \
        "  maximum displacement:   %.5f nm" \
        [expr {0.1*$maximum_displacement_A}]
]

puts [
    format \
        "  RMS displacement:       %.5f nm" \
        [expr {0.1*$rms_displacement_A}]
]

puts [
    format \
        "  final dx:               %.5f nm" \
        [expr {0.1*$accumulated_x_A}]
]

puts [
    format \
        "  final dy:               %.5f nm" \
        [expr {0.1*$accumulated_y_A}]
]

puts [
    format \
        "  cumulative path length: %.5f nm" \
        [expr {0.1*$cumulative_path_A}]
]

puts [
    format \
        "  largest 2 ps step:      %.5f nm" \
        [expr {0.1*$maximum_step_A}]
]

puts ""
puts "IMPORTANT:"
puts "  Cumulative path length is calculated from coordinates"
puts "  saved every 2 ps, so it is a lower bound on the true"
puts "  microscopic distance traveled."

puts ""
puts "CSV:"
puts "  $output_csv"

puts ""
puts "============================================================"
puts "PYRENE_5NS_LATERAL_ANALYSIS_PASS"
puts "============================================================"
puts ""