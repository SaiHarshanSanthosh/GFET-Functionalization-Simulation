
# ============================================================
# Pyrene motion analysis: 110 -> 500 ps
#
# Metrics:
#   1. Pyrene height above graphene
#   2. PBC-safe lateral displacement relative to graphene
#   3. Pyrene tilt relative to graphene plane
#
# VMD coordinates are in Angstrom.
# CSV distances are converted to nm.
# ============================================================

cd {C:/Users/saisa/GFET Simulation/graphene-functionalization-md}

set EXPECTED_ATOMS 19623
set EXPECTED_FRAMES 392

set START_PS 110.0
set DT_PS 1.0

set PYRENE_INDICES {1252 1253 1254 1255 1256 1257 1258 1259 1260 1261 1262 1263 1264 1265 1266 1267}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

proc mean_value {vals} {

    set n [llength $vals]

    if {$n == 0} {
        return 0.0
    }

    set s 0.0

    foreach x $vals {
        set s [expr {$s + $x}]
    }

    return [expr {$s / double($n)}]
}


proc std_value {vals} {

    set n [llength $vals]

    if {$n == 0} {
        return 0.0
    }

    set m [mean_value $vals]

    set ss 0.0

    foreach x $vals {

        set d [expr {$x - $m}]

        set ss [expr {
            $ss + $d*$d
        }]
    }

    return [expr {
        sqrt(
            $ss / double($n)
        )
    }]
}


proc min_image_xy {
    dx dy ax bx by
} {

    set fb [expr {
        $dy / $by
    }]

    set fa [expr {
        ($dx - $fb*$bx)
        /
        $ax
    }]

    set fa [expr {
        $fa - floor($fa + 0.5)
    }]

    set fb [expr {
        $fb - floor($fb + 0.5)
    }]

    set x [expr {
        $fa*$ax + $fb*$bx
    }]

    set y [expr {
        $fb*$by
    }]

    return [list $x $y]
}


proc get_cell_xy {molid} {

    set vals [
        molinfo $molid get {
            a b gamma
        }
    ]

    set a [
        lindex $vals 0
    ]

    set b [
        lindex $vals 1
    ]

    set gamma [
        lindex $vals 2
    ]

    set pi [
        expr {acos(-1.0)}
    ]

    set gr [
        expr {
            $gamma * $pi / 180.0
        }
    ]

    set ax $a

    set bx [
        expr {
            $b * cos($gr)
        }
    ]

    set by [
        expr {
            $b * sin($gr)
        }
    ]

    return [
        list $ax $bx $by
    ]
}


proc unwrap_cluster {
    coords ax bx by
} {

    set ref [
        lindex $coords 0
    ]

    set rx [
        lindex $ref 0
    ]

    set ry [
        lindex $ref 1
    ]

    set rz [
        lindex $ref 2
    ]

    set unwrapped {}

    set sx 0.0
    set sy 0.0
    set sz 0.0

    foreach p $coords {

        set dx [
            expr {
                [lindex $p 0] - $rx
            }
        ]

        set dy [
            expr {
                [lindex $p 1] - $ry
            }
        ]

        lassign [
            min_image_xy \
            $dx $dy \
            $ax $bx $by
        ] dxw dyw

        set x [
            expr {
                $rx + $dxw
            }
        ]

        set y [
            expr {
                $ry + $dyw
            }
        ]

        set z [
            lindex $p 2
        ]

        lappend unwrapped \
            [list $x $y $z]

        set sx [
            expr {$sx + $x}
        ]

        set sy [
            expr {$sy + $y}
        ]

        set sz [
            expr {$sz + $z}
        ]
    }

    set n [
        llength $coords
    ]

    set center [
        list \
        [expr {$sx/double($n)}] \
        [expr {$sy/double($n)}] \
        [expr {$sz/double($n)}]
    ]

    return [
        list $center $unwrapped
    ]
}


# ------------------------------------------------------------
# Find the correctly combined PDB + DCD molecule
# ------------------------------------------------------------

set candidates {}

foreach m [molinfo list] {

    set natoms [
        molinfo $m get numatoms
    ]

    set nframes [
        molinfo $m get numframes
    ]

    if {
        $natoms == $EXPECTED_ATOMS
        &&
        $nframes == $EXPECTED_FRAMES
    } {

        lappend candidates $m
    }
}


if {
    [llength $candidates] == 0
} {

    puts ""
    puts "ERROR:"
    puts "Could not find a molecule with:"
    puts "  atoms  = 19623"
    puts "  frames = 392"
    puts ""
    puts "Load the PDB first, then load the"
    puts "110to500ps DCD INTO that molecule."
    puts ""

    error "Correct trajectory molecule not found."
}


if {
    [llength $candidates] > 1
} {

    puts ""
    puts "ERROR: multiple matching molecules:"
    puts $candidates
    puts ""

    error "More than one 392-frame molecule found."
}


set molid [
    lindex $candidates 0
]

set original_frame [
    molinfo $molid get frame
]

puts ""
puts "============================================================"
puts "PYRENE MOTION ANALYSIS"
puts "============================================================"
puts ""
puts "Using molecule: $molid"
puts "Atoms:  [molinfo $molid get numatoms]"
puts "Frames: [molinfo $molid get numframes]"


# ------------------------------------------------------------
# Selections
# ------------------------------------------------------------

set pyrene_text \
    "index [join $PYRENE_INDICES " "]"

set pyrene [
    atomselect $molid $pyrene_text
]

set graphene [
    atomselect $molid "index 0 to 1249"
]


if {
    [$pyrene num] != 16
} {

    error "Pyrene selection is not 16 atoms."
}


if {
    [$graphene num] != 1250
} {

    error "Graphene selection is not 1250 atoms."
}


puts "Pyrene aromatic atoms: [$pyrene num]"
puts "Graphene carbons:      [$graphene num]"


# ------------------------------------------------------------
# Frame 1 = DCD time 110 ps
# Frame 0 is the topology PDB coordinate frame
# ------------------------------------------------------------

set first_frame 1
set last_frame 391

molinfo $molid set frame $first_frame

$pyrene frame $first_frame
$graphene frame $first_frame

$pyrene update
$graphene update


lassign [
    get_cell_xy $molid
] ax bx by


puts ""
puts "Periodic cell from trajectory:"
puts "  ax = $ax A"
puts "  bx = $bx A"
puts "  by = $by A"


# ------------------------------------------------------------
# Find nearest graphene carbon to initial pyrene center.
# This carbon acts as a moving lattice reference so whole-sheet
# translation does not masquerade as pyrene sliding.
# ------------------------------------------------------------

set py_orig [
    $pyrene get {x y z}
]

lassign [
    unwrap_cluster \
    $py_orig \
    $ax $bx $by
] py_center py_unwrapped


set gcoords [
    $graphene get {x y z}
]

set gindices [
    $graphene get index
]

set best_r2 1.0e99
set ref_index -1

for {
    set i 0
} {
    $i < [llength $gcoords]
} {
    incr i
} {

    set p [
        lindex $gcoords $i
    ]

    set dx [
        expr {
            [lindex $py_center 0]
            -
            [lindex $p 0]
        }
    ]

    set dy [
        expr {
            [lindex $py_center 1]
            -
            [lindex $p 1]
        }
    ]

    lassign [
        min_image_xy \
        $dx $dy \
        $ax $bx $by
    ] dxw dyw

    set r2 [
        expr {
            $dxw*$dxw
            +
            $dyw*$dyw
        }
    ]

    if {
        $r2 < $best_r2
    } {

        set best_r2 $r2

        set ref_index [
            lindex $gindices $i
        ]
    }
}


if {
    $ref_index < 0
} {

    error "Could not identify graphene reference atom."
}


set refsel [
    atomselect $molid \
    "index $ref_index"
]


puts ""
puts "Reference graphene carbon index: $ref_index"
puts "Initial lateral separation from reference:"
puts [format "  %.4f A" [expr {sqrt($best_r2)}]]


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

file mkdir analysis

set outfile \
    "analysis/pyrene_motion_110to500ps.csv"

set fh [
    open $outfile w
]

puts $fh \
"time_ps,pyrene_height_nm,lateral_displacement_nm,lateral_x_nm,lateral_y_nm,path_length_nm,tilt_deg"


set heights {}
set displacements {}
set tilts {}

set cumulative_x 0.0
set cumulative_y 0.0
set path_length 0.0

set previous_rel {}
set max_displacement 0.0


# ------------------------------------------------------------
# Analyze DCD frames 110 -> 500 ps
# ------------------------------------------------------------

for {
    set f $first_frame
} {
    $f <= $last_frame
} {
    incr f
} {

    molinfo $molid set frame $f

    $pyrene frame $f
    $graphene frame $f
    $refsel frame $f

    $pyrene update
    $graphene update
    $refsel update


    lassign [
        get_cell_xy $molid
    ] ax bx by


    # --------------------------------------------------------
    # PBC-safe pyrene center
    # --------------------------------------------------------

    set py_orig [
        $pyrene get {x y z}
    ]

    lassign [
        unwrap_cluster \
        $py_orig \
        $ax $bx $by
    ] py_center py_unwrapped


    # --------------------------------------------------------
    # Height: EXACT SAME concept as equilibration script.
    # mean pyrene aromatic z - mean graphene carbon z
    # --------------------------------------------------------

    set graph_center [
        measure center $graphene
    ]

    set height_A [
        expr {
            [lindex $py_center 2]
            -
            [lindex $graph_center 2]
        }
    ]

    set height_nm [
        expr {
            0.1 * $height_A
        }
    ]


    # --------------------------------------------------------
    # Lateral position relative to reference graphene carbon
    # --------------------------------------------------------

    set refcoord [
        lindex [
            $refsel get {x y z}
        ] 0
    ]

    set dx [
        expr {
            [lindex $py_center 0]
            -
            [lindex $refcoord 0]
        }
    ]

    set dy [
        expr {
            [lindex $py_center 1]
            -
            [lindex $refcoord 1]
        }
    ]

    lassign [
        min_image_xy \
        $dx $dy \
        $ax $bx $by
    ] relx rely


    if {
        $f == $first_frame
    } {

        set previous_rel \
            [list $relx $rely]

    } else {

        set ddx [
            expr {
                $relx
                -
                [lindex $previous_rel 0]
            }
        ]

        set ddy [
            expr {
                $rely
                -
                [lindex $previous_rel 1]
            }
        ]

        lassign [
            min_image_xy \
            $ddx $ddy \
            $ax $bx $by
        ] ddxw ddyw

        set cumulative_x [
            expr {
                $cumulative_x + $ddxw
            }
        ]

        set cumulative_y [
            expr {
                $cumulative_y + $ddyw
            }
        ]

        set step_distance [
            expr {
                sqrt(
                    $ddxw*$ddxw
                    +
                    $ddyw*$ddyw
                )
            }
        ]

        set path_length [
            expr {
                $path_length
                +
                $step_distance
            }
        ]

        set previous_rel \
            [list $relx $rely]
    }


    set displacement_A [
        expr {
            sqrt(
                $cumulative_x*$cumulative_x
                +
                $cumulative_y*$cumulative_y
            )
        }
    ]

    set displacement_nm [
        expr {
            0.1 * $displacement_A
        }
    ]

    set lateral_x_nm [
        expr {
            0.1 * $cumulative_x
        }
    ]

    set lateral_y_nm [
        expr {
            0.1 * $cumulative_y
        }
    ]

    set path_nm [
        expr {
            0.1 * $path_length
        }
    ]


    if {
        $displacement_nm > $max_displacement
    } {

        set max_displacement \
            $displacement_nm
    }


    # --------------------------------------------------------
    # Pyrene tilt
    #
    # Temporarily make pyrene whole across PBC.
    # Principal axis with largest moment is normal to a
    # planar aromatic group.
    #
    # tilt = 0 degrees => pyrene plane parallel to graphene.
    # --------------------------------------------------------

    $pyrene set {x y z} \
        $py_unwrapped

    set inertia [
        measure inertia \
        $pyrene \
        eigenvals
    ]

    set axes [
        lindex $inertia 1
    ]

    set evals [
        lindex $inertia 2
    ]

    set imax 0

    for {
        set j 1
    } {
        $j < 3
    } {
        incr j
    } {

        if {
            [lindex $evals $j]
            >
            [lindex $evals $imax]
        } {

            set imax $j
        }
    }

    set normal [
        lindex $axes $imax
    ]

    set nz [
        expr {
            abs(
                double(
                    [lindex $normal 2]
                )
            )
        }
    ]

    if {
        $nz > 1.0
    } {
        set nz 1.0
    }

    set tilt_deg [
        expr {
            acos($nz)
            *
            180.0
            /
            acos(-1.0)
        }
    ]


    # Restore original wrapped coordinates immediately.
    $pyrene set {x y z} \
        $py_orig


    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    set time_ps [
        expr {
            $START_PS
            +
            ($f - $first_frame)
            *
            $DT_PS
        }
    ]


    puts $fh [
        format \
        "%.3f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f" \
        $time_ps \
        $height_nm \
        $displacement_nm \
        $lateral_x_nm \
        $lateral_y_nm \
        $path_nm \
        $tilt_deg
    ]


    lappend heights \
        $height_nm

    lappend displacements \
        $displacement_nm

    lappend tilts \
        $tilt_deg
}


close $fh


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

set hmean [
    mean_value $heights
]

set hstd [
    std_value $heights
]

set hmin [
    tcl::mathfunc::min {*}$heights
]

set hmax [
    tcl::mathfunc::max {*}$heights
]

set tmean [
    mean_value $tilts
]

set tstd [
    std_value $tilts
]

set tmin [
    tcl::mathfunc::min {*}$tilts
]

set tmax [
    tcl::mathfunc::max {*}$tilts
]

set final_disp [
    lindex $displacements end
]


puts ""
puts "============================================================"
puts "RESULTS"
puts "============================================================"
puts ""
puts "Frames analyzed: 391"
puts "Time range:      110 -> 500 ps"
puts ""

puts "PYRENE HEIGHT"
puts [format \
    "  mean: %.5f nm  (%.3f A)" \
    $hmean \
    [expr {10.0*$hmean}]
]

puts [format \
    "  std:  %.5f nm" \
    $hstd
]

puts [format \
    "  range %.5f -> %.5f nm" \
    $hmin $hmax
]

puts ""

puts "LATERAL MOTION"
puts [format \
    "  final net displacement: %.5f nm" \
    $final_disp
]

puts [format \
    "  maximum displacement:   %.5f nm" \
    $max_displacement
]

puts [format \
    "  cumulative path length: %.5f nm" \
    [expr {0.1*$path_length}]
]

puts ""

puts "PYRENE TILT"
puts [format \
    "  mean: %.3f deg" \
    $tmean
]

puts [format \
    "  std:  %.3f deg" \
    $tstd
]

puts [format \
    "  range %.3f -> %.3f deg" \
    $tmin $tmax
]

puts ""
puts "CSV:"
puts "  $outfile"
puts ""
puts "============================================================"
puts "ANALYSIS COMPLETE"
puts "============================================================"


# Restore the frame VMD was displaying before analysis.
molinfo $molid set frame $original_frame

$pyrene delete
$graphene delete
$refsel delete
