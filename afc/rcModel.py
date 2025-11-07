# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Reduced-order module (auto-generated).
"""

# pylint: disable=invalid-name, too-many-arguments, too-many-locals, too-many-statements
# pylint: disable=too-many-positional-arguments

RCTYPES = ["R1C1", "R2C2", "R4C2", "R5C3", "R6C3", "R7C4"]

def R1C1(i, Ti_p, To, Qi_ext, param):
    '''Auto-generated reduced-order R1C1 model.'''

    # parsing parameter
    timestep = param["timestep"]
    Rw2i = param["Rw2i"]
    Ci = param["Ci"] / timestep

    # initialization
    if i == 0:
        return [Ti_p]

    # model equations
    Ti = ((Ci*Rw2i*Ti_p
        + Qi_ext*Rw2i
        + To)/(Ci*Rw2i
        + 1))

    return [Ti]

def R2C2(i, Ti_p, Tw_p, To, Qi_ext, Qw_ext, param):
    '''Auto-generated reduced-order R2C2 model.'''

    # parsing parameter
    timestep = param["timestep"]
    Rw2i = param["Rw2i"]
    Riw = param["Riw"]
    Ci = param["Ci"] / timestep
    Cw = param["Cw"] / timestep

    # initialization
    if i == 0:
        return [Ti_p, Tw_p]

    # model equations
    Ti = ((Ci*Cw*Riw*Rw2i*Ti_p
        + Ci*Rw2i*Ti_p
        + Cw*Qi_ext*Riw*Rw2i
        + Cw*Riw*To
        + Cw*Rw2i*Tw_p
        + Qi_ext*Rw2i
        + Qw_ext*Rw2i
        + To)/(Ci*Cw*Riw*Rw2i
        + Ci*Rw2i
        + Cw*Riw
        + Cw*Rw2i
        + 1))
    Tw = ((Ci*Cw*Riw*Rw2i*Tw_p
        + Ci*Qw_ext*Riw*Rw2i
        + Ci*Rw2i*Ti_p
        + Cw*Riw*Tw_p
        + Cw*Rw2i*Tw_p
        + Qi_ext*Rw2i
        + Qw_ext*Riw
        + Qw_ext*Rw2i
        + To)/(Ci*Cw*Riw*Rw2i
        + Ci*Rw2i
        + Cw*Riw
        + Cw*Rw2i
        + 1))

    return [Ti, Tw]

def R4C2(i, Ti_p, Tw_p, To, Qw1_ext, Qw2_ext, Qi_ext, Qw_ext, param):
    '''Auto-generated reduced-order R4C2 model.'''

    # parsing parameter
    timestep = param["timestep"]
    Rw2i = param["Rw2i"]
    Row1 = param["Row1"]
    Riw = param["Riw"]
    Rw1w2 = param["Rw1w2"]
    Ci = param["Ci"] / timestep
    Cw = param["Cw"] / timestep

    # initialization
    if i == 0:
        return [Ti_p, Tw_p]

    # model equations
    Ti = ((Ci*Cw*Riw*Row1*Ti_p
        + Ci*Cw*Riw*Rw1w2*Ti_p
        + Ci*Cw*Riw*Rw2i*Ti_p
        + Ci*Row1*Ti_p
        + Ci*Rw1w2*Ti_p
        + Ci*Rw2i*Ti_p
        + Cw*Qi_ext*Riw*Row1
        + Cw*Qi_ext*Riw*Rw1w2
        + Cw*Qi_ext*Riw*Rw2i
        + Cw*Qw1_ext*Riw*Row1
        + Cw*Qw2_ext*Riw*Row1
        + Cw*Qw2_ext*Riw*Rw1w2
        + Cw*Riw*To
        + Cw*Row1*Tw_p
        + Cw*Rw1w2*Tw_p
        + Cw*Rw2i*Tw_p
        + Qi_ext*Row1
        + Qi_ext*Rw1w2
        + Qi_ext*Rw2i
        + Qw1_ext*Row1
        + Qw2_ext*Row1
        + Qw2_ext*Rw1w2
        + Qw_ext*Row1
        + Qw_ext*Rw1w2
        + Qw_ext*Rw2i
        + To)/(Ci*Cw*Riw*Row1
        + Ci*Cw*Riw*Rw1w2
        + Ci*Cw*Riw*Rw2i
        + Ci*Row1
        + Ci*Rw1w2
        + Ci*Rw2i
        + Cw*Riw
        + Cw*Row1
        + Cw*Rw1w2
        + Cw*Rw2i
        + 1))
    Tw = ((Ci*Cw*Riw*Row1*Tw_p
        + Ci*Cw*Riw*Rw1w2*Tw_p
        + Ci*Cw*Riw*Rw2i*Tw_p
        + Ci*Qw_ext*Riw*Row1
        + Ci*Qw_ext*Riw*Rw1w2
        + Ci*Qw_ext*Riw*Rw2i
        + Ci*Row1*Ti_p
        + Ci*Rw1w2*Ti_p
        + Ci*Rw2i*Ti_p
        + Cw*Riw*Tw_p
        + Cw*Row1*Tw_p
        + Cw*Rw1w2*Tw_p
        + Cw*Rw2i*Tw_p
        + Qi_ext*Row1
        + Qi_ext*Rw1w2
        + Qi_ext*Rw2i
        + Qw1_ext*Row1
        + Qw2_ext*Row1
        + Qw2_ext*Rw1w2
        + Qw_ext*Riw
        + Qw_ext*Row1
        + Qw_ext*Rw1w2
        + Qw_ext*Rw2i
        + To)/(Ci*Cw*Riw*Row1
        + Ci*Cw*Riw*Rw1w2
        + Ci*Cw*Riw*Rw2i
        + Ci*Row1
        + Ci*Rw1w2
        + Ci*Rw2i
        + Cw*Riw
        + Cw*Row1
        + Cw*Rw1w2
        + Cw*Rw2i
        + 1))

    return [Ti, Tw]

def R5C3(i, Ti_p, Tw_p, Ts_p, To, Qw1_ext, Qw2_ext, Qi_ext, Qw_ext, Qs_ext, param):
    '''Auto-generated reduced-order R5C3 model.'''

    # parsing parameter
    timestep = param["timestep"]
    Rw2i = param["Rw2i"]
    Row1 = param["Row1"]
    Ris = param["Ris"]
    Riw = param["Riw"]
    Rw1w2 = param["Rw1w2"]
    Cs = param["Cs"] / timestep
    Ci = param["Ci"] / timestep
    Cw = param["Cw"] / timestep

    # initialization
    if i == 0:
        return [Ti_p, Tw_p, Ts_p]

    # model equations
    Ti = ((Ci*Cs*Cw*Ris*Riw*Row1*Ti_p
        + Ci*Cs*Cw*Ris*Riw*Rw1w2*Ti_p
        + Ci*Cs*Cw*Ris*Riw*Rw2i*Ti_p
        + Ci*Cs*Ris*Row1*Ti_p
        + Ci*Cs*Ris*Rw1w2*Ti_p
        + Ci*Cs*Ris*Rw2i*Ti_p
        + Ci*Cw*Riw*Row1*Ti_p
        + Ci*Cw*Riw*Rw1w2*Ti_p
        + Ci*Cw*Riw*Rw2i*Ti_p
        + Ci*Row1*Ti_p
        + Ci*Rw1w2*Ti_p
        + Ci*Rw2i*Ti_p
        + Cs*Cw*Qi_ext*Ris*Riw*Row1
        + Cs*Cw*Qi_ext*Ris*Riw*Rw1w2
        + Cs*Cw*Qi_ext*Ris*Riw*Rw2i
        + Cs*Cw*Qw1_ext*Ris*Riw*Row1
        + Cs*Cw*Qw2_ext*Ris*Riw*Row1
        + Cs*Cw*Qw2_ext*Ris*Riw*Rw1w2
        + Cs*Cw*Ris*Riw*To
        + Cs*Cw*Ris*Row1*Tw_p
        + Cs*Cw*Ris*Rw1w2*Tw_p
        + Cs*Cw*Ris*Rw2i*Tw_p
        + Cs*Cw*Riw*Row1*Ts_p
        + Cs*Cw*Riw*Rw1w2*Ts_p
        + Cs*Cw*Riw*Rw2i*Ts_p
        + Cs*Qi_ext*Ris*Row1
        + Cs*Qi_ext*Ris*Rw1w2
        + Cs*Qi_ext*Ris*Rw2i
        + Cs*Qw1_ext*Ris*Row1
        + Cs*Qw2_ext*Ris*Row1
        + Cs*Qw2_ext*Ris*Rw1w2
        + Cs*Qw_ext*Ris*Row1
        + Cs*Qw_ext*Ris*Rw1w2
        + Cs*Qw_ext*Ris*Rw2i
        + Cs*Ris*To
        + Cs*Row1*Ts_p
        + Cs*Rw1w2*Ts_p
        + Cs*Rw2i*Ts_p
        + Cw*Qi_ext*Riw*Row1
        + Cw*Qi_ext*Riw*Rw1w2
        + Cw*Qi_ext*Riw*Rw2i
        + Cw*Qs_ext*Riw*Row1
        + Cw*Qs_ext*Riw*Rw1w2
        + Cw*Qs_ext*Riw*Rw2i
        + Cw*Qw1_ext*Riw*Row1
        + Cw*Qw2_ext*Riw*Row1
        + Cw*Qw2_ext*Riw*Rw1w2
        + Cw*Riw*To
        + Cw*Row1*Tw_p
        + Cw*Rw1w2*Tw_p
        + Cw*Rw2i*Tw_p
        + Qi_ext*Row1
        + Qi_ext*Rw1w2
        + Qi_ext*Rw2i
        + Qs_ext*Row1
        + Qs_ext*Rw1w2
        + Qs_ext*Rw2i
        + Qw1_ext*Row1
        + Qw2_ext*Row1
        + Qw2_ext*Rw1w2
        + Qw_ext*Row1
        + Qw_ext*Rw1w2
        + Qw_ext*Rw2i
        + To)/(Ci*Cs*Cw*Ris*Riw*Row1
        + Ci*Cs*Cw*Ris*Riw*Rw1w2
        + Ci*Cs*Cw*Ris*Riw*Rw2i
        + Ci*Cs*Ris*Row1
        + Ci*Cs*Ris*Rw1w2
        + Ci*Cs*Ris*Rw2i
        + Ci*Cw*Riw*Row1
        + Ci*Cw*Riw*Rw1w2
        + Ci*Cw*Riw*Rw2i
        + Ci*Row1
        + Ci*Rw1w2
        + Ci*Rw2i
        + Cs*Cw*Ris*Riw
        + Cs*Cw*Ris*Row1
        + Cs*Cw*Ris*Rw1w2
        + Cs*Cw*Ris*Rw2i
        + Cs*Cw*Riw*Row1
        + Cs*Cw*Riw*Rw1w2
        + Cs*Cw*Riw*Rw2i
        + Cs*Ris
        + Cs*Row1
        + Cs*Rw1w2
        + Cs*Rw2i
        + Cw*Riw
        + Cw*Row1
        + Cw*Rw1w2
        + Cw*Rw2i
        + 1))
    Tw = ((Cw*Riw*Tw_p
        + Qw_ext*Riw
        + Ti)/(Cw*Riw
        + 1))
    Ts = ((Cs*Ris*Ts_p
        + Qs_ext*Ris
        + Ti)/(Cs*Ris
        + 1))

    return [Ti, Ts, Tw]

def R6C3(i, Ti_p, Tw_p, Ts_p, To, Qw1_ext, Qw2_ext, Qi_ext, Qw_ext, Qs_ext, param):
    '''Auto-generated reduced-order R6C3 model.'''

    # parsing parameter
    timestep = param["timestep"]
    Rw2i = param["Rw2i"]
    Row1 = param["Row1"]
    Ris = param["Ris"]
    Roi = param["Roi"]
    Riw = param["Riw"]
    Rw1w2 = param["Rw1w2"]
    Cs = param["Cs"] / timestep
    Ci = param["Ci"] / timestep
    Cw = param["Cw"] / timestep

    # initialization
    if i == 0:
        return [Ti_p, Tw_p, Ts_p]

    # model equations
    Ti = ((Ci*Cs*Cw*Ris*Riw*Roi*Row1*Ti_p
        + Ci*Cs*Cw*Ris*Riw*Roi*Rw1w2*Ti_p
        + Ci*Cs*Cw*Ris*Riw*Roi*Rw2i*Ti_p
        + Ci*Cs*Ris*Roi*Row1*Ti_p
        + Ci*Cs*Ris*Roi*Rw1w2*Ti_p
        + Ci*Cs*Ris*Roi*Rw2i*Ti_p
        + Ci*Cw*Riw*Roi*Row1*Ti_p
        + Ci*Cw*Riw*Roi*Rw1w2*Ti_p
        + Ci*Cw*Riw*Roi*Rw2i*Ti_p
        + Ci*Roi*Row1*Ti_p
        + Ci*Roi*Rw1w2*Ti_p
        + Ci*Roi*Rw2i*Ti_p
        + Cs*Cw*Qi_ext*Ris*Riw*Roi*Row1
        + Cs*Cw*Qi_ext*Ris*Riw*Roi*Rw1w2
        + Cs*Cw*Qi_ext*Ris*Riw*Roi*Rw2i
        + Cs*Cw*Qw1_ext*Ris*Riw*Roi*Row1
        + Cs*Cw*Qw2_ext*Ris*Riw*Roi*Row1
        + Cs*Cw*Qw2_ext*Ris*Riw*Roi*Rw1w2
        + Cs*Cw*Ris*Riw*Roi*To
        + Cs*Cw*Ris*Riw*Row1*To
        + Cs*Cw*Ris*Riw*Rw1w2*To
        + Cs*Cw*Ris*Riw*Rw2i*To
        + Cs*Cw*Ris*Roi*Row1*Tw_p
        + Cs*Cw*Ris*Roi*Rw1w2*Tw_p
        + Cs*Cw*Ris*Roi*Rw2i*Tw_p
        + Cs*Cw*Riw*Roi*Row1*Ts_p
        + Cs*Cw*Riw*Roi*Rw1w2*Ts_p
        + Cs*Cw*Riw*Roi*Rw2i*Ts_p
        + Cs*Qi_ext*Ris*Roi*Row1
        + Cs*Qi_ext*Ris*Roi*Rw1w2
        + Cs*Qi_ext*Ris*Roi*Rw2i
        + Cs*Qw1_ext*Ris*Roi*Row1
        + Cs*Qw2_ext*Ris*Roi*Row1
        + Cs*Qw2_ext*Ris*Roi*Rw1w2
        + Cs*Qw_ext*Ris*Roi*Row1
        + Cs*Qw_ext*Ris*Roi*Rw1w2
        + Cs*Qw_ext*Ris*Roi*Rw2i
        + Cs*Ris*Roi*To
        + Cs*Ris*Row1*To
        + Cs*Ris*Rw1w2*To
        + Cs*Ris*Rw2i*To
        + Cs*Roi*Row1*Ts_p
        + Cs*Roi*Rw1w2*Ts_p
        + Cs*Roi*Rw2i*Ts_p
        + Cw*Qi_ext*Riw*Roi*Row1
        + Cw*Qi_ext*Riw*Roi*Rw1w2
        + Cw*Qi_ext*Riw*Roi*Rw2i
        + Cw*Qs_ext*Riw*Roi*Row1
        + Cw*Qs_ext*Riw*Roi*Rw1w2
        + Cw*Qs_ext*Riw*Roi*Rw2i
        + Cw*Qw1_ext*Riw*Roi*Row1
        + Cw*Qw2_ext*Riw*Roi*Row1
        + Cw*Qw2_ext*Riw*Roi*Rw1w2
        + Cw*Riw*Roi*To
        + Cw*Riw*Row1*To
        + Cw*Riw*Rw1w2*To
        + Cw*Riw*Rw2i*To
        + Cw*Roi*Row1*Tw_p
        + Cw*Roi*Rw1w2*Tw_p
        + Cw*Roi*Rw2i*Tw_p
        + Qi_ext*Roi*Row1
        + Qi_ext*Roi*Rw1w2
        + Qi_ext*Roi*Rw2i
        + Qs_ext*Roi*Row1
        + Qs_ext*Roi*Rw1w2
        + Qs_ext*Roi*Rw2i
        + Qw1_ext*Roi*Row1
        + Qw2_ext*Roi*Row1
        + Qw2_ext*Roi*Rw1w2
        + Qw_ext*Roi*Row1
        + Qw_ext*Roi*Rw1w2
        + Qw_ext*Roi*Rw2i
        + Roi*To
        + Row1*To
        + Rw1w2*To
        + Rw2i*To)/(Ci*Cs*Cw*Ris*Riw*Roi*Row1
        + Ci*Cs*Cw*Ris*Riw*Roi*Rw1w2
        + Ci*Cs*Cw*Ris*Riw*Roi*Rw2i
        + Ci*Cs*Ris*Roi*Row1
        + Ci*Cs*Ris*Roi*Rw1w2
        + Ci*Cs*Ris*Roi*Rw2i
        + Ci*Cw*Riw*Roi*Row1
        + Ci*Cw*Riw*Roi*Rw1w2
        + Ci*Cw*Riw*Roi*Rw2i
        + Ci*Roi*Row1
        + Ci*Roi*Rw1w2
        + Ci*Roi*Rw2i
        + Cs*Cw*Ris*Riw*Roi
        + Cs*Cw*Ris*Riw*Row1
        + Cs*Cw*Ris*Riw*Rw1w2
        + Cs*Cw*Ris*Riw*Rw2i
        + Cs*Cw*Ris*Roi*Row1
        + Cs*Cw*Ris*Roi*Rw1w2
        + Cs*Cw*Ris*Roi*Rw2i
        + Cs*Cw*Riw*Roi*Row1
        + Cs*Cw*Riw*Roi*Rw1w2
        + Cs*Cw*Riw*Roi*Rw2i
        + Cs*Ris*Roi
        + Cs*Ris*Row1
        + Cs*Ris*Rw1w2
        + Cs*Ris*Rw2i
        + Cs*Roi*Row1
        + Cs*Roi*Rw1w2
        + Cs*Roi*Rw2i
        + Cw*Riw*Roi
        + Cw*Riw*Row1
        + Cw*Riw*Rw1w2
        + Cw*Riw*Rw2i
        + Cw*Roi*Row1
        + Cw*Roi*Rw1w2
        + Cw*Roi*Rw2i
        + Roi
        + Row1
        + Rw1w2
        + Rw2i))
    Tw = ((Cw*Riw*Tw_p
        + Qw_ext*Riw
        + Ti)/(Cw*Riw
        + 1))
    Ts = ((Cs*Ris*Ts_p
        + Qs_ext*Ris
        + Ti)/(Cs*Ris
        + 1))

    return [Ti, Ts, Tw]

def R7C4(i, Ti_p, Tw_p, Ts_p, Tf_p, To, Qw1_ext, Qw2_ext, Qi_ext, Qw_ext, Qs_ext, Qf_ext, param):
    '''Auto-generated reduced-order R7C4 model.'''

    # parsing parameter
    timestep = param["timestep"]
    Rof = param["Rof"]
    Rw2i = param["Rw2i"]
    Ris = param["Ris"]
    Rw1w2 = param["Rw1w2"]
    Rfi = param["Rfi"]
    Row1 = param["Row1"]
    Riw = param["Riw"]
    Cf = param["Cf"] / timestep
    Cs = param["Cs"] / timestep
    Ci = param["Ci"] / timestep
    Cw = param["Cw"] / timestep

    # initialization
    if i == 0:
        return [Ti_p, Tw_p, Ts_p, Tf_p]

    # model equations
    Ti = ((Cf*Ci*Cs*Cw*Rfi*Ris*Riw*Rof*Row1*Ti_p
        + Cf*Ci*Cs*Cw*Rfi*Ris*Riw*Rof*Rw1w2*Ti_p
        + Cf*Ci*Cs*Cw*Rfi*Ris*Riw*Rof*Rw2i*Ti_p
        + Cf*Ci*Cs*Rfi*Ris*Rof*Row1*Ti_p
        + Cf*Ci*Cs*Rfi*Ris*Rof*Rw1w2*Ti_p
        + Cf*Ci*Cs*Rfi*Ris*Rof*Rw2i*Ti_p
        + Cf*Ci*Cw*Rfi*Riw*Rof*Row1*Ti_p
        + Cf*Ci*Cw*Rfi*Riw*Rof*Rw1w2*Ti_p
        + Cf*Ci*Cw*Rfi*Riw*Rof*Rw2i*Ti_p
        + Cf*Ci*Rfi*Rof*Row1*Ti_p
        + Cf*Ci*Rfi*Rof*Rw1w2*Ti_p
        + Cf*Ci*Rfi*Rof*Rw2i*Ti_p
        + Cf*Cs*Cw*Qi_ext*Rfi*Ris*Riw*Rof*Row1
        + Cf*Cs*Cw*Qi_ext*Rfi*Ris*Riw*Rof*Rw1w2
        + Cf*Cs*Cw*Qi_ext*Rfi*Ris*Riw*Rof*Rw2i
        + Cf*Cs*Cw*Qw1_ext*Rfi*Ris*Riw*Rof*Row1
        + Cf*Cs*Cw*Qw2_ext*Rfi*Ris*Riw*Rof*Row1
        + Cf*Cs*Cw*Qw2_ext*Rfi*Ris*Riw*Rof*Rw1w2
        + Cf*Cs*Cw*Rfi*Ris*Riw*Rof*To
        + Cf*Cs*Cw*Rfi*Ris*Rof*Row1*Tw_p
        + Cf*Cs*Cw*Rfi*Ris*Rof*Rw1w2*Tw_p
        + Cf*Cs*Cw*Rfi*Ris*Rof*Rw2i*Tw_p
        + Cf*Cs*Cw*Rfi*Riw*Rof*Row1*Ts_p
        + Cf*Cs*Cw*Rfi*Riw*Rof*Rw1w2*Ts_p
        + Cf*Cs*Cw*Rfi*Riw*Rof*Rw2i*Ts_p
        + Cf*Cs*Cw*Ris*Riw*Rof*Row1*Tf_p
        + Cf*Cs*Cw*Ris*Riw*Rof*Rw1w2*Tf_p
        + Cf*Cs*Cw*Ris*Riw*Rof*Rw2i*Tf_p
        + Cf*Cs*Qi_ext*Rfi*Ris*Rof*Row1
        + Cf*Cs*Qi_ext*Rfi*Ris*Rof*Rw1w2
        + Cf*Cs*Qi_ext*Rfi*Ris*Rof*Rw2i
        + Cf*Cs*Qw1_ext*Rfi*Ris*Rof*Row1
        + Cf*Cs*Qw2_ext*Rfi*Ris*Rof*Row1
        + Cf*Cs*Qw2_ext*Rfi*Ris*Rof*Rw1w2
        + Cf*Cs*Qw_ext*Rfi*Ris*Rof*Row1
        + Cf*Cs*Qw_ext*Rfi*Ris*Rof*Rw1w2
        + Cf*Cs*Qw_ext*Rfi*Ris*Rof*Rw2i
        + Cf*Cs*Rfi*Ris*Rof*To
        + Cf*Cs*Rfi*Rof*Row1*Ts_p
        + Cf*Cs*Rfi*Rof*Rw1w2*Ts_p
        + Cf*Cs*Rfi*Rof*Rw2i*Ts_p
        + Cf*Cs*Ris*Rof*Row1*Tf_p
        + Cf*Cs*Ris*Rof*Rw1w2*Tf_p
        + Cf*Cs*Ris*Rof*Rw2i*Tf_p
        + Cf*Cw*Qi_ext*Rfi*Riw*Rof*Row1
        + Cf*Cw*Qi_ext*Rfi*Riw*Rof*Rw1w2
        + Cf*Cw*Qi_ext*Rfi*Riw*Rof*Rw2i
        + Cf*Cw*Qs_ext*Rfi*Riw*Rof*Row1
        + Cf*Cw*Qs_ext*Rfi*Riw*Rof*Rw1w2
        + Cf*Cw*Qs_ext*Rfi*Riw*Rof*Rw2i
        + Cf*Cw*Qw1_ext*Rfi*Riw*Rof*Row1
        + Cf*Cw*Qw2_ext*Rfi*Riw*Rof*Row1
        + Cf*Cw*Qw2_ext*Rfi*Riw*Rof*Rw1w2
        + Cf*Cw*Rfi*Riw*Rof*To
        + Cf*Cw*Rfi*Rof*Row1*Tw_p
        + Cf*Cw*Rfi*Rof*Rw1w2*Tw_p
        + Cf*Cw*Rfi*Rof*Rw2i*Tw_p
        + Cf*Cw*Riw*Rof*Row1*Tf_p
        + Cf*Cw*Riw*Rof*Rw1w2*Tf_p
        + Cf*Cw*Riw*Rof*Rw2i*Tf_p
        + Cf*Qi_ext*Rfi*Rof*Row1
        + Cf*Qi_ext*Rfi*Rof*Rw1w2
        + Cf*Qi_ext*Rfi*Rof*Rw2i
        + Cf*Qs_ext*Rfi*Rof*Row1
        + Cf*Qs_ext*Rfi*Rof*Rw1w2
        + Cf*Qs_ext*Rfi*Rof*Rw2i
        + Cf*Qw1_ext*Rfi*Rof*Row1
        + Cf*Qw2_ext*Rfi*Rof*Row1
        + Cf*Qw2_ext*Rfi*Rof*Rw1w2
        + Cf*Qw_ext*Rfi*Rof*Row1
        + Cf*Qw_ext*Rfi*Rof*Rw1w2
        + Cf*Qw_ext*Rfi*Rof*Rw2i
        + Cf*Rfi*Rof*To
        + Cf*Rof*Row1*Tf_p
        + Cf*Rof*Rw1w2*Tf_p
        + Cf*Rof*Rw2i*Tf_p
        + Ci*Cs*Cw*Rfi*Ris*Riw*Row1*Ti_p
        + Ci*Cs*Cw*Rfi*Ris*Riw*Rw1w2*Ti_p
        + Ci*Cs*Cw*Rfi*Ris*Riw*Rw2i*Ti_p
        + Ci*Cs*Cw*Ris*Riw*Rof*Row1*Ti_p
        + Ci*Cs*Cw*Ris*Riw*Rof*Rw1w2*Ti_p
        + Ci*Cs*Cw*Ris*Riw*Rof*Rw2i*Ti_p
        + Ci*Cs*Rfi*Ris*Row1*Ti_p
        + Ci*Cs*Rfi*Ris*Rw1w2*Ti_p
        + Ci*Cs*Rfi*Ris*Rw2i*Ti_p
        + Ci*Cs*Ris*Rof*Row1*Ti_p
        + Ci*Cs*Ris*Rof*Rw1w2*Ti_p
        + Ci*Cs*Ris*Rof*Rw2i*Ti_p
        + Ci*Cw*Rfi*Riw*Row1*Ti_p
        + Ci*Cw*Rfi*Riw*Rw1w2*Ti_p
        + Ci*Cw*Rfi*Riw*Rw2i*Ti_p
        + Ci*Cw*Riw*Rof*Row1*Ti_p
        + Ci*Cw*Riw*Rof*Rw1w2*Ti_p
        + Ci*Cw*Riw*Rof*Rw2i*Ti_p
        + Ci*Rfi*Row1*Ti_p
        + Ci*Rfi*Rw1w2*Ti_p
        + Ci*Rfi*Rw2i*Ti_p
        + Ci*Rof*Row1*Ti_p
        + Ci*Rof*Rw1w2*Ti_p
        + Ci*Rof*Rw2i*Ti_p
        + Cs*Cw*Qf_ext*Ris*Riw*Rof*Row1
        + Cs*Cw*Qf_ext*Ris*Riw*Rof*Rw1w2
        + Cs*Cw*Qf_ext*Ris*Riw*Rof*Rw2i
        + Cs*Cw*Qi_ext*Rfi*Ris*Riw*Row1
        + Cs*Cw*Qi_ext*Rfi*Ris*Riw*Rw1w2
        + Cs*Cw*Qi_ext*Rfi*Ris*Riw*Rw2i
        + Cs*Cw*Qi_ext*Ris*Riw*Rof*Row1
        + Cs*Cw*Qi_ext*Ris*Riw*Rof*Rw1w2
        + Cs*Cw*Qi_ext*Ris*Riw*Rof*Rw2i
        + Cs*Cw*Qw1_ext*Rfi*Ris*Riw*Row1
        + Cs*Cw*Qw1_ext*Ris*Riw*Rof*Row1
        + Cs*Cw*Qw2_ext*Rfi*Ris*Riw*Row1
        + Cs*Cw*Qw2_ext*Rfi*Ris*Riw*Rw1w2
        + Cs*Cw*Qw2_ext*Ris*Riw*Rof*Row1
        + Cs*Cw*Qw2_ext*Ris*Riw*Rof*Rw1w2
        + Cs*Cw*Rfi*Ris*Riw*To
        + Cs*Cw*Rfi*Ris*Row1*Tw_p
        + Cs*Cw*Rfi*Ris*Rw1w2*Tw_p
        + Cs*Cw*Rfi*Ris*Rw2i*Tw_p
        + Cs*Cw*Rfi*Riw*Row1*Ts_p
        + Cs*Cw*Rfi*Riw*Rw1w2*Ts_p
        + Cs*Cw*Rfi*Riw*Rw2i*Ts_p
        + Cs*Cw*Ris*Riw*Rof*To
        + Cs*Cw*Ris*Riw*Row1*To
        + Cs*Cw*Ris*Riw*Rw1w2*To
        + Cs*Cw*Ris*Riw*Rw2i*To
        + Cs*Cw*Ris*Rof*Row1*Tw_p
        + Cs*Cw*Ris*Rof*Rw1w2*Tw_p
        + Cs*Cw*Ris*Rof*Rw2i*Tw_p
        + Cs*Cw*Riw*Rof*Row1*Ts_p
        + Cs*Cw*Riw*Rof*Rw1w2*Ts_p
        + Cs*Cw*Riw*Rof*Rw2i*Ts_p
        + Cs*Qf_ext*Ris*Rof*Row1
        + Cs*Qf_ext*Ris*Rof*Rw1w2
        + Cs*Qf_ext*Ris*Rof*Rw2i
        + Cs*Qi_ext*Rfi*Ris*Row1
        + Cs*Qi_ext*Rfi*Ris*Rw1w2
        + Cs*Qi_ext*Rfi*Ris*Rw2i
        + Cs*Qi_ext*Ris*Rof*Row1
        + Cs*Qi_ext*Ris*Rof*Rw1w2
        + Cs*Qi_ext*Ris*Rof*Rw2i
        + Cs*Qw1_ext*Rfi*Ris*Row1
        + Cs*Qw1_ext*Ris*Rof*Row1
        + Cs*Qw2_ext*Rfi*Ris*Row1
        + Cs*Qw2_ext*Rfi*Ris*Rw1w2
        + Cs*Qw2_ext*Ris*Rof*Row1
        + Cs*Qw2_ext*Ris*Rof*Rw1w2
        + Cs*Qw_ext*Rfi*Ris*Row1
        + Cs*Qw_ext*Rfi*Ris*Rw1w2
        + Cs*Qw_ext*Rfi*Ris*Rw2i
        + Cs*Qw_ext*Ris*Rof*Row1
        + Cs*Qw_ext*Ris*Rof*Rw1w2
        + Cs*Qw_ext*Ris*Rof*Rw2i
        + Cs*Rfi*Ris*To
        + Cs*Rfi*Row1*Ts_p
        + Cs*Rfi*Rw1w2*Ts_p
        + Cs*Rfi*Rw2i*Ts_p
        + Cs*Ris*Rof*To
        + Cs*Ris*Row1*To
        + Cs*Ris*Rw1w2*To
        + Cs*Ris*Rw2i*To
        + Cs*Rof*Row1*Ts_p
        + Cs*Rof*Rw1w2*Ts_p
        + Cs*Rof*Rw2i*Ts_p
        + Cw*Qf_ext*Riw*Rof*Row1
        + Cw*Qf_ext*Riw*Rof*Rw1w2
        + Cw*Qf_ext*Riw*Rof*Rw2i
        + Cw*Qi_ext*Rfi*Riw*Row1
        + Cw*Qi_ext*Rfi*Riw*Rw1w2
        + Cw*Qi_ext*Rfi*Riw*Rw2i
        + Cw*Qi_ext*Riw*Rof*Row1
        + Cw*Qi_ext*Riw*Rof*Rw1w2
        + Cw*Qi_ext*Riw*Rof*Rw2i
        + Cw*Qs_ext*Rfi*Riw*Row1
        + Cw*Qs_ext*Rfi*Riw*Rw1w2
        + Cw*Qs_ext*Rfi*Riw*Rw2i
        + Cw*Qs_ext*Riw*Rof*Row1
        + Cw*Qs_ext*Riw*Rof*Rw1w2
        + Cw*Qs_ext*Riw*Rof*Rw2i
        + Cw*Qw1_ext*Rfi*Riw*Row1
        + Cw*Qw1_ext*Riw*Rof*Row1
        + Cw*Qw2_ext*Rfi*Riw*Row1
        + Cw*Qw2_ext*Rfi*Riw*Rw1w2
        + Cw*Qw2_ext*Riw*Rof*Row1
        + Cw*Qw2_ext*Riw*Rof*Rw1w2
        + Cw*Rfi*Riw*To
        + Cw*Rfi*Row1*Tw_p
        + Cw*Rfi*Rw1w2*Tw_p
        + Cw*Rfi*Rw2i*Tw_p
        + Cw*Riw*Rof*To
        + Cw*Riw*Row1*To
        + Cw*Riw*Rw1w2*To
        + Cw*Riw*Rw2i*To
        + Cw*Rof*Row1*Tw_p
        + Cw*Rof*Rw1w2*Tw_p
        + Cw*Rof*Rw2i*Tw_p
        + Qf_ext*Rof*Row1
        + Qf_ext*Rof*Rw1w2
        + Qf_ext*Rof*Rw2i
        + Qi_ext*Rfi*Row1
        + Qi_ext*Rfi*Rw1w2
        + Qi_ext*Rfi*Rw2i
        + Qi_ext*Rof*Row1
        + Qi_ext*Rof*Rw1w2
        + Qi_ext*Rof*Rw2i
        + Qs_ext*Rfi*Row1
        + Qs_ext*Rfi*Rw1w2
        + Qs_ext*Rfi*Rw2i
        + Qs_ext*Rof*Row1
        + Qs_ext*Rof*Rw1w2
        + Qs_ext*Rof*Rw2i
        + Qw1_ext*Rfi*Row1
        + Qw1_ext*Rof*Row1
        + Qw2_ext*Rfi*Row1
        + Qw2_ext*Rfi*Rw1w2
        + Qw2_ext*Rof*Row1
        + Qw2_ext*Rof*Rw1w2
        + Qw_ext*Rfi*Row1
        + Qw_ext*Rfi*Rw1w2
        + Qw_ext*Rfi*Rw2i
        + Qw_ext*Rof*Row1
        + Qw_ext*Rof*Rw1w2
        + Qw_ext*Rof*Rw2i
        + Rfi*To
        + Rof*To
        + Row1*To
        + Rw1w2*To
        + Rw2i*To)/(Cf*Ci*Cs*Cw*Rfi*Ris*Riw*Rof*Row1
        + Cf*Ci*Cs*Cw*Rfi*Ris*Riw*Rof*Rw1w2
        + Cf*Ci*Cs*Cw*Rfi*Ris*Riw*Rof*Rw2i
        + Cf*Ci*Cs*Rfi*Ris*Rof*Row1
        + Cf*Ci*Cs*Rfi*Ris*Rof*Rw1w2
        + Cf*Ci*Cs*Rfi*Ris*Rof*Rw2i
        + Cf*Ci*Cw*Rfi*Riw*Rof*Row1
        + Cf*Ci*Cw*Rfi*Riw*Rof*Rw1w2
        + Cf*Ci*Cw*Rfi*Riw*Rof*Rw2i
        + Cf*Ci*Rfi*Rof*Row1
        + Cf*Ci*Rfi*Rof*Rw1w2
        + Cf*Ci*Rfi*Rof*Rw2i
        + Cf*Cs*Cw*Rfi*Ris*Riw*Rof
        + Cf*Cs*Cw*Rfi*Ris*Rof*Row1
        + Cf*Cs*Cw*Rfi*Ris*Rof*Rw1w2
        + Cf*Cs*Cw*Rfi*Ris*Rof*Rw2i
        + Cf*Cs*Cw*Rfi*Riw*Rof*Row1
        + Cf*Cs*Cw*Rfi*Riw*Rof*Rw1w2
        + Cf*Cs*Cw*Rfi*Riw*Rof*Rw2i
        + Cf*Cs*Cw*Ris*Riw*Rof*Row1
        + Cf*Cs*Cw*Ris*Riw*Rof*Rw1w2
        + Cf*Cs*Cw*Ris*Riw*Rof*Rw2i
        + Cf*Cs*Rfi*Ris*Rof
        + Cf*Cs*Rfi*Rof*Row1
        + Cf*Cs*Rfi*Rof*Rw1w2
        + Cf*Cs*Rfi*Rof*Rw2i
        + Cf*Cs*Ris*Rof*Row1
        + Cf*Cs*Ris*Rof*Rw1w2
        + Cf*Cs*Ris*Rof*Rw2i
        + Cf*Cw*Rfi*Riw*Rof
        + Cf*Cw*Rfi*Rof*Row1
        + Cf*Cw*Rfi*Rof*Rw1w2
        + Cf*Cw*Rfi*Rof*Rw2i
        + Cf*Cw*Riw*Rof*Row1
        + Cf*Cw*Riw*Rof*Rw1w2
        + Cf*Cw*Riw*Rof*Rw2i
        + Cf*Rfi*Rof
        + Cf*Rof*Row1
        + Cf*Rof*Rw1w2
        + Cf*Rof*Rw2i
        + Ci*Cs*Cw*Rfi*Ris*Riw*Row1
        + Ci*Cs*Cw*Rfi*Ris*Riw*Rw1w2
        + Ci*Cs*Cw*Rfi*Ris*Riw*Rw2i
        + Ci*Cs*Cw*Ris*Riw*Rof*Row1
        + Ci*Cs*Cw*Ris*Riw*Rof*Rw1w2
        + Ci*Cs*Cw*Ris*Riw*Rof*Rw2i
        + Ci*Cs*Rfi*Ris*Row1
        + Ci*Cs*Rfi*Ris*Rw1w2
        + Ci*Cs*Rfi*Ris*Rw2i
        + Ci*Cs*Ris*Rof*Row1
        + Ci*Cs*Ris*Rof*Rw1w2
        + Ci*Cs*Ris*Rof*Rw2i
        + Ci*Cw*Rfi*Riw*Row1
        + Ci*Cw*Rfi*Riw*Rw1w2
        + Ci*Cw*Rfi*Riw*Rw2i
        + Ci*Cw*Riw*Rof*Row1
        + Ci*Cw*Riw*Rof*Rw1w2
        + Ci*Cw*Riw*Rof*Rw2i
        + Ci*Rfi*Row1
        + Ci*Rfi*Rw1w2
        + Ci*Rfi*Rw2i
        + Ci*Rof*Row1
        + Ci*Rof*Rw1w2
        + Ci*Rof*Rw2i
        + Cs*Cw*Rfi*Ris*Riw
        + Cs*Cw*Rfi*Ris*Row1
        + Cs*Cw*Rfi*Ris*Rw1w2
        + Cs*Cw*Rfi*Ris*Rw2i
        + Cs*Cw*Rfi*Riw*Row1
        + Cs*Cw*Rfi*Riw*Rw1w2
        + Cs*Cw*Rfi*Riw*Rw2i
        + Cs*Cw*Ris*Riw*Rof
        + Cs*Cw*Ris*Riw*Row1
        + Cs*Cw*Ris*Riw*Rw1w2
        + Cs*Cw*Ris*Riw*Rw2i
        + Cs*Cw*Ris*Rof*Row1
        + Cs*Cw*Ris*Rof*Rw1w2
        + Cs*Cw*Ris*Rof*Rw2i
        + Cs*Cw*Riw*Rof*Row1
        + Cs*Cw*Riw*Rof*Rw1w2
        + Cs*Cw*Riw*Rof*Rw2i
        + Cs*Rfi*Ris
        + Cs*Rfi*Row1
        + Cs*Rfi*Rw1w2
        + Cs*Rfi*Rw2i
        + Cs*Ris*Rof
        + Cs*Ris*Row1
        + Cs*Ris*Rw1w2
        + Cs*Ris*Rw2i
        + Cs*Rof*Row1
        + Cs*Rof*Rw1w2
        + Cs*Rof*Rw2i
        + Cw*Rfi*Riw
        + Cw*Rfi*Row1
        + Cw*Rfi*Rw1w2
        + Cw*Rfi*Rw2i
        + Cw*Riw*Rof
        + Cw*Riw*Row1
        + Cw*Riw*Rw1w2
        + Cw*Riw*Rw2i
        + Cw*Rof*Row1
        + Cw*Rof*Rw1w2
        + Cw*Rof*Rw2i
        + Rfi
        + Rof
        + Row1
        + Rw1w2
        + Rw2i))
    Tw = ((Cw*Riw*Tw_p
        + Qw_ext*Riw
        + Ti)/(Cw*Riw
        + 1))
    Ts = ((Cs*Ris*Ts_p
        + Qs_ext*Ris
        + Ti)/(Cs*Ris
        + 1))
    Tf = ((Cf*Rfi*Rof*Tf_p
        + Qf_ext*Rfi*Rof
        + Rfi*To
        + Rof*Ti)/(Cf*Rfi*Rof
        + Rfi
        + Rof))

    return [Tf, Ti, Ts, Tw]
