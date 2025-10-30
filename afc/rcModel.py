# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Reduced order thermal model.

This script contains the most up to date version of the RCModel used to 
describe the building in electrochromic (EC) windows model predictive control
(MPC) simulations. Here are some important pieces of nomenclature for
understanding these models:
    
R = Resistance
C = Capacitance
Q = Heat addition, representing gains
Subscript w = wall and refers to exterior walls
Subscript w# = window, with 1 referring to the interior pane of a window
    and 2 referring the exterior pane of a window
Subscript s = slab
Subscript o = outdoor
Subscript i = indoor
Subscript p = Previous. E.g. "Ti_p" = Indoor temperature at the previous
    timestep
Subscript ext = ???? (E.g. def R1C1(i, Ti_p, To, Qi_ext, param):)
i = Current time (Not just an index)
"""

# pylint: disable=invalid-name, too-many-arguments, too-many-locals, too-many-statements
# pylint: disable=too-many-positional-arguments

RCTYPES = ['R1C1', 'R2C2', 'R4C2', 'R5C3', 'R6C3']

def R1C1(i, Ti_p, To, Qi_ext, param):
    """Function for RC model considering indoor, outdoor conditions.

    i = Time
    Ti_p = previous/initial indoor temperature
    To = Outdoor temperature
    Qi_ext = Indoor space heat gains
    param = Dictionary of parameters defining R & C values
    """

    timestep = param['timestep'] #Read the timestep from param
    R1 = param['Rw2i'] #Resistance between the outside and the inside
    C1 = param['Ci'] / timestep #Ci = Capacitance inside

    if i == 0: #If calculating at time = 0
        return Ti_p #Return initial indoor temperature

    #If not the first timestep
    Ti = (R1*Qi_ext + R1*C1*Ti_p + To) / (1 + R1*C1) #Calculate indoor
    #temperature at the next time step using R1C1 model
    return [Ti] #Return the calculated indoor temperature at this timestep

def R2C2(i, T1_prev, T2_prev, T_out, Q1_ext, Q2_ext, param):
    """Function for RC model considering indoor, outdoor, slab conditions.
    Internal gains applied to both indoor and slab

    T1 = indoor air temperature
    T2 = slab temperatue
    T_out = outdoor temperature
    Q1_ext = internal gains in space
    Q2_ext = internal gains in slab
    """

    timestep = param['timestep'] #Read the timestep
    R1 = param['Rw2i'] #Read the resistane between indoor and outdoor
    C1 = param['Ci'] / timestep #Calculate the indoor air C/dt
    R2 = param['Ris'] #Read the resistance between indoor and the slab
    C2 = param['Cs'] / timestep #Calculate the slab C/dt

    if i == 0: #If time = 0
        return T1_prev, T2_prev #Return initial temperatures of indoor, slab

    #If not first time step perform calculations
    a = C1*C2*R1*R2*T2_prev
    a += C1*Q2_ext*R1*R2
    a += C1*R1*T1_prev
    a += C2*R1*T2_prev
    a += C2*R2*T2_prev
    a += Q1_ext*R1
    a += Q2_ext*R1
    a += Q2_ext*R2

    b = C1*C2*R1*R2
    b += C1*R1
    b += C2*R1
    b += C2*R2

    T2 = (a + T_out) / (b + 1) #Calculate final slab temperature
    T1 = C2*R2*T2 - C2*R2*T2_prev - Q2_ext*R2 + T2 #Calculate final indoor
    #temperature
    return T1, T2 #Return indoor and slab temperatures

def R4C2(i, Ti_p, Ts_p, To, Qw1_ext, Qw2_ext, Qi_ext, Qs_ext, param):
    """Function for RC model considering indoor, outdoor, slab, and a window.
    Internal gains applied to indoor, slab, 2x window panes

    i = Time
    Ti_p = Previous/initial indoor temperature
    Ts_p = Previous/initial slab temperature
    To = Outdoor temperature
    Qw1_ext = Heat gain in the external window pane
    Qw2_ext = Heat gain in the internal window pane
    Qi_ext = Indoor space heat gains
    Qs_ext = Slab heat gains
    param = Dictionary of parameters defining R & C values
    """

    timestep = param['timestep'] #Read the timestep
    Row1 = param['Row1'] #Read the resistance between outdoor and window 1
    Rw1w2 = param['Rw1w2'] #Read the resistance between windows 1 and 2
    Rw2i = param['Rw2i'] #Read the resistance between window 2 and indoor
    Ci = param['Ci'] / timestep #Calculate the indoor C/dt
    Ris = param['Ris'] #Read the resistance between indoor and slab
    Cs = param['Cs'] / timestep #Calculate the slab C/dt

    if i == 0: #If time = 0
        return Ti_p, Ts_p #Return initial indoor and slab temperatures

    #If not first timestep, perform calculations
    a = Ci*Cs*Ris*Row1*Ts_p
    a += Ci*Cs*Ris*Rw1w2*Ts_p
    a += Ci*Cs*Ris*Rw2i*Ts_p
    a += Ci*Qs_ext*Ris*Row1
    a += Ci*Qs_ext*Ris*Rw1w2
    a += Ci*Qs_ext*Ris*Rw2i
    a += Ci*Row1*Ti_p + Ci*Rw1w2*Ti_p
    a += Ci*Rw2i*Ti_p + Cs*Ris*Ts_p
    a += Cs*Row1*Ts_p + Cs*Rw1w2*Ts_p
    a += Cs*Rw2i*Ts_p + Qi_ext*Row1
    a += Qi_ext*Rw1w2 + Qi_ext*Rw2i
    a += Qs_ext*Ris + Qs_ext*Row1
    a += Qs_ext*Rw1w2 + Qs_ext*Rw2i
    a += Qw1_ext*Row1 + Qw2_ext*Row1
    a += Qw2_ext*Rw1w2 + To

    b = Ci*Cs*Ris*Row1
    b += Ci*Cs*Ris*Rw1w2
    b += Ci*Cs*Ris*Rw2i
    b += Ci*Row1 + Ci*Rw1w2
    b += Ci*Rw2i + Cs*Ris
    b += Cs*Row1 + Cs*Rw1w2
    b += Cs*Rw2i + 1

    Ts = a / b #Calculate slab temperature
    Ti = Cs*Ris*Ts - Cs*Ris*Ts_p - Qs_ext*Ris + Ts #Calculate indoor temp
    return Ti, Ts #Return indoor and slab temperatures

def R5C3(i, Ti_p, Ts_p, Tw_p, To, Qw1_ext, Qw2_ext,
         Qi_ext, Qs_ext, Qw_ext, param):
    """Function for RC model considering indoor, slab, 1 double-pane window, one wall.
    Applying internal gains to both window panes, interior, slab, wall

    i = Time
    Ti_p = Previous/initial indoor temperature
    Ts_p = Previous/initial slab temperature
    Tw_p = Previous/initial wall temperature
    To = Outdoor temperature
    Qw1_ext = Heat gain in the external window pane
    Qw2_ext = Heat gain in the internal window pane
    Qi_ext = Indoor space heat gains
    Qs_ext = Slab heat gains
    Qw_ext = Wall heat gains
    param = Dictionary of parameters defining R & C values
    """

    timestep = param['timestep'] #Read timestep
    Row1 = param['Row1'] #Read resistance of the outdoor window pane
    Rw1w2 = param['Rw1w2'] #Read resistance between the window panes
    Rw2i = param['Rw2i'] #Read resistance of the indoor window pane
    Ci = param['Ci'] / timestep #Calculate indoor C/dt
    Ris = param['Ris'] #Read resistance between indoor and slab
    Cs = param['Cs'] / timestep #Calculate slab C/dt
    Riw = param['Riw'] #Read resistance between indoor and wall
    Cw = param['Cw'] / timestep #Calculate wall C/dt

    if i == 0: #If time = 0
        return Ti_p, Ts_p, Tw_p #Return indoor, slab, wall temperatures

    #If not first timestep perform calculations
    a = Ci*Cs*Cw*Ris*Riw*Row1*Ts_p + Ci*Cs*Cw*Ris*Riw*Rw1w2*Ts_p
    a += Ci*Cs*Cw*Ris*Riw*Rw2i*Ts_p + Ci*Cs*Ris*Row1*Ts_p
    a += Ci*Cs*Ris*Rw1w2*Ts_p + Ci*Cs*Ris*Rw2i*Ts_p
    a += Ci*Cw*Qs_ext*Ris*Riw*Row1 + Ci*Cw*Qs_ext*Ris*Riw*Rw1w2
    a += Ci*Cw*Qs_ext*Ris*Riw*Rw2i + Ci*Cw*Riw*Row1*Ti_p
    a += Ci*Cw*Riw*Rw1w2*Ti_p + Ci*Cw*Riw*Rw2i*Ti_p
    a += Ci*Qs_ext*Ris*Row1 + Ci*Qs_ext*Ris*Rw1w2
    a += Ci*Qs_ext*Ris*Rw2i + Ci*Row1*Ti_p + Ci*Rw1w2*Ti_p
    a += Ci*Rw2i*Ti_p + Cs*Cw*Ris*Riw*Ts_p + Cs*Cw*Ris*Row1*Ts_p
    a += Cs*Cw*Ris*Rw1w2*Ts_p + Cs*Cw*Ris*Rw2i*Ts_p
    a += Cs*Cw*Riw*Row1*Ts_p + Cs*Cw*Riw*Rw1w2*Ts_p
    a += Cs*Cw*Riw*Rw2i*Ts_p + Cs*Ris*Ts_p
    a += Cs*Row1*Ts_p + Cs*Rw1w2*Ts_p + Cs*Rw2i*Ts_p
    a += Cw*Qi_ext*Riw*Row1 + Cw*Qi_ext*Riw*Rw1w2
    a += Cw*Qi_ext*Riw*Rw2i + Cw*Qs_ext*Ris*Riw
    a += Cw*Qs_ext*Ris*Row1 + Cw*Qs_ext*Ris*Rw1w2
    a += Cw*Qs_ext*Ris*Rw2i + Cw*Qs_ext*Riw*Row1
    a += Cw*Qs_ext*Riw*Rw1w2 + Cw*Qs_ext*Riw*Rw2i
    a += Cw*Qw1_ext*Riw*Row1 + Cw*Qw2_ext*Riw*Row1
    a += Cw*Qw2_ext*Riw*Rw1w2 + Cw*Riw*To + Cw*Row1*Tw_p
    a += Cw*Rw1w2*Tw_p + Cw*Rw2i*Tw_p + Qi_ext*Row1
    a += Qi_ext*Rw1w2 + Qi_ext*Rw2i + Qs_ext*Ris
    a += Qs_ext*Row1 + Qs_ext*Rw1w2 + Qs_ext*Rw2i
    a += Qw1_ext*Row1 + Qw2_ext*Row1 + Qw2_ext*Rw1w2
    a += Qw_ext*Row1 + Qw_ext*Rw1w2 + Qw_ext*Rw2i + To

    b = Ci*Cs*Cw*Ris*Riw*Row1 + Ci*Cs*Cw*Ris*Riw*Rw1w2
    b += Ci*Cs*Cw*Ris*Riw*Rw2i + Ci*Cs*Ris*Row1
    b += Ci*Cs*Ris*Rw1w2 + Ci*Cs*Ris*Rw2i
    b += Ci*Cw*Riw*Row1 + Ci*Cw*Riw*Rw1w2
    b += Ci*Cw*Riw*Rw2i + Ci*Row1 + Ci*Rw1w2
    b += Ci*Rw2i + Cs*Cw*Ris*Riw + Cs*Cw*Ris*Row1
    b += Cs*Cw*Ris*Rw1w2 + Cs*Cw*Ris*Rw2i + Cs*Cw*Riw*Row1
    b += Cs*Cw*Riw*Rw1w2 + Cs*Cw*Riw*Rw2i + Cs*Ris
    b += Cs*Row1 + Cs*Rw1w2 + Cs*Rw2i + Cw*Riw
    b += Cw*Row1 + Cw*Rw1w2 + Cw*Rw2i + 1

    Ts = a / b
    Ti = Cs*Ris*Ts - Cs*Ris*Ts_p - Qs_ext*Ris + Ts
    Tw = (Cw*Riw*Tw_p + Qw_ext*Riw + Ti) / (Cw*Riw + 1)
    return Ti, Ts, Tw #Return indoor, slab, wall temperatures

def R6C3(i, Ti_p, Ts_p, Tw_p, To, Qw1_ext, Qw2_ext,
         Qi_ext, Qs_ext, Qw_ext, param):
    """Function for RC model considering indoor, slab, 1 double-pane window, one wall.
    Applying internal gains to both window panes, interior, slab, wall

    i = Time
    Ti_p = Previous/initial indoor temperature
    Ts_p = Previous/initial slab temperature
    Tw_p = Previous/initial wall temperature
    To = Outdoor temperature
    Qw1_ext = Heat gain in the external window pane
    Qw2_ext = Heat gain in the internal window pane
    Qi_ext = Indoor space heat gains
    Qs_ext = Slab heat gains
    Qw_ext = Wall heat gains
    param = Dictionary of parameters defining R & C values
    """

    timestep = param['timestep'] #Read timestep
    Roi = param['Roi'] #Read resistance between indoor and outdoor
    Row1 = param['Row1'] #Read resistance between outdoor and window 1
    Rw1w2 = param['Rw1w2'] #Read resistance...across both windows?
    Rw2i = param['Rw2i'] #Read resistance between window 2 and indoor
    Ci = param['Ci'] / timestep #Calculate indoor C/dt
    Ris = param['Ris'] #Read resistance between slab and indoor
    Cs = param['Cs'] / timestep #Calculate slab C/dt
    Riw = param['Riw'] #Read resistance between indoor and wall
    Cw = param['Cw'] / timestep #Calculate wall C/dt

    if i == 0: #If time = 0
        return Ti_p, Ts_p, Tw_p #Return initial conditions

    #If time != 0 perform calculations
    a = Ci*Cs*Cw*Ris*Riw*Roi*Row1*Ts_p + Ci*Cs*Cw*Ris*Riw*Roi*Rw1w2*Ts_p
    a += Ci*Cs*Cw*Ris*Riw*Roi*Rw2i*Ts_p + Ci*Cs*Ris*Roi*Row1*Ts_p
    a += Ci*Cs*Ris*Roi*Rw1w2*Ts_p + Ci*Cs*Ris*Roi*Rw2i*Ts_p
    a += Ci*Cw*Qs_ext*Ris*Riw*Roi*Row1 + Ci*Cw*Qs_ext*Ris*Riw*Roi*Rw1w2
    a += Ci*Cw*Qs_ext*Ris*Riw*Roi*Rw2i + Ci*Cw*Riw*Roi*Row1*Ti_p
    a += Ci*Cw*Riw*Roi*Rw1w2*Ti_p + Ci*Cw*Riw*Roi*Rw2i*Ti_p
    a += Ci*Qs_ext*Ris*Roi*Row1 + Ci*Qs_ext*Ris*Roi*Rw1w2
    a += Ci*Qs_ext*Ris*Roi*Rw2i + Ci*Roi*Row1*Ti_p
    a += Ci*Roi*Rw1w2*Ti_p + Ci*Roi*Rw2i*Ti_p
    a += Cs*Cw*Ris*Riw*Roi*Ts_p + Cs*Cw*Ris*Riw*Row1*Ts_p
    a += Cs*Cw*Ris*Riw*Rw1w2*Ts_p + Cs*Cw*Ris*Riw*Rw2i*Ts_p
    a += Cs*Cw*Ris*Roi*Row1*Ts_p + Cs*Cw*Ris*Roi*Rw1w2*Ts_p
    a += Cs*Cw*Ris*Roi*Rw2i*Ts_p + Cs*Cw*Riw*Roi*Row1*Ts_p
    a += Cs*Cw*Riw*Roi*Rw1w2*Ts_p + Cs*Cw*Riw*Roi*Rw2i*Ts_p
    a += Cs*Ris*Roi*Ts_p + Cs*Ris*Row1*Ts_p + Cs*Ris*Rw1w2*Ts_p
    a += Cs*Ris*Rw2i*Ts_p + Cs*Roi*Row1*Ts_p + Cs*Roi*Rw1w2*Ts_p
    a += Cs*Roi*Rw2i*Ts_p + Cw*Qi_ext*Riw*Roi*Row1
    a += Cw*Qi_ext*Riw*Roi*Rw1w2 + Cw*Qi_ext*Riw*Roi*Rw2i
    a += Cw*Qs_ext*Ris*Riw*Roi + Cw*Qs_ext*Ris*Riw*Row1
    a += Cw*Qs_ext*Ris*Riw*Rw1w2 + Cw*Qs_ext*Ris*Riw*Rw2i
    a += Cw*Qs_ext*Ris*Roi*Row1 + Cw*Qs_ext*Ris*Roi*Rw1w2
    a += Cw*Qs_ext*Ris*Roi*Rw2i + Cw*Qs_ext*Riw*Roi*Row1
    a += Cw*Qs_ext*Riw*Roi*Rw1w2 + Cw*Qs_ext*Riw*Roi*Rw2i
    a += Cw*Qw1_ext*Riw*Roi*Row1 + Cw*Qw2_ext*Riw*Roi*Row1
    a += Cw*Qw2_ext*Riw*Roi*Rw1w2 + Cw*Riw*Roi*To + Cw*Riw*Row1*To
    a += Cw*Riw*Rw1w2*To + Cw*Riw*Rw2i*To + Cw*Roi*Row1*Tw_p
    a += Cw*Roi*Rw1w2*Tw_p + Cw*Roi*Rw2i*Tw_p + Qi_ext*Roi*Row1
    a += Qi_ext*Roi*Rw1w2 + Qi_ext*Roi*Rw2i + Qs_ext*Ris*Roi
    a += Qs_ext*Ris*Row1 + Qs_ext*Ris*Rw1w2 + Qs_ext*Ris*Rw2i
    a += Qs_ext*Roi*Row1 + Qs_ext*Roi*Rw1w2 + Qs_ext*Roi*Rw2i
    a += Qw1_ext*Roi*Row1 + Qw2_ext*Roi*Row1 + Qw2_ext*Roi*Rw1w2
    a += Qw_ext*Roi*Row1 + Qw_ext*Roi*Rw1w2 + Qw_ext*Roi*Rw2i
    a += Roi*To + Row1*To + Rw1w2*To + Rw2i*To

    b = Ci*Cs*Cw*Ris*Riw*Roi*Row1 + Ci*Cs*Cw*Ris*Riw*Roi*Rw1w2
    b += Ci*Cs*Cw*Ris*Riw*Roi*Rw2i + Ci*Cs*Ris*Roi*Row1
    b += Ci*Cs*Ris*Roi*Rw1w2 + Ci*Cs*Ris*Roi*Rw2i
    b += Ci*Cw*Riw*Roi*Row1 + Ci*Cw*Riw*Roi*Rw1w2
    b += Ci*Cw*Riw*Roi*Rw2i + Ci*Roi*Row1 + Ci*Roi*Rw1w2
    b += Ci*Roi*Rw2i + Cs*Cw*Ris*Riw*Roi + Cs*Cw*Ris*Riw*Row1
    b += Cs*Cw*Ris*Riw*Rw1w2 + Cs*Cw*Ris*Riw*Rw2i + Cs*Cw*Ris*Roi*Row1
    b += Cs*Cw*Ris*Roi*Rw1w2 + Cs*Cw*Ris*Roi*Rw2i + Cs*Cw*Riw*Roi*Row1
    b += Cs*Cw*Riw*Roi*Rw1w2 + Cs*Cw*Riw*Roi*Rw2i + Cs*Ris*Roi
    b += Cs*Ris*Row1 + Cs*Ris*Rw1w2 + Cs*Ris*Rw2i + Cs*Roi*Row1
    b += Cs*Roi*Rw1w2 + Cs*Roi*Rw2i + Cw*Riw*Roi + Cw*Riw*Row1
    b += Cw*Riw*Rw1w2 + Cw*Riw*Rw2i + Cw*Roi*Row1 + Cw*Roi*Rw1w2
    b += Cw*Roi*Rw2i + Roi + Row1 + Rw1w2 + Rw2i

    Ts = a / b
    Ti = Cs*Ris*Ts - Cs*Ris*Ts_p - Qs_ext*Ris + Ts
    Tw = (Cw*Riw*Tw_p + Qw_ext*Riw + Ti) / (Cw*Riw + 1)
    return Ti, Ts, Tw #Return indoor, slab, wall temperatures
