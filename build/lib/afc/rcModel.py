#This script contains the most up to date version of the RCModel used to 
#describe the building in electrochromic (EC) windows model predictive control
#(MPC) simulations. Here are some important pieces of nomenclature for
#understanding these models:
    
#   R = Resistance
#   C = Capacitance
#   Q = Heat addition, representing gains
#   Subscript w = wall and refers to exterior walls
#   Subscript w# = window, with 1 referring to the interior pane of a window
#       and 2 referring the exterior pane of a window
#   Subscript s = slab
#   Subscript o = outdoor
#   Subscript i = indoor
#   Subscript p = Previous. E.g. "Ti_p" = Indoor temperature at the previous
#       timestep
#   Subscript ext = ???? (E.g. def R1C1(i, Ti_p, To, Qi_ext, param):)
#   i = Current time (Not just an index)

#Model and documentation last updated Jan 19, 2021 by Peter Grant



def R1C1(i, Ti_p, To, Qi_ext, param):

#Function for RC model considering indoor, outdoor conditions

#i = Time
#Ti_p = previous/initial indoor temperature
#To = Outdoor temperature
#Qi_ext = Indoor space heat gains
#param = Dictionary of parameters defining R & C values

    timestep = param['timestep'] #Read the timestep from param
    Roi = param['Roi'] #Resistance between the outside and the inside
    Ci = param['Ci'] / timestep #Ci = Capacitance inside

    if i == 0: #If calculating at time = 0
        return Ti_p #Return initial indoor temperature
    else: #If not the first timestep
        Ti = (Roi*Qi_ext + Roi*Ci*Ti_p + To) / (1 + Roi*Ci) #Calculate indoor
        #temperature at the next time step using R1C1 model
        return Ti #Return the calculated indoor temperature at this timestep

def R2C2(i, T1_prev, T2_prev, T_out, Q1_ext, Q2_ext, param):

#Function for RC model considering indoor, outdoor, slab conditions
#Internal gains applied to both indoor and slab

#T1 = indoor air temperature
#T2 = slab temperatue
#T_out = outdoor temperature
#Q1_ext = internal gains in space
#Q2_ext = internal gains in slab

    timestep = param['timestep'] #Read the timestep
    R1 = param['Roi'] #Read the resistane between indoor and outdoor
    C1 = param['Ci'] / timestep #Calculate the indoor air C/dt
    R2 = param['Ris'] #Read the resistance between indoor and the slab
    C2 = param['Cs'] / timestep #Calculate the slab C/dt
    
    if i == 0: #If time = 0
        return T1_prev, T2_prev #Return initial temperatures of indoor, slab
    else: #If not first time step perform calculations
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
    
#Function for RC model considering indoor, outdoor, slab, and a window
#Internal gains applied to indoor, slab, 2x window panes

#i = Time
#Ti_p = Previous/initial indoor temperature
#Ts_p = Previous/initial slab temperature
#To = Outdoor temperature
#Qw1_ext = Heat gain in the external window pane
#Qw2_ext = Heat gain in the internal window pane
#Qi_ext = Indoor space heat gains
#Qs_ext = Slab heat gains
#param = Dictionary of parameters defining R & C values
    
    timestep = param['timestep'] #Read the timestep
    Row1 = param['Row1'] #Read the resistance between outdoor and window 1
    Rw1w2 = param['Rw1w2'] #Read the resistance between windows 1 and 2
    Rw2i = param['Rw2i'] #Read the resistance between window 2 and indoor
    Ci = param['Ci'] / timestep #Calculate the indoor C/dt
    Ris = param['Ris'] #Read the resistance between indoor and slab
    Cs = param['Cs'] / timestep #Calculate the slab C/dt
    
    if i == 0: #If time = 0
        return Ti_p, Ts_p #Return initial indoor and slab temperatures
    else: #If not first timestep, perform calculations
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
    
#Function for RC model considering indoor, slab, 1 double-pane window, one wall
#Applying internal gains to both window panes, interior, slab, wall

#i = Time
#Ti_p = Previous/initial indoor temperature
#Ts_p = Previous/initial slab temperature
#Tw_p = Previous/initial wall temperature
#To = Outdoor temperature
#Qw1_ext = Heat gain in the external window pane
#Qw2_ext = Heat gain in the internal window pane
#Qi_ext = Indoor space heat gains
#Qs_ext = Slab heat gains
#Qw_ext = Wall heat gains
#param = Dictionary of parameters defining R & C values
    
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
    else: #If not first timestep perform calculations
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
    
#Function for RC model considering indoor, slab, 1 double-pane window, one wall
#Applying internal gains to both window panes, interior, slab, wall

#i = Time
#Ti_p = Previous/initial indoor temperature
#Ts_p = Previous/initial slab temperature
#Tw_p = Previous/initial wall temperature
#To = Outdoor temperature
#Qw1_ext = Heat gain in the external window pane
#Qw2_ext = Heat gain in the internal window pane
#Qi_ext = Indoor space heat gains
#Qs_ext = Slab heat gains
#Qw_ext = Wall heat gains
#param = Dictionary of parameters defining R & C values
    
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
    else: #If time != 0 perform calculations
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
        
if __name__ == '__main__':
    
#Note: This code was from an initial validation of the FMUs, but has not been
#maintained over time. For an up-to-date validation see Validation_FMUvRC.ipynb
    
    import copy
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Configuration
    hor = 60*60*12
    timestep = 5*60
    
    # Setup of models
    
    #Define temperatures and internal gains for R1C1 model test
    test_R1C1 = {'Ti_p': [22.0], 'Qi_ext': 5}
    #Add parameters for R1C1
    test_R1C1['param'] = {'timestep': timestep,
                          'Roi': 0.15, 'Ci': 0.75e5}       
    
    #Define temperatures for R2C2 model test
    test_R2C2 = {'Ti_p': [22.0], 'Ts_p': [22.0], 'Tw_p': [22.0]}
    #Add internal gain values
    test_R2C2.update({'Qi_ext': 5, 'Qs_ext': 1})
    #Creates 'param' definition for R2C2 with timestep, C, R values
    test_R2C2['param'] = {'timestep': timestep,
                          'Roi': 0.15, 'Ci': 0.75e5, 'Ris': 0.05, 'Cs': 1e6}
    
    #Copies all definitions from R2C2 to R2C24C2
    test_R4C2 = copy.deepcopy(test_R2C2)
    #Adds additional internal gain parameters to R4C2
    test_R4C2.update({'Qw1_ext': 100, 'Qw2_ext': 10,
                      'Qi_ext': 10, 'Qs_ext': 10})
    #Creates 'param' definition for R4C2with timestep, C, R values
    test_R4C2['param'] = {'timestep': timestep,
                          'Row1': 1.0, 'Rw1w2': 0.02, 'Rw2i': 1.0, 'Ci': 1e5,
                          'Ris': 0.05, 'Cs': 1e6}
    
    #Copies all R2C2 definitions to form the basis for R5C3                      
    test_R5C3 = copy.deepcopy(test_R2C2)
    #Adds internal gain definitions to R5C3
    test_R5C3.update({'Qw1_ext': 100, 'Qw2_ext': 10,
                      'Qi_ext': 10, 'Qs_ext': 10, 'Qw_ext': 15})
    #Creates 'param' definition for R5C3 with timestep, C, R value
    test_R5C3['param'] = {'timestep': timestep,
                          'Row1': 1.0, 'Rw1w2': 0.02, 'Rw2i': 1.0, 'Ci': 1e5,
                          'Ris': 0.05, 'Cs': 1e6, 'Riw': 0.05, 'Cw': 1e5}
                          
    #Copies all R2C2 definitions to form basis of R6C3
    test_R6C3 = copy.deepcopy(test_R2C2)
    #Adds internal gain definitions for R6C3
    test_R6C3.update({'Qw1_ext': 100, 'Qw2_ext': 10,
                      'Qi_ext': 10, 'Qs_ext': 10, 'Qw_ext': 15})
    #Creates 'param' definition for R6C3 including timestep, C, R values
    test_R6C3['param'] = {'timestep': timestep, 'Roi': 2.0, 
                          'Row1': 2.0, 'Rw1w2': 0.02, 'Rw2i': 2.0, 'Ci': 1e5,
                          'Ris': 0.05, 'Cs': 1e6, 'Riw': 0.05, 'Cw': 1e5}
    
    ix = list(range(0, hor+timestep, timestep)) #Creates a list of indices
    #T_out = np.sin([i*timestep/4 for i in ix]) + 22.0
    T_out = [22] * len(ix) #Sets T_out = 22 for all timesteps
    for i in ix: #Iterates through all indices
        oat = T_out[ix.index(i)] #Sets outside air temperature = T_out
        
        #R1C1 Model
        tip = R1C1(i, test_R1C1['Ti_p'][-1], oat, test_R1C1['Qi_ext'], test_R1C1['param'])
        test_R1C1['Ti_p'].append(tip)        
        
        # R2C2 Model
        tip, tsp = R2C2(i, test_R2C2['Ti_p'][-1], test_R2C2['Ts_p'][-1], oat, 
                        test_R2C2['Qi_ext'], test_R2C2['Qs_ext'],
                        test_R2C2['param']) #Calculate indoor, slab temps
        test_R2C2['Ti_p'].append(tip) #Append indoor temp to list
        test_R2C2['Ts_p'].append(tsp) #Append slab temp to list
        
        # R4C2 Model
        tip, tsp = R4C2(i, test_R4C2['Ti_p'][-1], test_R4C2['Ts_p'][-1], oat, 
                        test_R4C2['Qw1_ext'], test_R4C2['Qw2_ext'],
                        test_R4C2['Qi_ext'], test_R4C2['Qs_ext'],
                        test_R4C2['param']) #Calculate indoor, slab temps
        test_R4C2['Ti_p'].append(tip) #Append indoor temp to list
        test_R4C2['Ts_p'].append(tsp) #Append slab temp to list

        # R5C3 Model
        #Calculate indoor, slab, wall temperatures
        tip, tsp, twp = R5C3(i, test_R5C3['Ti_p'][-1], test_R5C3['Ts_p'][-1],
                             test_R5C3['Tw_p'][-1], oat, 
                             test_R5C3['Qw1_ext'], test_R5C3['Qw2_ext'],
                             test_R5C3['Qi_ext'], test_R5C3['Qs_ext'],
                             test_R5C3['Qw_ext'], test_R5C3['param'])
        test_R5C3['Ti_p'].append(tip) #Append indoor temp to list
        test_R5C3['Ts_p'].append(tsp) #Append slab temp to list
        test_R5C3['Tw_p'].append(twp) #Append wall temp to list
        
        # R6C3 Model
        #Calculate indoor, slab, wall temperatures
        tip, tsp, twp = R6C3(i, test_R6C3['Ti_p'][-1], test_R6C3['Ts_p'][-1],
                             test_R6C3['Tw_p'][-1], oat, 
                             test_R6C3['Qw1_ext'], test_R6C3['Qw2_ext'],
                             test_R6C3['Qi_ext'], test_R6C3['Qs_ext'],
                             test_R6C3['Qw_ext'], test_R6C3['param'])
        test_R6C3['Ti_p'].append(tip) #Append indoor temp to list
        test_R6C3['Ts_p'].append(tsp) #Append slab temp to list
        test_R6C3['Tw_p'].append(twp) #Append wall temp to list
             
    # Evaluate R2C2
    # TDB

    # Evaluate R4C2 (using Dymola and R4C2.mo model)
    ix_dymola = list(range(0, hor+60*60, 60*60))
    Ti_dymola = [22,23.684183,24.57737,25.08907,25.43141,25.706179,25.96421,
                 26.203693,26.442825,26.676924,26.91098,27.142605,27.37423]
    Ts_dymola = [22,22.098358,22.273655,22.485842,22.714176,22.948702,
                 23.184359,23.420916,23.657452,23.893692,24.129923,24.36558,
                 24.601234]
    plt.plot(ix_dymola, Ti_dymola)
    plt.plot(ix_dymola, Ts_dymola)
    plt.plot(ix, test_R4C2['Ti_p'][1:])
    plt.plot(ix, test_R4C2['Ts_p'][1:])
    plt.title('R4C2')
    plt.legend(['Ti', 'Ts', 'Ti_rc', 'Ts_rc'])
    plt.show()
    
    # Evaluate R5C3 (using Dymola and R5C3.mo model)
    ix_dymola = list(range(0, hor+60*60, 60*60))
    Ti_dymola = [22,23.684183,24.57737,25.08907,25.43141,25.706179,25.96421,
                 26.203693,26.442825,26.676924,26.91098,27.142605,27.37423]
    Ts_dymola = [22,22.098358,22.273655,22.485842,22.714176,22.948702,
                 23.184359,23.420916,23.657452,23.893692,24.129923,24.36558,
                 24.601234]
    Tw_dymola = [22,22.098358,22.273655,22.485842,22.714176,22.948702,
                 23.184359,23.420916,23.657452,23.893692,24.129923,24.36558,
                 24.601234]
    plt.plot(ix_dymola, Ti_dymola)
    plt.plot(ix_dymola, Ts_dymola)
    plt.plot(ix_dymola, Tw_dymola)
    plt.plot(ix, test_R5C3['Ti_p'][1:])
    plt.plot(ix, test_R5C3['Ts_p'][1:])
    plt.plot(ix, test_R5C3['Tw_p'][1:])
    plt.title('R5C3')
    plt.legend(['Ti', 'Ts', 'Tw', 'Ti_rc', 'Ts_rc', 'Tw_rc'])
    plt.show()
    
    # Evaluate R6C3 (using Dymola and R6C3.mo model)
    ix_dymola = list(range(0, hor+60*60, 60*60))
    Ti_dymola = [22,23.684183,24.57737,25.08907,25.43141,25.706179,25.96421,
                 26.203693,26.442825,26.676924,26.91098,27.142605,27.37423]
    Ts_dymola = [22,22.098358,22.273655,22.485842,22.714176,22.948702,
                 23.184359,23.420916,23.657452,23.893692,24.129923,24.36558,
                 24.601234]
    Tw_dymola = [22,22.098358,22.273655,22.485842,22.714176,22.948702,
                 23.184359,23.420916,23.657452,23.893692,24.129923,24.36558,
                 24.601234]
    plt.plot(ix_dymola, Ti_dymola)
    plt.plot(ix_dymola, Ts_dymola)
    plt.plot(ix_dymola, Tw_dymola)
    plt.plot(ix, test_R6C3['Ti_p'][1:])
    plt.plot(ix, test_R6C3['Ts_p'][1:])
    plt.plot(ix, test_R6C3['Tw_p'][1:])
    plt.title('R6C3')
    plt.legend(['Ti', 'Ts', 'Tw', 'Ti_rc', 'Ts_rc', 'Tw_rc'])
    plt.show()
    
    
    