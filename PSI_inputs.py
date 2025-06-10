# PSI_inputs

# This script runs Pipe-Soil Interaction Calculations for a single set of
# input parameters defined below
# Outputs will be saved in specified directory

###########################################################################
# Housekeeping
import os
directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(directory) # finding folder where this script is saved and setting as the working directory - if all functions are saved in the same folder this will prevent any directory errors
d = {} # empty dictionary to take input parameters to add to PSI_case object in the initiation step

###########################################################################
# Model Selection
d['Emb_aslaid_model'] = 0   # undrained: DNVGL-RP-F114 Model 1 = 0, DNVGL-RP-F114 Model 2/SAFEBUCK = 1
                            # drained: DNVGL-RP-F114 = 10
d['Emb_hydro_model'] = 0    # as above
d['Lat_brk_model'] = 0      # undrained: DNVGL-RP-F114 Model 1 = 0, DNVGL-RP-F114 Model 2/SAFEBUCK = 1, Merifield et al 2008/2009 = 2
                            # drained: DNVGL-RP-F114 Model 1 = 10, DNVGL-RP-F114 Model 2/SAFEBUCK = 11
d['Lat_brk_suction'] = 0    # switch for suction at rear of pipe, not allowing = 0, allowing = 1 (relevant for lateral breakout undrained DNVGL-RP-F114 Model 1)
d['Lat_res_model'] = 0      # undrained: modified DNV = 0 (using instantaneous embedment into undisturbed soil profile then same approach as DNVGL-RP-F114 Lat Brk UD Model 1), DNVGL-RP-F114/SAFEBUCK = 1 (note the same approach cannot be used with the Merifield brkout model as the residual embedment corresponds to the vertical bearing capacity such that the envelope gives 0 horizontal capacity)
                            # drained: DNVGL-RP-F114 Model 1 = 10, DNVGL-RP-F114 Model 2/SAFEBUCK = 11
d['Lat_res_suction'] = 0    # switch for suction at rear of pipe, not allowing = 0, allowing = 1 (relevant for lateral residual undrained modified method)
d['Emb_res_model'] = 0      # as for above embedment models (relevant for lateral residual undrained modified method and drained DNVGL-RP-F114 Model 1)
d['Ax_model'] = 0           # undrained: SAFEBUCK / DNVGL-RP-F114 = 0 (SAFEBUCK version directly uses interface properties whereas DNV uses alpha interface reduction factor; DNV includes rate effects; apart from these details the underlying model is the same)
                            # drained: DNVGL-RP-F114/SAFEBUCK = 10
d['Lat_cyc_model'] = 0      # undrained: SAFEBUCK = 0 (not included in DNVGL-RP-F114)
d['No_cycles'] = 10         # number of cycles to model for lateral cyclic resistance calculations

###########################################################################
# Pipe Inputs
d['D'] = 0.3299             # pipe diameter in (m)
d['W_empty'] = 0.8975       # submerged weight of empty pipe per unit length (kN/m)
d['W_hydro'] = 1.5022       # submerged weight of flooded pipe during hydrotest per unit length (kN/m)
d['W_op'] = 1.0745          # submerged weight of operating pipe per unit length (kN/m) (CAMAGO: max 1.0745, mid 1.0155, min 0.9565)
d['alpha'] = 0.5            # pipe-soil interface adhesion coefficient, fully smooth = 0 and fully rough = 1 (CAMAGO: estimated such that axial residual LE corresponds ok with interface shear data)
d['EI'] = 61675             # pipe bending stiffness (kNm2) from E = 210GPa for steel and I for pipe annulus dimensions
d['T0'] = 44                # bottom lay tension (kN) (CAMAGO: Crondall use 27 to 236 kN; possible update to 44 to 257 kN from SEA)
d['z_ini'] = d['D']         # initial guess of pipe embedment as starting point for iteration (m)
d['t_aslaid'] = 1/12        # time between pipe lay and hydrotest (years)
d['t_hydro'] = 1/24         # time to complete hydrotest plus time pipe is left flooded between hydrotest and operation (years)
d['t_preop'] = 1/12         # time pipe is left empty between hydrotest and operation (years)

###########################################################################
# Soil Inputs
d['su_profile'] = 1         # soil-soil: only considering the top 1m, linear profile = 0, bi-linear profile = 1
d['su_mudline'] = 0.35         # soil-soil: undrained shear strength at mudline (kPa) (CAMAGO: LE 0, BE 0.35, HE 3.57, OC HE 64.29)
d['su_inv'] = 4.84          # soil-soil: undrained shear strength at inversion point for a bi-linear profile, z_su_inv (kPa) (CAMAGO: LE 1.33, BE 4.84, HE 14.29, OC HE 75)
d['z_su_inv'] = 0.5         # soil-soil: depth for inversion point for a bi-linear undrained shear strength profile (m) 
d['delta_su'] = 3.00        # soil-soil: increase in undrained shear strength with depth (kPa/m), below inversion for a bi-linear profile (CAMAGO: LE 1.33, BE 3.00, HE 7.72, OC HE 7.14)
d['gamma_sub'] = 4.20       # submerged unit weight of soil (kN/m3) (CAMAGO: avg over top 1m LE 3.04, BE 4.20, HE 6.96, OC HE 7.74)
d['pipelay_St'] = 7         # soil-soil: sensitivity factor for pipe lay, adjusted as part of calibration with other pipes in the area so may not correspond to standard soil sensitivity (CAMAGO: not changed from calibration to nearby pipelines at this stage)
d['lateral_St'] = 4.33      # soil-soil: sensitivity factor for lateral breakout relating to the soil in the active and passive failure zones on either side of the embedded pipe (CAMAGO: avg over top 1m LE 1.5, BE 4.33, HE 8.33)
d['cv'] = 23                 # soil-soil: coefficient of consolidation (m2/year) (CAMAGO: avg over top 1m LE 1, BE 23, HE 357, OC HE 1281)
d['SHANSEP_S'] = 0.285       # soil-soil: normalised shear strength for NC condition, found from best fit between CPTs and oedometer tests with SHANSEP approach (CAMAGO: LE 0.22, BE 0.285, HE 0.35)
d['SHANSEP_m'] = 0.825        # soil-soil: SHANSEP exponent, found from best fit between CPT and oedometer tests (CAMAGO: LE 0.8, BE 0.825, HE 0.85)
d['ka'] = 2.25              # soil-soil: pressure resistance coefficients (note not earth pressure coefficient Ka), 2 <= ka <= 2.5 from DNVGL-RP-F114 Section 4.4.2.2
d['kp'] = 2.25              # soil-soil: pressure resistance coefficients (note not earth pressure coefficient Kp), 2 <= kp <= 2.5 from DNVGL-RP-F114 Section 4.4.2.2
d['phi'] = 30               # soil-soil: friction angle (deg) (CAMAGO: using LE 24, BE 30, HE 42)

###########################################################################
# Interface Inputs
d['int_SHANSEP_S'] = 0.36   # soil-pipe interface: normalised shear strength for NC condition (CAMAGO: 0.36 for all)
d['int_SHANSEP_m'] = 0.4    # soil-pipe interface: SHANSEP exponent (CAMAGO: 0.4 for all)
d['delta'] = 34.6           # soil-pipe interface: friction angle (deg) (CAMAGO: using LE 26.6, BE 34.6, HE 41.3 from atan of values from interface friction tests)

###########################################################################
# Run PSI
# [z_aslaid, z_hydro, z_res, ff_lat_brk, y_lat_brk, ff_lat_res, y_lat_res, ff_ax, x_ax, ff_lat_cyc, ff_lat_berm, z_cyc] = 
from PSI_class import PSI
PSI_case = PSI(d)

import PSI_soils
PSI_case = PSI_case.PSI_master()
