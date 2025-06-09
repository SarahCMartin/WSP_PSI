import Common
import numpy as np

def latbrk(Lat_brk_model, Lat_brk_suction, D, W, alpha, int_vert_eff_max, int_vert_eff, insitu_calc_depths, insitu_su_inc, lat_su_inc, hydro_calc_depths, su_consol_preop, gamma_sub, int_SHANSEP_S, int_SHANSEP_m, ka, kp, delta, z, B):
    """This function calculates the lateral breakout friction factors using 
    the chosen method with embedment and strength from previous calculation 
    stages."""

    ###########################################################################
    # DNVGL-RP-F114 Undrained Model 1, Section 4.4.2.2 with adjustment to use interface strength rather than alpha factor, and OCR rather than gamma_preloading as these only correspond if the soil was fully consolidated under previous loading.
    if Lat_brk_model == 0:
        # Note DNV appears to have erroneously labelled ff_L,brk,u,fric as F_L,brk,u,fric based on the structure of equations and units. This could be resolved by multiplying by the vertical force, however in this case it is resolved as part of the improvement to allow suOC to reflect partial consolidation
        # Note DNV equation for F_L,brk,u,remain uses full z and su_z/2 which is unrealistic for z > D, so here z is replaced with min(z,D) and su at z - min(z,D)/2

        # Defining inputs
        su_active = np.interp(z - min(z,D)/2, insitu_calc_depths, lat_su_inc) # average undrained shear strength in active failure zone, DNV notes typically taken at z/2 (adjustment as noted above); profiles have been developed with a lateral sensitivity factor to reflect partial remoulding of the soil either side of the pipe
        su_passive = su_active # average undrained shear strength in passive failure, taken as the same as active in this case, notes as per active
        gamma_rate = 1
        if int_vert_eff_max == int_vert_eff:
            OCR = 1 # setting as 1 instead of calculating to avoid NaN where both values are 0
        else:
            OCR = int_vert_eff_max/int_vert_eff # previous maximum vertical effective stress over current (for immediate undrained response this is pre-operational vertical effective stress as consolidation will not have occurred yet under the slightly higher operational weight)

        # Calculating breakout resistance from components
        # Equation separated into steps from suOC to F to ff as SAFEBUCK/DNV formula doesn't account for W_op not being fully transferred to the soilimmediately, and therefore over-estimates strength as W_op > W_empty ?> W_tranferred to soil during the previous consolidation stages
        int_su_lat_brk = int_SHANSEP_S*(OCR**int_SHANSEP_m)*int_vert_eff # strength using SHANSEP approach and vertical effective stress at the start of operation based omn the previous consolidation stages
        F_lat_brk_friction = int_su_lat_brk*gamma_rate*B # horizontal friction component of resistance per m length
        if Lat_brk_suction == 0: # not allowing for suction at the rear of the pipe
            F_lat_brk_remain = min(z,D)*(kp*su_passive + 0.5*gamma_sub*z)*gamma_rate # first z replaced with min(z,D) as noted above, second z remains as it relates to the weight of soil above the base of the pipe
        else:
            F_lat_brk_remain = min(z,D)*(ka*su_active + kp*su_passive)*gamma_rate # z replaced with min(z,D) as noted above

        F_lat_brk = F_lat_brk_friction + F_lat_brk_remain
        ff_lat_brk = F_lat_brk/W # friction factor is ratio of resistance to vertical force 
    
    ###########################################################################
    # Merifield et al 2008 approach: generating empirical H-V envelopes, Vmax and Hmax formulation from Merifield et al 2009
    elif Lat_brk_model == 2:
        # Note same approach applied to take average strength as z - min(z,D)/2 rather than z/2 to better reflect actual su which would be relevant where z > D; note Merifield et al 2009 assume a uniform su so they do not specify an averaging method
        su_horz = np.interp(z - min(z,D)/2, insitu_calc_depths, lat_su_inc) # for Hmax calc
        su_vert = np.interp(z, hydro_calc_depths, su_consol_preop) # for Vmax calc taking su at pipe invert, no formulation specified as Merifield assume uniform su
        
        Ncmax_smooth = 9.14 # theoretical max from Randolph and Houlsby (1984)
        Ncmax_rough = 11.94 # as above

        # Finding maximum vertical load made up of components for bearing capacity (D*su*NcV) and soil self-weight (D*gamma)sub*z*NswV)
        if z/D <= 0.5:
            NcV_smooth = 5.66*(z/D)**0.32
            NcV_rough = 7.4*(z/D)**0.4
        else: # adjusting fit above z/D=0.5 based on FE extrapolation of Merifield et al (in excel file 'Extrapolating Merifield v2')
            NcV_smooth = 5.1*(z/D)**0.14
            NcV_rough = 5.8*(z/D)**0.05

        NcV_smooth = min(NcV_smooth, Ncmax_smooth)
        NcV_rough = min(NcV_rough, Ncmax_rough)

        if alpha == 0: # smooth pipe
            NcV = NcV_smooth
        elif alpha == 1: # rough pipe
            NcV = NcV_rough
        else: # pipe between smooth and rough, linearly interpolating
            NcV = NcV_smooth*(1-alpha) + NcV_rough*alpha

        if z/D <= 0.5: # results only realistic up to this limit which was tested by Merifield; alternative developed above this by extrapolating the plots
            NswV = (D/(4*z))*(np.arcsin(np.sqrt(4*(z/D)*(1-z/D))) - 2*(1-2*(z/D))*np.sqrt((z/D)*(1-(z/D))))
        elif z <= D:
            NswV = -0.3074*(z/D)**2 + 0.6448*(z/D) + 0.5342 # reference excel spreadsheet 'Extrapolating Merifield 2009' for fitting data; considered non-conservative to cap at z/D=0.5 as the FE models available to validate againts used weightless soil (and high estimate is conservative for lateral breakout)
        else:
            NswV = -0.3074*(1)**2 + 0.6448*(1) + 0.5342 # capping at value for z=D as this is where the best fit levels off before it decreases unrealistically and without data

        Vmax = D*su_vert*NcV + D*gamma_sub*z*NswV

        # Finding maximum horizontal load made up of components for bearing capacity (D*su*NcH) and soil self-weight (D*gamma_sub*z*NswH)
        if z <= D:
            NcH_smooth = 2.72*(z/D)**0.78
            NcH_rough = 3.26*(z/D)**0.82
        else:  # adjusting fit above z/D=0.5 based on FE extrapolation of Merifield et al (in excel file 'Extrapolating Merifield v2')
            NcH_smooth = 2.04*(z/D)**0.26
            NcH_rough = 2.32*(z/D)**0.27

        NcH_smooth = min(NcH_smooth, Ncmax_smooth)
        NcH_rough = min(NcH_rough, Ncmax_rough)

        if alpha == 0: # smooth pipe
            NcH = NcH_smooth
        elif alpha == 1: # rough pipe
            NcH = NcH_rough
        else: # pipe between smooth and rough, linearly interpolating
            NcH = NcH_smooth*(1-alpha) + NcH_rough*alpha
            
        NswH = z/(2*D) # considered non-conservative to cap at z/D = 0.5 as the FE models available to validate against use weightless soil (and high estimate is conservative for lateral brekout)

        Hmax = D*su_horz*NcH + D*gamma_sub*z*NswH

        # Finding lateral breakout resistance from H-V envelope using skew factors
        if z/D < 0.6: # mathematically valid results up to z/D=1.2 but FE extrapolation found best fit for large z/D found to be with skew factors from z/D = 0.6
            beta1 = (0.8 - 0.15*alpha)*(1.2 - z/D)
            beta2 = 0.35*(2.5 - z/D)
        else:
            beta1 = (0.8 - 0.15*alpha)*(1.2 - 0.6)
            beta2 = 0.35*(2.5 - 0.6)

        beta = (beta1+beta2)**(beta1+beta2)/(beta1**beta1*beta2**beta2)

        F_lat_brk = Hmax*(beta*((W/Vmax)**beta1)*((1-W/Vmax)**beta2))
        ff_lat_brk = F_lat_brk/W # friction factor is ratio of resistance to vertical force

    ###########################################################################
    # DNVGL-RP-F114 Drained Model 1, Section 4.4.2.3, adjusted to use tan(delta) directly, i.e. interface friction, rather than r*tan(phi)
    elif Lat_brk_model == 10:
        Kp = (1 + np.sin(np.deg2rad(delta)))/(1 - np.sin(np.deg2rad(delta)))
        F_lat_brk_passive = 0.5*Kp*gamma_sub*z**2
        F_lat_brk_friction = np.tan(np.deg2rad(delta))*(max(0, (W - np.tan(np.deg2rad(delta))*F_lat_brk_passive)))

        # Breakout resistance from components
        F_lat_brk = F_lat_brk_passive + F_lat_brk_friction
        ff_lat_brk = F_lat_brk/W # friction factor is ratio of resistance to vertical force

    ###########################################################################
    # No valid lateral breakout model selected
    else:
        ff_lat_brk = []
        print("Please select a valid model for lateral brekout resistance calculation.")

    ###########################################################################
    # Lateral breakout mobilisation displacements from DNVGL-RP-F114, Table 4-4
    yLE_lat_brk = (0.004 + 0.02*z/D)*D
    yBE_lat_brk = (0.02 + 0.25*z/D)*D
    yHE_lat_brk = (0.1 + 0.7*z/D)*D
    y_lat_brk = [yLE_lat_brk, yBE_lat_brk, yHE_lat_brk]

    return [ff_lat_brk, y_lat_brk]


def latres(Lat_res_model, Lat_res_suction, D, W, alpha, int_vert_eff_max, int_vert_eff, calc_depths, insitu_su_inc, gamma_sub, int_SHANSEP_S, int_SHANSEP_m, ka, kp, delta, z_hydro, z_res, B_res):
    """This function calculates the lateral residual friction factors 
    (i.e. after breakout) using the chosen method with embedment and 
    strength from previous calculation stages."""

    ###########################################################################
    # DNVGL-RP-F114 Undrained Lateral Breakout Model 1, Section 4.4.2.2 with embedment reduced to post-breakout depth; this is the same approach used by DNV for drained residual relative to drained lateral breakout and no non-empirical UD formulation is presented. Adjustments to lat brk formulation as above.
    if Lat_res_model == 0:
        # See notes for DNV UD lat brk Model 1 above in latbrk function, same calculation so passing to that function with relevant inputs for the residual scenario, i.e. residual z and B, both strength increase profiles are the insitu as material either side of pipe has not been disturbed during laying as assumed for latbrk
        [ff_lat_res, _] = latbrk(0, Lat_res_suction, D, W, [], int_vert_eff_max, int_vert_eff, calc_depths, [], insitu_su_inc, [], [], gamma_sub, int_SHANSEP_S, int_SHANSEP_m, ka, kp, [], z_res, B_res)

    ###########################################################################
    # DNVGL-RP-F114 Drained Lateral Breakout Model 1, Section 4.4.2.3 with embedment reduced to post-breakout depth. Adjustments to lat brk formulation as above.
    elif Lat_res_model == 10:
        # See notes for DNV D lat brk Model 1 above in latbrk function, same calculation so passing to that function with relevant inputs for the residual scenario, i.e. residual z and B, both strength increase profiles are the insitu as material either side of pipe has not been disturbed during laying as assumed for latbrk
        [ff_lat_res, _] = latbrk(10, [], D, W, [], [], [], [], [], [], [], [], gamma_sub, [], [], [], [], delta, z_res, [])
        
    ###########################################################################
    # Lateral residual mobilisation displacements from DNVGL-RP-F114, Table 4-5
    yLE_lat_res = 0.6*D
    yBE_lat_res = 1.5*D
    yHE_lat_res = 2.8*D
    y_lat_res = [yLE_lat_res, yBE_lat_res, yHE_lat_res]

    return [ff_lat_res, y_lat_res]

def axial(Ax_model, D, W, alpha, int_vert_eff_max, int_vert_eff, int_SHANSEP_S, int_SHANSEP_m, delta, z, B):
    """This function calculates the axial friction factors using the 
    chosen method with embedment and strength from previous 
    calculation stages."""
    
    ###########################################################################
    # SAFEBUCK / DNVGL-RP-F114 Undrained Model, directly using interface properties per SAFEBUCK but rate effects per DNV, adjustment to interface strength as descriped in lateral breakout section to allow for partial consolidation
    if Ax_model == 0:

        # Defining inputs
        gamma_rate = 1
        if int_vert_eff_max == int_vert_eff:
            OCR = 1 # setting as 1 instead of calculating to avoid NaN where both values are 0
        else:
            OCR = int_vert_eff_max/int_vert_eff # previous maximum vertical effective stress over current (for immediate undrained response this is pre-operational vertical effective stress as consolidation will not have occurred yet under the slightly higher operational weight)

        # Finding wedging factor (beta formulation from SAFEBUCK Section C.4.3 as this includes cutoff which DNV doesn't list but is required for sensible results)
        if z/D > 1: # beta constant at 180deg if pipe fully below mudline
            beta = np.pi
        else: # z < D/2, embedded to less than pipe centreline
            beta = np.acos(1-(2*z/D))
            
        if beta > np.pi/2: # limit for wedging factor formula in SAFEBUCK, corresponds to z > D/2, DNV notes wedging factor constant above this embedment, so beta set to 90deg (value at z = D/2)
            beta = np.pi/2
    
        wedge = 2*np.sin(beta)/(beta+np.sin(beta)*np.cos(beta))

        # Calculating axial resistance
        # Equation separated into steps from suOC to F to ff as SAFEBUCK/DNV formula doesn't account for W_op not being fully transferred to the soilimmediately, and therefore over-estimates strength as W_op > W_empty ?> W_tranferred to soil during the previous consolidation stages
        int_su_ax = int_SHANSEP_S*(OCR**int_SHANSEP_m)*int_vert_eff # strength using SHANSEP approach and vertical effective stress at the start of operation based on the previous consolidation stages
        F_ax = wedge*int_su_ax*gamma_rate*B # resistance per m length (assuming it only acts over the horizontal distance B and not the full contact length; the difference being partially accounted for in the wedging factor)
        ff_ax = F_ax/W # friction factor is ratio of resistance to vertical force (W_op)