import Common
import numpy as np

def strength_profile(Emb_aslaid_model, Emb_hydro_model, Lat_brk_model, Lat_res_model, Emb_res_model, Ax_model, PhaseNo, D, su_profile, su_mudline, su_inv, z_su_inv, delta_su, St, z, prev_calc_depths, prev_su_inc):
    """This function turns the input linear or bi-linear strength profile into
    incremental profile with corresponding depths and applies sensitivity to 
    reflect remoulding profiles as relevant."""

    ###########################################################################
    # Constants, Fixed Inputs or Input Adjustments
    max_depth = Common.round_up(max(2*D,1),1) # considering calculation for a maximum depth of either 1m or 2D (below pipe invert if relevant, adjusted for in next step), whichever is greater, rounding up to the nearest 0.1m for subsequent increment definition
    if PhaseNo == 1 or PhaseNo == 5: # profile to be used for as-laid embedment calculation and soil either side of pipe for lateral resistance, so needs to start from mudline
        calc_depths = Common.float_range(0,max_depth,0.02) # making 0.02 m increments
    else: # profile to be used for consolidation calculations at different stages
        calc_depths = Common.float_range(z,(max_depth+z),0.02)

    ###########################################################################
    # Setting strength profile output as empty if all phases are modelled as drained
    if Emb_aslaid_model >= 10 and Emb_hydro_model >= 10 and Lat_brk_model >= 10 and Lat_res_model >= 10 and Emb_res_model >= 10 and Ax_model >= 10: # soil always modelled as drained
        su_inc = []

    ###########################################################################
    # Calculating in-situ undrained strength profile if any phases are modelled as undrained
    else: # soil modelled as undrained in at least 1 calculation stage so profiles need to be updated for consolidation throughout
        if PhaseNo == 1 or PhaseNo == 2 or PhaseNo == 5: # profile to be used for as-laid embedment, consolidation calculation between pipe-lay and hydrotest, and material either side of pipe for lateral resistance
            su_ini = [0]*len(calc_depths)
            for i in range(len(calc_depths)):
                if su_profile == 1: # bi-linear su profile
                    if calc_depths[i] < z_su_inv: # interpolate strength for zone between mudline and strength profile inversion point
                        su_ini[i] = ((su_inv - su_mudline)/z_su_inv)*calc_depths[i] + su_mudline
                    else: # use delta_su for zone below strength profile inversion point
                        su_ini[i] = su_inv + delta_su*(calc_depths[i]-z_su_inv)
                else: # linear su profile, can use delta_su throughout
                    su_ini[i] = su_mudline + delta_su*calc_depths[i]
        elif PhaseNo == 4 or PhaseNo == 6: # profile to be used for consolidation calculation during hydrotest and between hydrotest and operation
            su_ini = Common.linear_extrapolate(calc_depths, prev_calc_depths, prev_su_inc)

    ###########################################################################
    # Applying sensitivity to reflect remoulding over varying depths depending on the calculation phase
    if PhaseNo == 1 or PhaseNo == 5: # profile to be used for as-laid embedment calculation and soil either side of pipe for lateral resistance with their associates St factors
        su_inc = [x/St for x in su_ini]
    elif PhaseNo == 2: # profile to be used for consolidation calculation
        su_inc = su_ini
        for i in range(len(calc_depths)):
            if calc_depths[i] <= (z+0.1*D): # fully remoulded in zone 0.1D below pipe invert
                su_inc[i] = su_ini[i]/St
            elif calc_depths[i] > (z+0.1*D) and calc_depths[i] <= (z+0.2*D): # transition from fully rmoulded to in-situ strength from 0.1D to 0.2D below pipe invert
                partial_St = St - (St-1)*(calc_depths[i]-(z+0.1*D))/(0.1*D)
                su_inc[i] = su_ini[i]/partial_St
    elif PhaseNo == 4 or PhaseNo == 6: # profile to be used for consolidation calculation during hydrotest and between hydrotest and operation
        su_inc = su_ini # remoulded zone and transition unchanged from previous profile so St not applied here

    return [calc_depths, su_inc]


def consolidation(PhaseNo, pressure_pipe, t, calc_depths, su_inc, gamma_sub, cv, SHANSEP_S, SHANSEP_m, D, z, B, prev_yield_stress, prev_vert_eff, int_switch):
    """This function updates the undrained strength profile to account for 
    consolidation where time passes between installation, hydrotesting and 
    operation based on input variables and fixed correlations (i.e. stress 
    distribution with depth). It is not used if initial as-laid embedment 
    calculation was drained."""
    
    ###########################################################################
    # Constants, Fixed Inputs or Input Adjustments
    # Stress distribution below pipe invert, taken from strip footing stress bulb diagrams
    vert_percentage = [0.06, 0.08, 0.1, 0.15] + Common.float_range(0.2, 1, 0.1)
    vert_percentage.reverse()
    z_B_ratio = [0.0, 0.28, 0.48, 0.65, 0.85, 1.2, 1.5, 2.1, 3.0, 4.2, 6.35, 7.9, 10.75]
    depth_dist = [x*B + z for x in z_B_ratio] # multiplying by contact width to change depth ratios to actual depths, adding initial embedment so the maximum occurs at invert depth, though the need for this simplification is a shortcoming of using the strip footing formulation

    # Degree of PWP dissipation constants (to fit Krost et al 2011 for strip footing at 0.2D (shallowest with good match and suitable low value for our data) to 0.5D embedment (deepest available), see excel 'Consolidation Equations')
    T50_z_D_0_2 = 0.2
    n_z_D_0_2 = 0.6
    T50_z_D_0_5 = 0.12
    n_z_D_0_5 = 0.48
    # Linearly varying between 0.2 and 0.5D embedment or applying these values as constant either side
    if z/D >= 0.5:
        T50 = T50_z_D_0_5
        n = n_z_D_0_5
    elif z/D > 0.2 and z/D < 0.5:
        T50 = np.interp(z/D, [0.2, 0.5], [T50_z_D_0_2, T50_z_D_0_5])
        n = np.interp(z/D, [0.2, 0.5], [n_z_D_0_2, n_z_D_0_5])
    else: # z/D <= 0.2
        T50 = T50_z_D_0_2
        n = n_z_D_0_2

    ###########################################################################
    # Calculating change in total stress at each depth increment
    vert_percentage_inc = Common.linear_extrapolate(calc_depths, depth_dist, vert_percentage)
    vert_percentage_inc[vert_percentage_inc<0] = 0
    vert_new = pressure_pipe*vert_percentage_inc # stress from full additional weight applied at pipe invert, decreasing with depth based on stress distribution previously defined

    ###########################################################################
    # Calculating degree of consolidation at each depth increment
    T_inc = cv*(t/(D**2))
    deg_consol_inc = 1 - np.exp(-1*np.log(2)*(T_inc/T50)**n)
    if deg_consol_inc > 1: # replaces any erroneous result above 1
        deg_consol_inc = 1
    elif deg_consol_inc < 0:
        deg_consol_inc = 0 # replaces -Inf results for time = 0

    ###########################################################################
    # Calculating excess pore water pressure and effective stress at each depth increment after pipe loading for time t
    if int_switch == 0: # soil-soil condition
        vert_eff_soil = [gamma_sub*x for x in calc_depths] # initial effective stress calculated from effective unit weight times depth for each increment
    else: # int_swicth == 1, pipe-soil interface condition
        vert_eff_soil = [gamma_sub*(x-z) for x in calc_depths] # assumed no soil weight above pipe invert impacting interface strength

    if isinstance(prev_vert_eff, np.ndarray): # consolidation stages after the first have initial stress coming from end of previous stage
        pwp_ini = vert_new - (prev_vert_eff-np.array(vert_eff_soil))
        pwp_ini = [max(x, 0) for x in pwp_ini]
        vert_eff_ini = [x + pressure_pipe - y for x,y in zip(vert_eff_soil, pwp_ini)]
    else: # first consolidation stage so no previous loading to consider
        pwp_ini = vert_new
        vert_eff_ini = vert_eff_soil

    pwp_fin = (1-deg_consol_inc)*np.array(pwp_ini)
    vert_eff_fin = vert_eff_soil + vert_new - pwp_fin

    ###########################################################################
    # Calculating consolidated undrained shear strength profile after pipe loading for time t
    
    if len(prev_yield_stress) > 0: # consolidation stages after the first have initial yield stress coming from end of previous stage
        yield_stress_ini = prev_yield_stress

    else: # first consolidation stage so no previous loading to consider
        if int_switch == 0: # soil-soil behaviour
            ratio = [x/y for x,y in zip(su_inc,vert_eff_ini)] # consider numpy arrays to improve calculation efficiency
            YSR_ini = [(x/SHANSEP_S)**(1/SHANSEP_m) for x in ratio]
            yield_stress_ini = [x*y for x,y in zip(YSR_ini,vert_eff_ini)]

        else: # int_switch == 1, pipe-soil interface behaviour
            YSR_ini = 1 # interface assumed to loose any previous consolidation through the remoulding, therefore behviour here is normally consolidated, note this unlinks the behaviour from the initial interface strength profile
            yield_stress_ini = [YSR_ini*x for x in vert_eff_ini]

    yield_stress = [max(x,y) for x,y in zip(yield_stress_ini,vert_eff_fin)]
    YSR_fin = yield_stress/vert_eff_fin
    YSR_fin[np.isnan(YSR_fin)] = 1 # replaces NaN results coming from a vertical effective stress of zero where time is zero with 1 to accurately get 0 increasing in interface strength at pipe invert
    su_consol = (SHANSEP_S*YSR_fin**SHANSEP_m)*vert_eff_fin

    return [su_consol, yield_stress, vert_eff_fin]