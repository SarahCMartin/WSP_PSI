import math
import numpy as np
from scipy.optimize import fsolve
import Common

def embedment(PhaseNo, Emb_model, D, W, alpha, EI, T0, z_ini, gamma_sub, calc_depths, su_inc, phi):
    """This function iterates to find the pipe embedment based on supplied
    input variables. It is applicable for as-laid or hydrotest embedment with
    variations to account for these differences switched using PhaseNo."""

    ###########################################################################
    # Calculate bearing capacity corresponding to initial guess for z (z_ini)
    # Contact width, B (m) and penetrated cross-sectional area, Abm (m2)
    (B, Abm) = emb_geometry(z_ini, D)
    if Emb_model < 10: # Undrained
        # Reference level for depth effects, z_su0 (m)
        z_su0 = depth_effects(z_ini, D, B)
        # Undrained shear strength at key depths - mudline (z = 0), reference level (z_su0), initial estimate of pipe embedment depth (z)
        su_mudline = Common.linear_extrapolate(0, calc_depths, su_inc) # allowing extrapolation for phases where strength profile starts from pipe invert, but the strength at mudline still relevant for later calculations
        su0 = Common.linear_extrapolate(z_su0, calc_depths, su_inc) # allowing extrapolation for phases where strength profile starts from pipe invert, but the reference depth will be above this point
        su_z = np.interp(z_ini, calc_depths, su_inc)

        # DNVGL-RP-F114 Undrained Model 1, Section 4.2.3.2
        if Emb_model == 0:
            # Bearing capacity incl. depth and bouyancy effects, Qv (kN/m)
            (_, _, Qv) = bearing_capacity_0(su_z, su_mudline, su0, z_ini, z_su0, alpha, B, D, gamma_sub, Abm) # passing back additional variables for later warning messages

        # DNVGL-RP-F114 Undrained Model 2, Section 4.2.3.3
        elif Emb_model == 1:
            # Bearing capacity incl. depth and bouyancy effects, Qv (kN/m)
            Qv = bearing_capacity_1(su_z, z_ini, D, gamma_sub, Abm)

    else: # Drained
        # DNVGL-RP-F114 Drained Model, Section 4.2.4
        # Bearing capacity, Qv (kN/m)
        Qv = bearing_capacity_10(z_ini, B, D, phi, gamma_sub)

    ###########################################################################
    # Calculate touchdown lay factor, klay (manually rearranged equation from DVNGL-RP-F114 Section 4.2.5.2 to be 4th order polynomial of klay, dependant on variable z)
    # Only relevant for as-laid embedment, not hydrotest embedment
    if PhaseNo == 1: # as-laid
        klay = lay_factor(T0, EI, W, z_ini)

    else: # PhaseNo == 3, hydrotest or PhaseNo == 6, post-breakout
        klay = 1 # setting to 1 where not relevant rather than removing from later formulas

    ###########################################################################
    # Iterate for z_aslaid, the embedment depth where equilibrium occurs between empty pipe submerged weight and soil bearing capacity)
    z_track = [z_ini]
    Qv_track = [Qv]
    W_track = [W*klay]

    if abs(Qv - W_track[-1]) < 0.01: # prevents error if initial guess for z gives result within target tolerance
        z = z_ini

    while abs(Qv - W_track[-1]) > 0.001:
        ###########################################################################
        # Select value of embedment, z (m), for next iteration depending on the relative values of weight and bearing capacity
        if Qv_track[-1] > W_track[-1]: # bearing capacity exceeds weight, i.e. too much embedment
            if len(Qv_track) == 1: # first iteration, taking very small embedment to check if results are within viable range before further iterations
                if PhaseNo == 1 or PhaseNo == 6: # as-laid or post-breakout where actual embedment may be larger or smaller than initial estimate so iteration needs to go both ways
                    z = 0.001
                else: # hydrotest where actual embedment can't reduce from as-laid phase, e.g. if consolidation and the absence of dynamic lay effects means hydrotest weight would iterate to smaller embedment than as-laid, the embedment remains at the as-laid level and it is more efficient to exit here than iterate and check at the end
                    z = z_ini
                    break
            elif len(Qv_track) == 2 and Qv_track[0] > W_track[0]: # second iteration; unsufficient weight for embedment using both initial guess and 0.001m, exit calculation but set to 0.001 so it doesn't cause error
                print("Embedment < 0.001m, please check inputs. z = 0.001m used for subsequent calculations of this dice roll.")
                z = 0.001
                break
            elif Qv_track[-2] < W_track[-2]: # previous estimate was too much embedment but the one before that was too little, taking mid-point for next iteration
                z = (z_track[-1] + z_track[-2])/2
            elif Qv_track[-2] > W_track[-2]: # previous two estimates were too much embedment but somewhere before that was too little, taking mid-point of latest and previous where capacity < weight for next iteration
                z = (z_track[-1] + z_track[next(x for x in range(len(z_track)-1,-1,-1) if Qv_track[x] < W_track[x])])/2

        else: # weight exceeds last bearing capacity
            if len(Qv_track) == 1: # first iteration, taking 2D embedment to check if results are within viable range before further iterations
                z = 2*D
            elif len(Qv_track) == 2 and Qv_track[0] < W_track[0]: # second iteration; too much weight for embedment using both initial guess and 2D, exit calculation but set to 2D so it doesn't cause error
                print("Embedment > 2D, please check inputs. z = 2D used for subsequent calculations of this dice roll.")
                z = 2*D
                break
            elif Qv_track[-2] > W_track[-2]: # previous estimate was too little embedment but the one before that was too much, taking mid-point for next iteration
                z = (z_track[-1] + z_track[-2])/2
            elif Qv_track[-2] < W_track[-2]: # previous two estimates were too little embedment but somewhere before that was too much, taking mid-point of latest and previous where capacity > weight for next iteration
                z = (z_track[-1] + z_track[next(x for x in range(len(z_track)-1,-1,-1) if Qv_track[x] > W_track[x])])/2;
        
        ###########################################################################
        # Calculate bearing capacity corresponding to new value of embedment, z (m)
        # Contact width, B (m) and penetrated cross-sectional area, Abm (m2)
        (B, Abm) = emb_geometry(z, D)

        if Emb_model < 10: # Undrained
            # Reference level for depth effects, z_su0 (m)
            z_su0 = depth_effects(z, D, B)

            # Undrained shear strength at key depths - mudline (z = 0), reference level (z_su0), initial estimate of pipe embedment depth (z)
            su_mudline = Common.linear_extrapolate(0, calc_depths, su_inc) # allowing extrapolation for phases where strength profile starts from pipe invert, but the strength at mudline still relevant for later calculations
            su0 = Common.linear_extrapolate(z_su0, calc_depths, su_inc) # allowing extrapolation for phases where strength profile starts from pipe invert, but the reference depth will be above this point
            su_z = np.interp(z, calc_depths, su_inc)

            # DNVGL-RP-F114 Undrained Model 1, Section 4.2.3.2
            if Emb_model == 0:
                # Bearing capacity incl. depth and bouyancy effects, Qv (kN/m)
                (str_grad_var, delta_su, Qv) = bearing_capacity_0(su_z, su_mudline, su0, z, z_su0, alpha, B, D, gamma_sub, Abm)

            # DNVGL-RP-F114 Undrained Model 2, Section 4.2.3.3
            elif Emb_model == 1:
                # Bearing capacity incl. depth and bouyancy effects, Qv (kN/m)
                Qv = bearing_capacity_1(su_z, z, D, gamma_sub, Abm)

        else: # Drained
            # DNVGL-RP-F114 Drained Model, Section 4.2.4
            # Bearing capacity, Qv (kN/m)
            Qv = bearing_capacity_10(z, B, D, phi, gamma_sub)

        Qv_track += [Qv]
        z_track += [z]

        ###########################################################################
        # Calculate touchdown lay factor, klay (manually rearranged equation from DVNGL-RP-F114 Section 4.2.5.2 to be 4th order polynomial of klay, dependant on variable z)
        # Only relevant for as-laid embedment, not hydrotest embedment
        if PhaseNo == 1: # as-laid
            klay = lay_factor(T0, EI, W, z)

        W_track += [W*klay] # klay remains set at initial value of 1 for hydrotest and post-breakout cases (Phase 3 and 6)
        #print(z_track, Qv_track, W_track)

    ###########################################################################
    # Results to pass back to master and warning messages to display if required (only those relating to z if statements as others occur as required during their individual function)
    if Emb_model == 1 and z > D/2: # DNVGL-RP-F114 Section 4.2.3.3 notes undrained model 2 may underpredict embedments where z > D/2
        print("Undrained embedments z > D/2 may be underpredicted using DNVGL-RP-F114 Model 2/SAFEBUCK, please confirm with a different model.")
    
    return z, B


def emb_geometry(z, D):
    if z >= D/2:
        B = D
        Abm = math.pi*(D**2)/8 + D*(z - D/2)
    else:
        B = 2*math.sqrt(D*z - z**2)
        Abm = (math.asin(B/D))*(D**2)/4 - B*(D/4)*math.cos(math.asin(B/D))
    return (B, Abm)


def depth_effects(z, D, B):
    # Reference level for depth effects, z_su0 (m)
    if z < (D/2)*(1-math.sqrt(2)/2):
        z_su0 = 0
    else:
        z_su0 = z + (D/2)*(math.sqrt(2)-1) - (B/2)
    return z_su0


def bearing_capacity_0(su_z, su_mudline, su0, z, z_su0, alpha, B, D, gamma_sub, Abm):
    delta_su = (su_z - su_mudline)/z # taking average gradient over initial estimate of embedded depth
    
    if delta_su > 4: # DNVGL-RP-F114 model 1 found to give significantly higher embedments for Malampaya validation when using su profile with high gradient but similar results to model 2 when using a lesser gradient, cut-off value for warning is arbitrary and can be adjusted
        print("High undrained shear strength gradients may give unrealistic results using DNVGL-RP-F114 Model 1, please confirm with a different model.")

    # Friction and strength gradient correction factor, F (-)
    (str_grad_var, F) = grad_correction(su_z, su_mudline, su0, delta_su, z, B, alpha)

    # Bearing capacity factor, Nc varying with embedment and roughness following equation 9a from Gao et al. 2017 referenced in DNVGL-RP-F114 Section 4.2.3.2 definition of this factor (note Nc = 5.14 as for strip footing BC when penetrations are small and interface is fully smooth)
    Nc = (alpha*(1-math.sqrt(1-(B/D)**2)) - 2*(math.sqrt(1-(B/D)**2)-1))/(B/D) + 1 + math.asin(alpha) + math.pi + math.sqrt(1-alpha**2) - 2*math.asin(B/D)
    
    # Bearing capacity excl. depth and bouyancy effects, Qv0 (kN/m)
    Qv0 = F*(Nc*su0 + delta_su*B/4)*B
    
    # Depth correction factor, dca (-)
    su1 = (su_mudline + su0)/2
    su2 = Qv0/(B*Nc)
    dca = 0.3*(su1/su2)*math.atan(z_su0/B)

    # Bearing capacity incl. depth and bouyancy effects, Qv (kN/m)
    Qv = Qv0*(1+dca) + gamma_sub*Abm
    return (str_grad_var, delta_su, Qv)


def grad_correction(su_z, su_mudline, su0, delta_su, z, B, alpha):
    # Friction and strength gradient correction factor from DNVGL-RP-F114 Figure 4-4, fit to 6th order polynomial in excel
    F_fit_rough = [-6E-7, 3E-5, -7E-4, 8.8E-3, -6.14E-2, 0.2706, 1.0007] # coefficients for 6th order polynomial assuming rough pipe
    F_fit_smooth = [-2E-7, 1E-5, -3E-4, 3.6E-3, -2.62E-2, 0.1341, 0.9999] # coefficients for 6th order polynomial assuming smooth pipe
    F_bounds = [0, 16] # boundary of (delta_su*B/su0) values for which 6th order polynomial can be used for F

    delta_su = (su_z - su_mudline)/z # taking average gradient over initial estimate of embedded depth
    str_grad_var = delta_su*B/su0 # variable relating to strength variation with depth, used to find correction factor F
    if str_grad_var < F_bounds[0]: # warning will display if value at the end of iteration is outside the bounds from DNV plot
        str_grad_var = 0
    elif str_grad_var > F_bounds[1]: # warning will display if value at the end of iteration is outside the bounds from DNV plot
        str_grad_var = 16
    
    F_rough = F_fit_rough[0]*str_grad_var**6 + F_fit_rough[1]*str_grad_var**5 + F_fit_rough[2]*str_grad_var**4 + F_fit_rough[3]*str_grad_var**3 + F_fit_rough[4]*str_grad_var**2 + F_fit_rough[5]*str_grad_var + F_fit_rough[6];
    F_smooth = F_fit_smooth[0]*str_grad_var**6 + F_fit_smooth[1]*str_grad_var**5 + F_fit_smooth[2]*str_grad_var**4 + F_fit_smooth[3]*str_grad_var**3 + F_fit_smooth[4]*str_grad_var**2 + F_fit_smooth[5]*str_grad_var + F_fit_smooth[6];
    if alpha == 0:
        F = F_smooth
    elif alpha == 1:
        F = F_rough
    else:
        F = (F_rough-F_smooth)*alpha + F_smooth; # linearly interpolating if selected to be between smooth and rough
    
    if str_grad_var < F_bounds[0]: # str_grad_var will be final value from iteration corresponding to z_aslaid
        print("Strength gradient variable appears erroneous (< 0), please check. Assumed delta_su*B/su0 = 0.")
    elif str_grad_var > F_bounds[1]:
        print("Strength gradient variable outside of DNVGL-PR-F114 range (> 16), please check. Assumed delta_su*B/su0 = 16.")
    
    return (str_grad_var, F)


def bearing_capacity_1(su_z, z, D, gamma_sub, Abm):
    # Bearing capacity incl. depth and bouyancy effects, Qv (kN/m)
    Qv = (min(6*(z/D)**0.25, 3.4*(10*z/D)**0.5)+1.5*gamma_sub*Abm/(D*su_z))*D*su_z
    return Qv


def bearing_capacity_10(z, B, D, phi, gamma_sub):
    # Reference level for depth effects, z_0 (m)
    if z < (D/2)*(1-math.cos(math.pi/4+np.deg2rad(phi)/2)):
        z_0 = 0
    else: # note DNV has > rather than >= but no specific = option so included here with the 'else'
        z_0 = z - (D/2) + ((D/2)/math.sin(math.pi/4+np.deg2rad(phi)/2) - B/2)*math.tan(math.pi/4+np.deg2rad(phi)/2)

    # Depth correction factor, dq (-)
    dq = 1 + 1.2*(z_0/B)*math.tan(np.deg2rad(phi))*(1-math.sin(np.deg2rad(phi)))**2

    # Bearing capacity factors, Nq and Ngamma (-)
    Nq = math.exp(math.pi*math.tan(np.deg2rad(phi)))*(math.tan(np.deg2rad(45+phi/2)))**2
    Ngamma = (1.5*(Nq-1)*math.tan(np.deg2rad(phi)) + 2*(Nq+1)*math.tan(np.deg2rad(phi)))/2 # DNV allows either Brinck-Hanson or Vesic formulation, here the average of the two has been taken

    # Bearing capacity, Qv (kN/m)
    Qv = 0.5*gamma_sub*Ngamma*B**2 + z_0*gamma_sub*Nq*dq*B
    return Qv


def lay_factor(T0, EI, W, z):
    if T0 <= (3*(EI**0.5)*W)**(2/3): # range of applicability for klay equations defined in DVNGL-RP-F114 Section 4.2.5.2
        print("Pipe inputs outside calibration range for DNVGL-RP-F114 klay equation, please check.")
    
    f = lambda x: x**4 -4*0.6*x**3 + 6*0.6**2*x**2 -(4*0.6**3+(0.4**4)*(EI*W/(z*T0**2)))*x + 0.6**4
    klay = fsolve(f,[1,3])

    klay = [x for x in klay if x >= 1] # minimum value is 1, which represents no lay touchdown impact
    if klay == []: # no roots >= 1
        klay = [1]
    klay = [x for x in klay if x <= 3] # maximum value of 3 taken from DNV-RP-F114 Section 4.2.5.2
    if klay == []: # no roots <= 3
        klay = [3]
    if isinstance(klay, list):
        if len(klay) > 1:
            print("Multiple values found for 1 <= klay <= 3, please check. First value has been adopted for subsequent calculations.")
        else:
            klay = klay[0]
    return klay

