class PSI:
    """Represents a set of all input parameters and the associated output results from PSI analysis"""
    def __init__(self, dictionary, z_aslaid=0, z_hydro=0, z_res=0, ff_lat_brk_D=0, ff_lat_brk_UD=0, y_lat_brk=[0,0,0], ff_lat_res_D=0, ff_lat_res_UD=0, y_lat_res=[0,0,0], ff_ax_D=0, ff_ax_UD=0, x_ax=[0,0,0], ff_lat_cyc=[0], ff_lat_berm=[0], ff_ax_cyc = [0], z_cyc=[0,0]):
        for k, v in dictionary.items():
            setattr(self, k, v)
        self.z_aslaid = z_aslaid
        self.z_hydro = z_hydro
        self.z_res = z_res
        self.ff_lat_brk_D = ff_lat_brk_D
        self.ff_lat_brk_UD = ff_lat_brk_UD
        self.y_lat_brk = y_lat_brk
        self.ff_lat_res_D = ff_lat_res_D
        self.ff_lat_res_UD = ff_lat_res_UD
        self.y_lat_res = y_lat_res
        self.ff_ax_D = ff_ax_D
        self.ff_ax_UD = ff_ax_UD
        self.x_ax = x_ax
        self.ff_lat_cyc = ff_lat_cyc
        self.ff_lat_berm = ff_lat_berm
        self.ff_ax_cyc = ff_ax_cyc
        self.z_cyc = z_cyc


    def __str__(self):
        temp = vars(self)
        output = ''
        for item in temp:
            output += str(item)+' = '+str(temp[item])+'\n'
        return output


    def PSI_master(self):
        """This functions runs all the components of the PSI and produces relevant outputs."""
        ###########################################################################
        # As-laid Embedment
        import PSI_soils
        PhaseNo = 1
        [insitu_calc_depths, insitu_su_inc] = PSI_soils.strength_profile(self.Emb_aslaid_model, self.Emb_hydro_model, self.Lat_brk_model, self.Lat_res_model, self.Emb_res_model, self.Ax_model, PhaseNo, self.D, self.su_profile, self.su_mudline, self.su_inv, self.z_su_inv, self.delta_su, 1, [], [], [])
        [_, emb_su_inc] = PSI_soils.strength_profile(self.Emb_aslaid_model, self.Emb_hydro_model, self.Lat_brk_model, self.Lat_res_model, self.Emb_res_model, self.Ax_model, PhaseNo, self.D, self.su_profile, self.su_mudline, self.su_inv, self.z_su_inv, self.delta_su, self.pipelay_St, [], [], [])

        import PSI_embedment
        [self.z_aslaid, B_aslaid] = PSI_embedment.embedment(PhaseNo, self.Emb_aslaid_model, self.D, self.W_empty, self.alpha, self.EI, self.T0, self.z_ini, self.gamma_sub, insitu_calc_depths, emb_su_inc, [], self.phi)
        #print("As-laid Embedment:", self.z_aslaid)
        
        ###########################################################################
        # Consolidation Between Pipelay and Hydrotest
        PhaseNo = 2
        [postlay_calc_depths, postlay_su_inc] = PSI_soils.strength_profile(self.Emb_aslaid_model, self.Emb_hydro_model, self.Lat_brk_model, self.Lat_res_model, self.Emb_res_model, self.Ax_model, PhaseNo, self.D, self.su_profile, self.su_mudline, self.su_inv, self.z_su_inv, self.delta_su, self.pipelay_St, self.z_aslaid, [], [])
        # print(postlay_calc_depths, postlay_su_inc)

        if self.Emb_aslaid_model >= 10 and self.Emb_hydro_model >= 10 and all(model>=10 for model in self.Lat_brk_model) and all(model>=10 for model in self.Lat_res_model) and all(model>=10 for model in self.Ax_model): # soil always modelled as drained
            su_consol_postlay = []
            int_su_consol_postlay = []
        else: #  soil modelled as undrained in at least 1 calculation stage (excl. cyclic where only method available used in-situ strength) so profiles need to be updated for consolidation throughout
            pressure_empty = self.W_empty/B_aslaid
            [su_consol_postlay, yield_stress_postlay, vert_eff_postlay] = PSI_soils.consolidation(PhaseNo, pressure_empty, self.t_aslaid, postlay_calc_depths, postlay_su_inc, self.gamma_sub, self.cv, self.SHANSEP_S, self.SHANSEP_m, self.D, self.z_aslaid, B_aslaid, [], [], 0)
            # Interface assumed to become normally consolidated in first pass through PSI_consolidation, so becomes unlinked to strength input profile and only defined by interface SHANSEP parameters
            [int_su_consol_postlay, int_yield_stress_postlay, int_vert_eff_postlay] = PSI_soils.consolidation(PhaseNo, pressure_empty, self.t_aslaid, postlay_calc_depths, postlay_su_inc, self.gamma_sub, self.cv, self.int_SHANSEP_S, self.int_SHANSEP_m, self.D, self.z_aslaid, B_aslaid, [], [], 1)
            # print(su_consol_postlay, yield_stress_postlay, vert_eff_postlay)

        ###########################################################################
        # Hydrotest Embedment
        PhaseNo = 3
        [self.z_hydro, B_hydro] = PSI_embedment.embedment(PhaseNo, self.Emb_hydro_model, self.D, self.W_hydro, self.alpha, [], [], self.z_aslaid, self.gamma_sub, postlay_calc_depths, su_consol_postlay, [], self.phi)
        #print("Hydrotest Embedment:", self.z_hydro)

        ###########################################################################
        # Consolidation During Hydrotest and Until Operation (incl. time flooded and empty)
        PhaseNo = 4
        [hydro_calc_depths, hydro_su_inc] = PSI_soils.strength_profile(self.Emb_aslaid_model, self.Emb_hydro_model, self.Lat_brk_model, self.Lat_res_model, self.Emb_res_model, self.Ax_model, PhaseNo, self.D, [], [], [], [], [], [], self.z_hydro, postlay_calc_depths, su_consol_postlay)
        [_,int_hydro_su_inc] = PSI_soils.strength_profile(self.Emb_aslaid_model, self.Emb_hydro_model, self.Lat_brk_model, self.Lat_res_model, self.Emb_res_model, self.Ax_model, PhaseNo, self.D, [], [], [], [], [], [], self.z_hydro, postlay_calc_depths, int_su_consol_postlay)
        # print(hydro_calc_depths, hydro_su_inc, int_hydro_su_inc)

        # Hydrotest
        if self.Emb_aslaid_model >= 10 and self.Emb_hydro_model >= 10 and all(model>=10 for model in self.Lat_brk_model) and all(model>=10 for model in self.Lat_res_model) and all(model>=10 for model in self.Ax_model): # soil always modelled as drained
            su_consol_hydro = []
            int_su_consol_hydro = []
        else: #  soil modelled as undrained in at least 1 calculation stage (excl. cyclic where only method available used in-situ strength) so profiles need to be updated for consolidation throughout
            pressure_hydro = self.W_hydro/B_hydro

            # offsetting to correct depths for additional embedment which occurred during hydrotest
            import Common
            yield_stress_postlay = Common.linear_extrapolate(hydro_calc_depths, postlay_calc_depths, yield_stress_postlay)
            vert_eff_postlay = Common.linear_extrapolate(hydro_calc_depths, postlay_calc_depths, vert_eff_postlay)
            int_yield_stress_postlay = Common.linear_extrapolate(hydro_calc_depths, postlay_calc_depths, int_yield_stress_postlay)
            int_vert_eff_postlay = Common.linear_extrapolate(hydro_calc_depths, postlay_calc_depths, int_vert_eff_postlay)

            [su_consol_hydro, yield_stress_hydro, vert_eff_hydro] = PSI_soils.consolidation(PhaseNo, pressure_hydro, self.t_hydro, hydro_calc_depths, hydro_su_inc, self.gamma_sub, self.cv, self.SHANSEP_S, self.SHANSEP_m, self.D, self.z_hydro, B_hydro, yield_stress_postlay, vert_eff_postlay, 0)
            [int_su_consol_hydro, int_yield_stress_hydro, int_vert_eff_hydro] = PSI_soils.consolidation(PhaseNo, pressure_hydro, self.t_hydro, hydro_calc_depths, int_hydro_su_inc, self.gamma_sub, self.cv, self.int_SHANSEP_S, self.int_SHANSEP_m, self.D, self.z_hydro, B_hydro, int_yield_stress_postlay, int_vert_eff_postlay, 1)
            # print(su_consol_hydro, yield_stress_hydro, vert_eff_hydro)

        # Empty between hydrotest and operation
        if self.Emb_aslaid_model >= 10 and self.Emb_hydro_model >= 10 and all(model>=10 for model in self.Lat_brk_model) and all(model>=10 for model in self.Lat_res_model) and all(model>=10 for model in self.Ax_model): # soil always modelled as drained
            su_consol_preop = []
            int_su_consol_preop = []
        else: #  soil modelled as undrained in at least 1 calculation stage (excl. cyclic where only method available used in-situ strength) so profiles need to be updated for consolidation throughout
            pressure_preop = self.W_empty/B_hydro
            [su_consol_preop, yield_stress_preop, vert_eff_preop] = PSI_soils.consolidation(PhaseNo, pressure_preop, self.t_preop, hydro_calc_depths, su_consol_hydro, self.gamma_sub, self.cv, self.SHANSEP_S, self.SHANSEP_m, self.D, self.z_hydro, B_hydro, yield_stress_hydro, vert_eff_hydro, 0)
            [int_su_consol_preop, int_yield_stress_preop, int_vert_eff_preop] = PSI_soils.consolidation(PhaseNo, pressure_preop, self.t_preop, hydro_calc_depths, int_su_consol_hydro, self.gamma_sub, self.cv, self.int_SHANSEP_S, self.int_SHANSEP_m, self.D, self.z_hydro, B_hydro, int_yield_stress_hydro, int_vert_eff_hydro, 1)
            # print(su_consol_preop, yield_stress_preop, vert_eff_preop)

        # Maximum previous vertical effective stress at interface (i.e. pipe invert level, first row in each profile from interface consolidation calc even though corresponding depth moves slightly with any additional hydrotest embedment)
        int_vert_eff_max = max([int_vert_eff_postlay[0], int_vert_eff_hydro[0], int_su_consol_preop[0]])
        # print(int_vert_eff_max)

        ###########################################################################
        # Lateral Breakout Resistance
        PhaseNo = 5
        [_, lat_su_inc] = PSI_soils.strength_profile(self.Emb_aslaid_model, self.Emb_hydro_model, self.Lat_brk_model, self.Lat_res_model, self.Emb_res_model, self.Ax_model, PhaseNo, self.D, self.su_profile, self.su_mudline, self.su_inv, self.z_su_inv, self.delta_su, self.lateral_St, [], [], [])
        # print(lat_su_inc)

        import PSI_frictionfcts
        ff_lat_brk_UD_temp = {}
        for model in self.Lat_brk_model:
            if model < 10: # Undrained
                [ff_lat_brk_UD_temp[model], self.y_lat_brk] = PSI_frictionfcts.latbrk(PhaseNo, model, self.Lat_brk_suction, self.D, self.W_op, self.alpha, int_vert_eff_max, int_vert_eff_preop[0], insitu_calc_depths, insitu_su_inc, lat_su_inc, hydro_calc_depths, su_consol_preop, self.gamma_sub, self.int_SHANSEP_S, self.int_SHANSEP_m, self.ka, self.kp, [], self.z_hydro, B_hydro)
            else: # Drained; fine to keep overriding y_lat_brk as it is unrelated to strength parameters
                [self.ff_lat_brk_D, self.y_lat_brk] = PSI_frictionfcts.latbrk(PhaseNo, model, [], self.D, self.W_op, [], [], [], [], [], [], [], [], self.gamma_sub, [], [], [], [], self.delta, self.z_hydro, B_hydro)
        
        self.ff_lat_brk_UD = apply_weighting(self.Lat_brk_weighting, ff_lat_brk_UD_temp, self.z_hydro, self.D)
        
        #print("UD Lateral Breakout FF:", self.ff_lat_brk_UD, "D Lateral Breakout FF:", self.ff_lat_brk_D, "Lateral Breakout Mobilisation Displacements:", self.y_lat_brk)

        ###########################################################################
        # Lateral Residual Resistance
        PhaseNo = 6
        if 0 in self.Lat_res_model or 2 in self.Lat_res_model or 10 in self.Lat_res_model: # methods which require instantaneous embedment into undisturbed soil profile to be calculated
            [self.z_res, B_res] = PSI_embedment.embedment(PhaseNo, self.Emb_res_model, self.D, self.W_op, self.alpha, [], [], self.z_ini, self.gamma_sub, insitu_calc_depths, insitu_su_inc, lat_su_inc, self.phi)
            #print(self.z_res, B_res)

            [res_calc_depths, res_su_inc] = PSI_soils.strength_profile(self.Emb_aslaid_model, self.Emb_hydro_model, self.Lat_brk_model, self.Lat_res_model, self.Emb_res_model, self.Ax_model, PhaseNo, self.D, [], [], [], [], [], [], self.z_res, insitu_calc_depths, insitu_su_inc)
            # print(res_calc_depths, res_su_inc)

        else: # Lat_res_model 1 or 11 (SAFEBUCK empirical relationships) use pre-breakout embedment z_hydro
            self.z_res = self.z_hydro
            B_res = B_hydro

        ff_lat_res_UD_temp = {}
        for model in self.Lat_res_model:
            if model < 10: # Undrained
                if model == 0: # in-situ yield stress ratio and existing vertical stresses needed for SHANSEP part of this model
                    # Note: this model uses interface behaviour, therefore consolidation calculation run with interface SHANSEP parameters but not assuming full remoulding as for the interface under pipelay
                    # position due to the disturbance effects of installation, therefore interface swicth set to 0 instead of 1
                    pressure_res_op = self.W_op/B_res
            
                    # Time always assumed to be 0 as this immediately follows breakout, therefore no change in su and first output can be blank
                    [_, int_yield_stress_res, int_vert_eff_res] = PSI_soils.consolidation(PhaseNo, pressure_res_op, 0, res_calc_depths, res_su_inc, self.gamma_sub, self.cv, self.int_SHANSEP_S, self.int_SHANSEP_m, self.D, self.z_res, B_res, [], [], 0)
                    # print(int_yield_stress_res, int_vert_eff_res)

                    # Max previous for subsequent SHANSEP calc will come from int_yield_stress_res[0] as strength will be defined by the past load history rather than loading phases applied during installation, hydrotest, etc.
                    [ff_lat_res_UD_temp[model], self.y_lat_res] = PSI_frictionfcts.latres(PhaseNo, model, self.Lat_res_suction, self.D, self.W_op, self.alpha, int_yield_stress_res[0], int_vert_eff_res[0], insitu_calc_depths, insitu_su_inc, self.gamma_sub, self.int_SHANSEP_S, self.int_SHANSEP_m, self.ka, self.kp, [], [], self.z_res, B_res)
                else: 
                    [ff_lat_res_UD_temp[model], self.y_lat_res] = PSI_frictionfcts.latres(PhaseNo, model, [], self.D, self.W_op, self.alpha, [], [], insitu_calc_depths, insitu_su_inc, self.gamma_sub, self.int_SHANSEP_S, self.int_SHANSEP_m, [], [], [], self.z_hydro, self.z_res, B_res)
        
            else: # Drained; fine to keep overriding y_lat_brk as it is unrelated to strength parameters
                [self.ff_lat_res_D, self.y_lat_res] = PSI_frictionfcts.latres(PhaseNo, model, [], self.D, self.W_op, [], [], [], [], [], self.gamma_sub, [], [], [], [], self.delta, [], self.z_res, [])
        
        self.ff_lat_res_UD = apply_weighting(self.Lat_res_weighting, ff_lat_res_UD_temp, self.z_res, self.D)

        #print("UD Lateral Residual FF:", self.ff_lat_res_UD, "D Lateral Residual FF:", self.ff_lat_res_D, "Lateral Residual Mobilisation Displacements:", self.y_lat_res)

        ###########################################################################
        # Axial Resistance
        PhaseNo = 7
        for model in self.Ax_model:
            if model < 10: # Undrained
                [self.ff_ax_UD, self.x_ax] = PSI_frictionfcts.axial(model, self.D, self.W_op, self.alpha, int_vert_eff_max, int_vert_eff_preop[0], self.int_SHANSEP_S, self.int_SHANSEP_m, [], self.z_hydro, B_hydro)
            else: # Drained; fine to keep overriding y_lat_brk as it is unrelated to strength parameters
                [self.ff_ax_D, self.x_ax] = PSI_frictionfcts.axial(model, self.D, self.W_op, [], [], [], [], [], self.delta, self.z_hydro, B_hydro)
            
        #print("UD Axial FF:", self.ff_ax_UD, "D Axial FF:", self.ff_ax_D, "Axial Mobilisation Displacements:", self.x_ax)

        ###########################################################################
        # Cyclic Resistances
        PhaseNo = 8
        if self.Cyc_model == 0 or self.Cyc_model == 1: # SAFEBUCK or White & Cheuk, only UD lateral model available; note it is also debatable whether initial or residual embedment should be used as the former is suggested but the latter gives more comparible results to other methods
            # [self.ff_lat_cyc, self.ff_lat_berm, self.z_cyc] = PSI_frictionfcts.latcyc(self.Cyc_model, self.No_cycles, self.D, self.W_op, self.z_hydro, insitu_calc_depths, insitu_su_inc, self.gamma_sub)
            # print("Cyclic Lat Mid-Sweep FF:", self.ff_lat_cyc, "Cyclic Lat Berm FF:", self.ff_lat_berm, "Cyclic Embedment:", self.z_cyc)
            # self.ff_ax_cyc = []
            # print("No Cyclic Axial avaiable for the selected model")
            print("Empirical cyclic methodologies are considered unreliable; the in-house UD to D transition method is recommended")
        else: 
            # Lateral Berm
            # Capping lateral berm FF using drained lateral breakout at z = D embedment; this may under-estimate the peak height of the berm but it will also not be functioning as a standard passive triangle like if the pipe were fully covered
            z_cap = self.D
            B_cap = PSI_embedment.emb_geometry(z_cap, self.D)
            [lat_berm_cap, _] = PSI_frictionfcts.latbrk(PhaseNo, 10, [], self.D, self.W_op, [], [], [], [], [], [], [], [], self.gamma_sub, [], [], [], [], self.delta, z_cap, B_cap)
            [self.ff_lat_berm] = PSI_frictionfcts.cyc_transition(self.N50, self.ff_lat_res_UD, lat_berm_cap, self.No_cycles)

            # Lateral Mid-Sweep
            # Capping lateral mid-sweep FF using drained lateral breakout at z = 0 embedment to remove passive component
            z_cap = 0
            B_cap = 0
            [lat_mid_cap, _] = PSI_frictionfcts.latbrk(PhaseNo, 10, [], self.D, self.W_op, [], [], [], [], [], [], [], [], self.gamma_sub, [], [], [], [], self.delta, z_cap, B_cap)
            [self.ff_lat_cyc] = PSI_frictionfcts.cyc_transition(self.N50, self.ff_lat_res_UD, lat_berm_cap, self.No_cycles)

            # Axial
            [self.ff_ax_cyc] = PSI_frictionfcts.cyc_transition(self.N50, self.ff_ax_UD, self.ff_ax_D, self.No_cycles)

        ###########################################################################
        # Producing figures of undrained shear strength evolution
        #if self.Emb_aslaid_model < 10 or self.Emb_hydro_model < 10 or self.Lat_brk_model < 10 or self.Lat_res_model < 10 or self.Ax_model < 10: # soil modelled as undrained in at least 1 calculation stage (excl. cyclic where the only method available uses in-situ strength) so profiles developed throughout, no figures if all strages drained
        #    PSI_results.strengthplot(insitu_calc_depths, insitu_su_inc, emb_su_inc, lat_su_inc, postlay_calc_depths, postlay_su_inc, su_consol_postlay, hydro_calc_depths, su_consol_hydro, su_consol_preop, 0)
        #    PSI_results.strengthplot(insitu_calc_depths, insitu_su_inc, [], [], postlay_calc_depths, [], int_su_consol_postlay, hydro_calc_depths, int_su_consol_hydro, int_su_consol_preop, 1)

        return self
    

def apply_weighting(weighting_range, dict, z, D):
    import numpy as np
    # DNV Model 1 (non-empirical) is more reliable at low embedment than Generalised FE Envelope Method, so varying from max weighting for this at z=0 to min at z=0.5D
    # DNV Model 2 / SAFEBUCK (empirical) is unreliable above z=0.5D, so varying from max weighting for this at z=0 to 0 at z=0.5D 
    zmin = 0
    zmax = 0.5*D
    if z <= zmin:
        weighting = weighting_range[0]/100
        weighting_SFBK = 1-weighting_range[0]/100
    elif z >= zmax:
        weighting = weighting_range[1]/100
        weighting_SFBK = 0
    else:
        weighting = np.interp(z, [zmin, zmax], [x/100 for x in weighting_range])
        weighting_SFBK = np.interp(z, [zmin, zmax], [1-weighting_range[0]/100, 0])

    weighted_sum = 0
    total_weighting = 0 # should add up to 1, but allowing it to be different if someone selected 3 UD models, for example
    for model in dict:
        if model == 2: # Generalised FE Envelope Method
            if isinstance(dict[2], float) and not np.isnan(dict[2]):
                weighted_sum = weighted_sum + dict[2]*weighting
                total_weighting = total_weighting + weighting
        elif model == 0: # DNV non-empirical
            if isinstance(dict[0], float) and not np.isnan(dict[0]):
                weighted_sum = weighted_sum + dict[0]*(1-weighting)
                total_weighting = total_weighting + (1-weighting)
        else: # SAFEBUCK, unlikely to be used but included here in case this or another model is ultimately included as model 1
            if isinstance(dict[1], float) and not np.isnan(dict[1]):
                weighted_sum = weighted_sum + dict[1]*weighting_SFBK
                total_weighting = total_weighting + weighting_SFBK

    return weighted_sum/total_weighting