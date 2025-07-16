import matplotlib.pyplot as plt


def hard_coded_headings(fig, ax, param_name):
    if param_name == None:
        fig.tight_layout()
        return # if no name is provided this will make the plot the standard size and exit the function
    
    ###########################################################################
    # Inputs
    elif param_name == 'D':
        fig.suptitle("Diameter - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel("Diameter, D (m)")
        ax[1].set_xlabel("Diameter, D (m)")

    elif param_name == 't':
        fig.suptitle("Wall Thickness - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel("Wall Thickness, t (m)")
        ax[1].set_xlabel("Wall Thickness, t (m)")

    elif param_name == 'W_empty':
        fig.suptitle("Submerged Empty Unit Weight - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Submerged Empty Unit Weight, $W_{empty}$ (kN/m)")
        ax[1].set_xlabel(r"Submerged Empty Unit Weight, $W_{empty}$ (kN/m)")

    elif param_name == 'W_hydro':
        fig.suptitle("Submerged Flooded Unit Weight - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Submerged Flooded Unit Weight, $W_{hydro}$ (kN/m)")
        ax[1].set_xlabel(r"Submerged Flooded Unit Weight, $W_{hydro}$ (kN/m)")

    elif param_name == 'W_op':
        fig.suptitle("Submerged Operational Unit Weight - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Submerged Operational Unit Weight, $W_{op}$ (kN/m)")
        ax[1].set_xlabel(r"Submerged Operational Unit Weight, $W_{op}$ (kN/m)")

    elif param_name == 'alpha':
        fig.suptitle("Pipe-Soil Interface Roughness Coefficient - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Pipe-Soil Interface Roughness, $\alpha$ (-)")
        ax[1].set_xlabel(r"Pipe-Soil Interface Roughness, $\alpha$ (-)")

    elif param_name == 'EI':
        fig.suptitle("Bending Stiffness - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Bending Stiffness, EI (kNm$^2$)")
        ax[1].set_xlabel(r"Bending Stiffness, EI (kNm$^2$)")

    elif param_name == 'T0':
        fig.suptitle("Lay Tension as Touchdown - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Lay Tension, $T_0$ (kN)")
        ax[1].set_xlabel(r"Lay Tension, $T_0$ (kN)")

    elif param_name == 't_aslaid':
        fig.suptitle("Time Empty between Pipelay and Hydrotest - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Time Empty before Hydrotest, $t_{\text{as-laid}}$ (years)")
        ax[1].set_xlabel(r"Time Empty before Hydrotest, $t_{\text{as-laid}}$ (years)")

    elif param_name == 't_hydro':
        fig.suptitle("Time Flooded during Hydrotest/Wet-Parked - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Time Flooded during/after Hydrotest, $t_{flooded}$ (years)")
        ax[1].set_xlabel(r"Time Flooded during/after Hydrotest, $t_{flooded}$ (years)")

    elif param_name == 't_preop':
        fig.suptitle("Time Dry-Parked between Hydrotest and Start of Operation - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Time Empty after Hydrotest, $t_{\text{pre-op}}$ (years)")
        ax[1].set_xlabel(r"Time Empty after Hydrotest, $t_{\text{pre-op}}$ (years)")

    elif param_name == 'su_mudline':
        fig.suptitle("Undrained Shear Strength at Mudline - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Undrained Shear Strength at Mudline, $s_{{u,z=0}}$ (kPa)")
        ax[1].set_xlabel(r"Undrained Shear Strength at Mudline, $s_{{u,z=0}}$ (kPa)")

    elif param_name == 'su_inv':
        fig.suptitle(r"Undrained Shear Strength at $z_{inv}$ - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Undrained Shear Strength at $z_{inv}$, $s_{{u,z=inv}}$ (kPa)")
        ax[1].set_xlabel(r"Undrained Shear Strength at $z_{inv}$, $s_{{u,z=inv}}$ (kPa)")

    elif param_name == 'delta_su':
        fig.suptitle(r"Gradient of Undrained Shear Strength Increase below $z_{inv}$ - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Undrained Shear Strength Gradient, $\Delta s_{u}$ (kPa/m)")
        ax[1].set_xlabel(r"Undrained Shear Strength Gradient, $\Delta s_{u}$ (kPa/m)")

    elif param_name == 'gamma_sub':
        fig.suptitle("Submerged Unit Weight of Soil - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Submerged Unit Weight of Soil, $\gamma\,'$ (kN/m$^3$)")
        ax[1].set_xlabel(r"Submerged Unit Weight of Soil, $\gamma\,'$ (kN/m$^3$)")

    elif param_name == 'pipelay_St':
        fig.suptitle("Soil Sensitivity for Pipelay Calculations - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Pipelay Soil Sensitivity, $S_{{t,pipelay}}$ (-)")
        ax[1].set_xlabel(r"Pipelay Soil Sensitivity, $S_{{t,pipelay}}$ (-)")

    elif param_name == 'lateral_St':
        fig.suptitle("Soil Sensitivity for Lateral Breakout Calculations - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Lateral Soil Sensitivity, $S_{{t,lateral}}$ (-)")
        ax[1].set_xlabel(r"Lateral Soil Sensitivity, $S_{{t,lateral}}$ (-)")

    elif param_name == 'cv':
        fig.suptitle("Coefficient of Consolidation - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"Coefficient of Consolidation, $c_v$ (m$^2$/year)")
        ax[1].set_xlabel(r"Coefficient of Consolidation, $c_v$ (m$^2$/year)")

    elif param_name == 'SHANSEP_S':
        fig.suptitle("Normalised Shear Strength for NC Condition (Soil-Soil) - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"SHANSEP S for Soil-Soil, $S_{soil}$ (-)")
        ax[1].set_xlabel(r"SHANSEP S for Soil-Soil, $S_{soil}$ (-)")

    elif param_name == 'SHANSEP_m':
        fig.suptitle("SHANSEP Exponent (Soil-Soil) - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"SHANSEP m for Soil-Soil, $m_{soil}$ (-)")
        ax[1].set_xlabel(r"SHANSEP m for Soil-Soil, $m_{soil}$ (-)")

    elif param_name == 'ka':
        fig.suptitle("DNV Pressure Resistance Coefficient - Active", fontsize=14)
        ax[0].set_xlabel(r"Pressure Resistance Coefficient, $k_a$ (-)")
        ax[1].set_xlabel(r"Pressure Resistance Coefficient, $k_a$ (-)")

    elif param_name == 'kp':
        fig.suptitle("DNV Pressure Resistance Coefficient - Passive", fontsize=14)
        ax[0].set_xlabel(r"Pressure Resistance Coefficient, $k_p$ (-)")
        ax[1].set_xlabel(r"Pressure Resistance Coefficient, $k_p$ (-)")

    elif param_name == 'phi':
        fig.suptitle("Friction Angle (Soil-Soil)", fontsize=14)
        ax[0].set_xlabel(r"Friction Angle for Soil-Soil, $\phi$ ($^\circ$)")
        ax[1].set_xlabel(r"Friction Angle for Soil-Soil, $\phi$ ($^\circ$)")

    elif param_name == 'int_SHANSEP_S':
        fig.suptitle("Normalised Shear Strength for NC Condition (Soil-Interface) - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"SHANSEP S for Soil-Interface, $S_{int}$ (-)")
        ax[1].set_xlabel(r"SHANSEP S for Soil-Interface, $S_{int}$ (-)")

    elif param_name == 'int_SHANSEP_m':
        fig.suptitle("SHANSEP Exponent (Soil-Interface) - Fit and Generated Inputs", fontsize=14)
        ax[0].set_xlabel(r"SHANSEP m for Soil-Interface, $m_{int}$ (-)")
        ax[1].set_xlabel(r"SHANSEP m for Soil-Interface, $m_{int}$ (-)")

    elif param_name == 'delta':
        fig.suptitle("Friction Angle (Soil-Interface)", fontsize=14)
        ax[0].set_xlabel(r"Friction Angle for Soil-Interface, $\delta$ ($^\circ$)")
        ax[1].set_xlabel(r"Friction Angle for Soil-Interface, $\delta$ ($^\circ$)")

    ###########################################################################
    # Outputs
    elif param_name == 'z_aslaid':
        fig.suptitle("As-laid Embedment - Results", fontsize=14)
        ax[0].set_xlabel(r"As-laid Embedment, $z_{\text{as-laid}}$ (m)")
        ax[1].set_xlabel(r"As-laid Embedment, $z_{\text{as-laid}}$ (m)")
    
    elif param_name == 'z_hydro':
        fig.suptitle("Hydrotest Embedment - Results", fontsize=14)
        ax[0].set_xlabel(r"Hydrotest Embedment, $z_{hydro}$ (m)")
        ax[1].set_xlabel(r"Hydrotest Embedment, $z_{hydro}$ (m)")

    elif param_name == 'z_res':
        fig.suptitle("Residual (post-breakout) Embedment - Results", fontsize=14)
        ax[0].set_xlabel(r"Residual Embedment, $z_{res}$ (m)")
        ax[1].set_xlabel(r"Residual Embedment, $z_{res}$ (m)")

    elif param_name == 'ff_lat_brk_UD':
        fig.suptitle("Lateral Breakout Friction Factor - Undrained Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk,UD}}$ (-)")
        ax[1].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk,UD}}$ (-)")
    
    elif param_name == 'ff_lat_brk_D':
        fig.suptitle("Lateral Breakout Friction Factor - Drained Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk,D}}$ (-)")
        ax[1].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk,D}}$ (-)")

    elif param_name == 'ff_lat_res_UD':
        fig.suptitle("Lateral Residual Friction Factor - Undrained Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res,UD}}$ (-)")
        ax[1].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res,UD}}$ (-)")

    elif param_name == 'ff_lat_res_D':
        fig.suptitle("Lateral Residual Friction Factor - Drained Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res,D}}$ (-)")
        ax[1].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res,D}}$ (-)")

    elif param_name == 'ff_ax_UD':
        fig.suptitle("Axial Friction Factor - Undrained Results", fontsize=14)
        ax[0].set_xlabel(r"Axial FF, $\mu_{{ax,UD}}$ (-)")
        ax[1].set_xlabel(r"Axial FF, $\mu_{{ax,UD}}$ (-)")

    elif param_name == 'ff_ax_D':
        fig.suptitle("Axial Friction Factor - Drained Results", fontsize=14)
        ax[0].set_xlabel(r"Axial FF, $\mu_{{ax,D}}$ (-)")
        ax[1].set_xlabel(r"Axial FF, $\mu_{{ax,D}}$ (-)")

    # Adjust layout to give space for the subtitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def hard_coded_corr_headings(ax, name_base_var, name_changing_var):
    if name_base_var == None and name_changing_var == None:
        return # if no name is provided this will make the plot the standard size and exit the function
    
    ###########################################################################
    # Horizontal Axis - base variable
    if name_base_var == 'D':
        ax.set_xlabel("Diameter, D (m)")
    elif name_base_var == 't':
        ax.set_xlabel("Wall Thickness, t (m)")
    elif name_base_var == 'W_empty':
        ax.set_xlabel(r"Submerged Empty Unit Weight, $W_{empty}$ (kN/m)")
    elif name_base_var == 'W_hydro':
        ax.set_xlabel(r"Submerged Flooded Unit Weight, $W_{hydro}$ (kN/m)")
    elif name_base_var == 'W_op':
        ax.set_xlabel(r"Submerged Operational Unit Weight, $W_{op}$ (kN/m)")
    elif name_base_var == 'alpha':
        ax.set_xlabel(r"Pipe-Soil Interface Roughness, $\alpha$ (-)")
    elif name_base_var == 'EI':
        ax.set_xlabel(r"Bending Stiffness, EI (kNm$^2$)")
    elif name_base_var == 'T0':
        ax.set_xlabel(r"Lay Tension, $T_0$ (kN)")
    elif name_base_var == 't_aslaid':
        ax.set_xlabel(r"Time Empty before Hydrotest, $t_{\text{as-laid}}$ (years)")
    elif name_base_var == 't_hydro':
        ax.set_xlabel(r"Time Flooded during/after Hydrotest, $t_{flooded}$ (years)")
    elif name_base_var == 't_preop':
        ax.set_xlabel(r"Time Empty after Hydrotest, $t_{\text{pre-op}}$ (years)")
    elif name_base_var == 'su_mudline':
        ax.set_xlabel(r"Undrained Shear Strength at Mudline, $s_{{u,z=0}}$ (kPa)")
    elif name_base_var == 'su_inv':
        ax.set_xlabel(r"Undrained Shear Strength at $z_{inv}$, $s_{{u,z=inv}}$ (kPa)")
    elif name_base_var == 'delta_su':
        ax.set_xlabel(r"Undrained Shear Strength Gradient, $\Delta s_{u}$ (kPa/m)")
    elif name_base_var == 'gamma_sub':
        ax.set_xlabel(r"Submerged Unit Weight of Soil, $\gamma\,'$ (kN/m$^3$)")
    elif name_base_var == 'pipelay_St':
        ax.set_xlabel(r"Pipelay Soil Sensitivity, $S_{{t,pipelay}}$ (-)")
    elif name_base_var == 'lateral_St':
        ax.set_xlabel(r"Lateral Soil Sensitivity, $S_{{t,lateral}}$ (-)")
    elif name_base_var == 'cv':
        ax.set_xlabel(r"Coefficient of Consolidation, $c_v$ (m$^2$/year)")
    elif name_base_var == 'SHANSEP_S':
        ax.set_xlabel(r"SHANSEP S for Soil-Soil, $S_{soil}$ (-)")
    elif name_base_var == 'SHANSEP_m':
        ax.set_xlabel(r"SHANSEP m for Soil-Soil, $m_{soil}$ (-)")
    elif name_base_var == 'ka':
        ax.set_xlabel(r"Pressure Resistance Coefficient, $k_a$ (-)")
    elif name_base_var == 'kp':
        ax.set_xlabel(r"Pressure Resistance Coefficient, $k_p$ (-)")
    elif name_base_var == 'phi':
        ax.set_xlabel(r"Friction Angle for Soil-Soil, $\phi$ ($^\circ$)")
    elif name_base_var == 'int_SHANSEP_S':
        ax.set_xlabel(r"SHANSEP S for Soil-Interface, $S_{int}$ (-)")
    elif name_base_var == 'int_SHANSEP_m':
        ax.set_xlabel(r"SHANSEP m for Soil-Interface, $m_{int}$ (-)")
    elif name_base_var == 'delta':
        ax.set_xlabel(r"Friction Angle for Soil-Interface, $\delta$ ($^\circ$)")

    
    ###########################################################################
    # Vertical Axis - changing variable
    if name_changing_var == 'D':
        ax.set_ylabel("Diameter, D (m)")
    elif name_changing_var == 't':
        ax.set_ylabel("Wall Thickness, t (m)")
    elif name_changing_var == 'W_empty':
        ax.set_ylabel(r"Submerged Empty Unit Weight, $W_{empty}$ (kN/m)")
    elif name_changing_var == 'W_hydro':
        ax.set_ylabel(r"Submerged Flooded Unit Weight, $W_{hydro}$ (kN/m)")
    elif name_changing_var == 'W_op':
        ax.set_ylabel(r"Submerged Operational Unit Weight, $W_{op}$ (kN/m)")
    elif name_changing_var == 'alpha':
        ax.set_ylabel(r"Pipe-Soil Interface Roughness, $\alpha$ (-)")
    elif name_changing_var == 'EI':
        ax.set_ylabel(r"Bending Stiffness, EI (kNm$^2$)")
    elif name_changing_var == 'T0':
        ax.set_ylabel(r"Lay Tension, $T_0$ (kN)")
    elif name_changing_var == 't_aslaid':
        ax.set_ylabel(r"Time Empty before Hydrotest, $t_{\text{as-laid}}$ (years)")
    elif name_changing_var == 't_hydro':
        ax.set_ylabel(r"Time Flooded during/after Hydrotest, $t_{flooded}$ (years)")
    elif name_changing_var == 't_preop':
        ax.set_ylabel(r"Time Empty after Hydrotest, $t_{\text{pre-op}}$ (years)")
    elif name_changing_var == 'su_mudline':
        ax.set_ylabel(r"Undrained Shear Strength at Mudline, $s_{{u,z=0}}$ (kPa)")
    elif name_changing_var == 'su_inv':
        ax.set_ylabel(r"Undrained Shear Strength at $z_{inv}$, $s_{{u,z=inv}}$ (kPa)")
    elif name_changing_var == 'delta_su':
        ax.set_ylabel(r"Undrained Shear Strength Gradient, $\Delta s_{u}$ (kPa/m)")
    elif name_changing_var == 'gamma_sub':
        ax.set_ylabel(r"Submerged Unit Weight of Soil, $\gamma\,'$ (kN/m$^3$)")
    elif name_changing_var == 'pipelay_St':
        ax.set_ylabel(r"Pipelay Soil Sensitivity, $S_{{t,pipelay}}$ (-)")
    elif name_changing_var == 'lateral_St':
        ax.set_ylabel(r"Lateral Soil Sensitivity, $S_{{t,lateral}}$ (-)")
    elif name_changing_var == 'cv':
        ax.set_ylabel(r"Coefficient of Consolidation, $c_v$ (m$^2$/year)")
    elif name_changing_var == 'SHANSEP_S':
        ax.set_ylabel(r"SHANSEP S for Soil-Soil, $S_{soil}$ (-)")
    elif name_changing_var == 'SHANSEP_m':
        ax.set_ylabel(r"SHANSEP m for Soil-Soil, $m_{soil}$ (-)")
    elif name_changing_var == 'ka':
        ax.set_ylabel(r"Pressure Resistance Coefficient, $k_a$ (-)")
    elif name_changing_var == 'kp':
        ax.set_ylabel(r"Pressure Resistance Coefficient, $k_p$ (-)")
    elif name_changing_var == 'phi':
        ax.set_ylabel(r"Friction Angle for Soil-Soil, $\phi$ ($^\circ$)")
    elif name_changing_var == 'int_SHANSEP_S':
        ax.set_ylabel(r"SHANSEP S for Soil-Interface, $S_{int}$ (-)")
    elif name_changing_var == 'int_SHANSEP_m':
        ax.set_ylabel(r"SHANSEP m for Soil-Interface, $m_{int}$ (-)")
    elif name_changing_var == 'delta':
        ax.set_ylabel(r"Friction Angle for Soil-Interface, $\delta$ ($^\circ$)")


def hard_coded_caption(param_name):    
    if param_name == None:
        return # if no name is provided then exit the function
    
    ###########################################################################
    # Inputs
    elif param_name == 'D':
        param_string = "Diameter, D"
    elif param_name == 't':
        param_string = "Wall Thickness, t"
    elif param_name == 'W_empty':
        param_string = r"Submerged Empty Unit Weight, $W_{empty}$"
    elif param_name == 'W_hydro':
        param_string = r"Submerged Flooded Unit Weight, $W_{hydro}$"
    elif param_name == 'W_op':
        param_string = r"Submerged Operational Unit Weight, $W_{op}$"
    elif param_name == 'alpha':
        param_string = r"Pipe-Soil Interface Roughness, $\alpha$"
    elif param_name == 'EI':
        param_string = r"Bending Stiffness, EI"
    elif param_name == 'T0':
        param_string = r"Lay Tension, $T_0$"
    elif param_name == 't_aslaid':
        param_string = r"Time Empty before Hydrotest, $t_{\text{as-laid}}$"
    elif param_name == 't_hydro':
        param_string = r"Time Flooded during/after Hydrotest, $t_{flooded}$"
    elif param_name == 't_preop':
        param_string = r"Time Empty after Hydrotest, $t_{\text{pre-op}}$"
    elif param_name == 'su_mudline':
        param_string = r"Undrained Shear Strength at Mudline, $s_{{u,z=0}}$"
    elif param_name == 'su_inv':
        param_string = r"Undrained Shear Strength at $z_{inv}$, $s_{{u,z=inv}}$"
    elif param_name == 'delta_su':
        param_string = r"Undrained Shear Strength Gradient, $\Delta s_{u}$"
    elif param_name == 'gamma_sub':
        param_string = r"Submerged Unit Weight of Soil, $\gamma\,'$"
    elif param_name == 'pipelay_St':
        param_string = r"Pipelay Soil Sensitivity, $S_{{t,pipelay}}$"
    elif param_name == 'lateral_St':
        param_string = r"Lateral Soil Sensitivity, $S_{{t,lateral}}$"
    elif param_name == 'cv':
        param_string = r"Coefficient of Consolidation, $c_v$"
    elif param_name == 'SHANSEP_S':
        param_string = r"SHANSEP S for Soil-Soil, $S_{soil}$"
    elif param_name == 'SHANSEP_m':
        param_string = r"SHANSEP m for Soil-Soil, $m_{soil}$"
    elif param_name == 'ka':
        param_string = r"Pressure Resistance Coefficient, $k_a$"
    elif param_name == 'kp':
        param_string = r"Pressure Resistance Coefficient, $k_p$"
    elif param_name == 'phi':
        param_string = r"Friction Angle for Soil-Soil, $\phi$"
    elif param_name == 'int_SHANSEP_S':
        param_string = r"SHANSEP S for Soil-Interface, $S_{int}$"
    elif param_name == 'int_SHANSEP_m':
        param_string = r"SHANSEP m for Soil-Interface, $m_{int}$"
    elif param_name == 'delta':
        param_string = r"Friction Angle for Soil-Interface, $\delta$"

    ###########################################################################
    # Outputs
    elif param_name == 'z_aslaid':
        param_string = r"As-laid Embedment, $z_{\text{as-laid}}$"
    elif param_name == 'z_hydro':
        param_string = r"Hydrotest Embedment, $z_{hydro}$"
    elif param_name == 'z_res':
        param_string = r"Residual Embedment, $z_{res}$"
    elif param_name == 'ff_lat_brk_UD':
        param_string = r"UD Lateral Breakout FF, $\mu_{{lat,brk,UD}}$"
    elif param_name == 'ff_lat_brk_D':
        param_string = r"D Lateral Breakout FF, $\mu_{{lat,brk,D}}$"
    elif param_name == 'ff_lat_res_UD':
        param_string = r"UD Lateral Residual FF, $\mu_{{lat,res,UD}}$"
    elif param_name == 'ff_lat_res_D':
        param_string = r"D Lateral Residual FF, $\mu_{{lat,res,D}}$"
    elif param_name == 'ff_ax_UD':
        param_string = r"UD Axial FF, $\mu_{{ax,UD}}$"
    elif param_name == 'ff_ax_D':
        param_string = r"D Axial FF, $\mu_{{ax,D}}$"

    else:
        return # if name doesn't match those with a hard coded option then exit the function
    
    return param_string


def hard_coded_units(param_name):
    if param_name == None:
        unit = '(-)'
    
    elif param_name == 'D' or param_name == 't' or param_name == 'z_aslaid' or param_name == 'z_hydro' or param_name == 'z_res':
        unit = '(m)'
    elif param_name == 'W_empty' or param_name == 'W_hydro' or param_name == 'W_op':
        unit = '(kN/m)'
    elif param_name == 'EI':
        unit = r"(kNm$^2$)"
    elif param_name == 'T0':
        unit = "(kN)"
    elif param_name == 't_aslaid' or param_name == 't_hydro' or param_name == 't_preop':
        unit = '(years)'
    elif param_name == 'su_mudline' or param_name == 'su_inv':
        unit = '(kPa)'
    elif param_name == 'delta_su':
        unit = '(kPa/m)'
    elif param_name == 'gamma_sub':
        unit = r"(kN/m$^3$)"
    elif param_name == 'cv':
        unit = r"(m$^2$/year)"
    elif param_name == 'phi' or param_name == 'delta':
        unit = r"($^\circ$)"

    else:
        unit = '(-)'
    
    return unit