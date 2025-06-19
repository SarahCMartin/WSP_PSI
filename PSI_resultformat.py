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

    elif param_name == 'ff_lat_brk':
        fig.suptitle("Lateral Breakout Friction Factor - Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk}}$ (-)")
        ax[1].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk}}$ (-)")

    elif param_name == 'ff_lat_res':
        fig.suptitle("Lateral Residual Friction Factor - Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res}}$ (-)")
        ax[1].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res}}$ (-)")

    elif param_name == 'ff_ax':
        fig.suptitle("Axial Friction Factor - Results", fontsize=14)
        ax[0].set_xlabel(r"Axial FF, $\mu_{{ax}}$ (-)")
        ax[1].set_xlabel(r"Axial FF, $\mu_{{ax}}$ (-)")

    # Adjust layout to give space for the subtitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])
