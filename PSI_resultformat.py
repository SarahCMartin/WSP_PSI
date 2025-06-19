import matplotlib.pyplot as plt


def hard_coded_headings(fig, ax, param_name):
    if param_name == None:
        fig.tight_layout()
        return # if no name is provided this will make the plot the standard size and exit the function
    
    elif param_name == 'z_aslaid':
        fig.suptitle("As-laid Embedment Results", fontsize=14)
        ax[0].set_xlabel(r"As-laid Embedment, $z_{as-laid}$ (m)")
        ax[1].set_xlabel(r"As-laid Embedment, $z_{as-laid}$ (m)")
    
    elif param_name == 'z_hydro':
        fig.suptitle("Hydrotest Embedment Results", fontsize=14)
        ax[0].set_xlabel(r"Hydrotest Embedment, $z_{hydro}$ (m)")
        ax[1].set_xlabel(r"Hydrotest Embedment, $z_{hydro}$ (m)")

    elif param_name == 'z_res':
        fig.suptitle("Residual (post-breakout) Embedment Results", fontsize=14)
        ax[0].set_xlabel(r"Residual Embedment, $z_{res}$ (m)")
        ax[1].set_xlabel(r"Residual Embedment, $z_{res}$ (m)")

    elif param_name == 'ff_lat_brk':
        fig.suptitle("Lateral Breakout Friction Factor Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk}}$ (-)")
        ax[1].set_xlabel(r"Lateral Breakout FF, $\mu_{{lat,brk}}$ (-)")

    elif param_name == 'ff_lat_res':
        fig.suptitle("Lateral Residual Friction Factor Results", fontsize=14)
        ax[0].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res}}$ (-)")
        ax[1].set_xlabel(r"Lateral Residual FF, $\mu_{{lat,res}}$ (-)")

    elif param_name == 'ff_ax':
        fig.suptitle("Axial Friction Factor Results", fontsize=14)
        ax[0].set_xlabel(r"Axial FF, $\mu_{{ax}}$ (-)")
        ax[1].set_xlabel(r"Axial FF, $\mu_{{ax}}$ (-)")

    # Adjust layout to give space for the subtitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])
