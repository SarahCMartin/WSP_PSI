# PSI_rerun

"""This script runs Pipe-Soil Interaction Calculations for a set of 
already generated input parameters. This is to allow the same set of 
inputs to be used repeatedly to compare any required changes to the 
code or for validation/ troubleshooting purposes where the user wants 
to use the same set repeatedly."""

###########################################################################
# Housekeeping
import os
directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(directory) # finding folder where this script is saved and setting as the working directory - if all functions are saved in the same folder this will prevent any directory errors
d = {} # empty dictionary to take input parameters to add to PSI_case object in the initiation step

###########################################################################
# Import inputs from selected json results filed, saved after original run of those parameters
import Common
file_name = Common.import_json()

import json

if file_name: # Only save if user selected a file name
    with open(file_name, 'r') as file:
        compiled = json.load(file)
else:
    print("Results file not selected")

###########################################################################
# Select location to save results and figures from the new run
from pathlib import Path
parent_dir = Path(Common.select_folder())

if parent_dir.name.lower() == "results":
    results_path = parent_dir
else:
    results_path = parent_dir / 'Results'
    if not results_path.exists():
        results_path.mkdir()

###########################################################################
# Read Monte Carlo and Model Selection Parameters from selected json file
var_names = ['No_rolls', 'Emb_aslaid_model', 'Emb_hydro_model', 'Lat_brk_suction', 'Lat_res_suction', 'Emb_res_model', 'Cyc_model', 'N50', 'su_profile', 'z_su_inv', 'Output_dist', 'Model_fct', 'Lat_brk_model', 'Lat_brk_weighting', 'Lat_res_model', 'Lat_res_weighting', 'Ax_model', 'No_cycles', 'Berm', 'Spanning']
d = {name: compiled[0]['inputs'][name] for name in var_names} # dictionary containing constant parameters (don't need to separate list inputs when reading from the json vs when reading from excel)

# Handle special logic for z_su_inv
if d['su_profile'] != 1:
    d['z_su_inv'] = []

###########################################################################
# Run PSI for prescribed number of parameter sets (rolls)
# No_rolls = compiled[0]['inputs']['No_rolls']
from PSI_class import PSI
import PSI_soils
results = []

for i in range(d['No_rolls']):
    # Compile model parameters (fixed variables) with a set of pipe, soil and interface parameters for that roll (random probabilistic variables)
    input_dict = compiled[i]['inputs']

    # Run model for a given roll with the corresponding set of 1 of each parameters
    PSI_case = PSI(input_dict)
    PSI_case = PSI_case.PSI_master()

    # Compiling results dictionaries - 1 for inputs and 1 for outputs for every roll (inputs taken directly in case they are edited at all throughout the running process)
    input_keys = set(input_dict.keys())
    output_dict = {
        attr: getattr(PSI_case, attr)  
        for attr in dir(PSI_case) 
        if not attr.startswith("_") 
        and not callable(getattr(PSI_case, attr)) 
        and attr not in input_keys
        }
    results.append({"inputs": input_dict, "outputs": output_dict})

# Converting to lists of each variable to be easier to plot outputs
list_inputs = {}
list_results = {}

for entry in results:
    for k, v in entry['inputs'].items():
        list_inputs.setdefault(k, []).append(v)
    for k, v in entry['outputs'].items():
        list_results.setdefault(k, []).append(v)

# suffix_map = {
#     2: ["LE", "HE"],
#     3: ["LE", "BE", "HE"],
# }

# for key, value in list(list_results.items()):
#     if (
#         isinstance(value, list)
#         and all(isinstance(i, list) for i in value)
#         and len(value) > 0
#         and all(len(i) == len(value[0]) for i in value) # consistent inner length
#         and len(value[0]) in suffix_map                 # only process 2 or 3 values because these are known to be LE-HE or LE-BE-HE, any other length >1 would be erroneous
#     ):
#         # Transpose the list of lists
#         transposed = list(map(list, zip(*value)))

#         # Assign each column to a new key with a suffix
#         for i, suffix in enumerate(suffix_map[len(value[0])]):
#             list_results[f"{key}_{suffix}"] = transposed[i]

#         # Remove the original unsplit key
#         del list_results[key]

###########################################################################
# Saving inputs and results
os.chdir(results_path)
file_name = Common.get_filename()

import json

if file_name: # Only save if user selected a file name
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Dictionary saved to {file_name}. Note: this saved data does not include model factors so that they can be varied as required in post-processing / re-processing.")
else:
    print("Results dictionary not saved")

###########################################################################
# Plotting results and fitting distributions to them
to_fit_and_plot = ['zD_aslaid', 'zD_hydro', 'zD_op_eff', 'zD_res', 'ff_lat_brk_UD', 'ff_lat_brk_D', 'ff_lat_res_UD', 'ff_lat_res_D', 'ff_ax_UD', 'ff_ax_D']

cyclic_keys = ['ff_lat_berm', 'ff_lat_cyc', 'ff_ax_cyc']
for key in cyclic_keys:
    list_of_lists = list_results[key]
    for n in d['No_cycles']:
        new_key = f'{key}_n={n}' # Creating lists of results for selected cycles of interest named ff_lat_berm_n=10 for example
        n_cycles_vals = [sublist[n-1] if len(sublist) > n-1 else None for sublist in list_of_lists]
        list_results[new_key] = n_cycles_vals
        to_fit_and_plot.append(new_key)
        
Output_dist = compiled[0]['inputs']['Output_dist']
Model_fct = compiled[0]['inputs']['Model_fct']
Common.process_results(list_results, Output_dist, to_fit_and_plot, results_path, Model_fct)

Common.process_per_cycle(list_results, cyclic_keys, results_path, Model_fct)

os.chdir(parent_dir)