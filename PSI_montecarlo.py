# PSI_montecarlo

"""This script runs Pipe-Soil Interaction Calculations for a defined number
of randomly generated combinations of parameters with input distributions
for each parameter. It is the alternative to PSI_inputs which runs the 
PSI for a single parameter set."""

###########################################################################
# Housekeeping
import os
directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(directory) # finding folder where this script is saved and setting as the working directory - if all functions are saved in the same folder this will prevent any directory errors
d = {} # empty dictionary to take input parameters to add to PSI_case object in the initiation step

###########################################################################
# Import inputs from excel file 'PSI_Inputs.xlsx' and select location to save results
import Common
(input_data, input_data_str, file_path) = Common.import_excel('Inputs')

from pathlib import Path
parent_dir = Path(Common.select_folder())
if parent_dir.name.lower() == "results":
    results_path = parent_dir
else:
    results_path = parent_dir / 'Results'
    if not results_path.exists():
        results_path.mkdir()

import time
start = time.time() # Start timing after selection of the excel file to accurately reflect the monte carlo time which could be optimised

###########################################################################
# Read Monte Carlo and Model Selection Parameters from Excel input file
var_names = ['No_rolls', 'Emb_aslaid_model', 'Emb_hydro_model', 'Lat_brk_suction', 'Lat_res_suction', 'Emb_res_model', 'Cyc_model', 'N50', 'su_profile', 'z_su_inv', 'Output_dist', 'Model_fct']
d = {name: Common.find_var_value(input_data, input_data_str, name) for name in var_names} # dictionary containing constant parameters

# Handle special logic for z_su_inv
if d['su_profile'] != 1:
    d['z_su_inv'] = []

# Adding separately model inputs which can be in the form of a list
list_names = ['Lat_brk_model', 'Lat_brk_weighting', 'Lat_res_model', 'Lat_res_weighting', 'Ax_model', 'No_cycles']
d.update({name: Common.find_var_list(input_data, input_data_str, name) for name in list_names})

###########################################################################
# Read Pipe, Soil and Interface Parameters from Excel input file
column_headings = ['Parameter', 'LE', 'BE', 'HE', 'Min', 'Distribution to Fit']
data_type = [str, float, float, float, float, str] # corresponding to the headings in 'column headings'
start_heading = 'Inputs Requiring Probabalistic Distribution Fitting' # below which the table to read from starts (including column headings)

p = Common.read_columns(input_data, input_data_str, column_headings, data_type, start_heading) # dictionary containing variables which need to be allocated statistically
p = Common.restructure_col_to_row(p, column_headings)

###########################################################################
# Generate values for each dice roll according to best fit probability distributions for pipe, soil and interface parameters
random_inputs, _ = Common.generate_rolls(p, d['No_rolls'], results_path)

###########################################################################
# Import correlations from excel then apply to the randomly generated variables
(corr_data, corr_data_str, _) = Common.import_excel('Correlations', file_path)
corr_headings = ['Parameter', 'Base Variable', 'Correlation Type', 'Correlation Index']
corr_data_type = [str, str, str, float] # corresponding to the headings in 'column headings'

corr = Common.read_columns(corr_data, corr_data_str, corr_headings, corr_data_type, None) # dictionary containing variables which need to be allocated statistically
corr = Common.restructure_col_to_row(corr, corr_headings)
import PSI_correlate
random_inputs = PSI_correlate.apply_correlations(corr, random_inputs, results_path)
random_inputs['z_ini'] = [float(x / 2) for x in random_inputs['D']]

###########################################################################
# Run PSI for prescribed number of parameter sets (rolls)
from PSI_class import PSI
import PSI_soils
results = []

for i in range(d['No_rolls']):
    # Compile model parameters (fixed variables) with a set of pipe, soil and interface parameters for that roll (random probabilistic variables)
    current_random = {k: v[i] for k, v in random_inputs.items()}
    input_dict = {**d, **current_random}

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

suffix_map = {
    2: ["LE", "HE"],
    3: ["LE", "BE", "HE"],
}

for key, value in list(list_results.items()):
    if (
        isinstance(value, list)
        and all(isinstance(i, list) for i in value)
        and len(value) > 0
        and all(len(i) == len(value[0]) for i in value) # consistent inner length
        and len(value[0]) in suffix_map                 # only process 2 or 3 values because these are known to be LE-HE or LE-BE-HE, any other length >1 would be erroneous
    ):
        # Transpose the list of lists
        transposed = list(map(list, zip(*value)))

        # Assign each column to a new key with a suffix
        for i, suffix in enumerate(suffix_map[len(value[0])]):
            list_results[f"{key}_{suffix}"] = transposed[i]

        # Remove the original unsplit key
        del list_results[key]

end = time.time()
#print(list_results['z_aslaid'])
print(f"Elapsed time: {end - start:.4f} seconds")

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
to_fit_and_plot = ['z_aslaid', 'z_hydro', 'z_res', 'ff_lat_brk_UD', 'ff_lat_brk_D', 'ff_lat_res_UD', 'ff_lat_res_D', 'ff_ax_UD', 'ff_ax_D']
Common.process_results(list_results, d['Output_dist'], to_fit_and_plot, results_path, d['Model_fct'])

os.chdir(parent_dir)