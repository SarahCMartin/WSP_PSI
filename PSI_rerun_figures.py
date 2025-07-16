# PSI_resultsonly

"""This script runs loads results from previous PSI Monte Carlo simulations 
for re-processing / additional processing."""

###########################################################################
# Housekeeping
import os
directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(directory) # finding folder where this script is saved and setting as the working directory - if all functions are saved in the same folder this will prevent any directory errors
d = {} # empty dictionary to take input parameters to add to PSI_case object in the initiation step

###########################################################################
# Loading inputs and results
import Common
file_name = Common.import_json()

import json

if file_name: # Only save if user selected a file name
    with open(file_name, 'r') as file:
        results = json.load(file)
else:
    print("Results file not selected")

###########################################################################
# Select location to save figures from the new run
from pathlib import Path
parent_dir = Path(Common.select_folder())

if parent_dir.name.lower() == "results":
    results_path = parent_dir
else:
    results_path = parent_dir / 'Results'
    if not results_path.exists():
        results_path.mkdir()

###########################################################################
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

###########################################################################
# Plotting results and fitting distributions to them
to_fit_and_plot = ['z_aslaid', 'z_hydro', 'z_res', 'ff_lat_brk_UD', 'ff_lat_brk_D', 'ff_lat_res_UD', 'ff_lat_res_D', 'ff_ax_UD', 'ff_ax_D']
Common.process_results(list_results, results[0]['inputs']['Output_dist'], to_fit_and_plot, results_path)