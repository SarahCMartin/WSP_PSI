# def float_range(x, y, jump):
#     list = [x]
#     while (x + jump) < y:
#         x += jump
#         list += [x]
#     list += [y]
#     return list


def float_range(x, y, jump):
    import numpy
    no_steps = round((y-x)/jump)
    temp_array = numpy.linspace(x, y, no_steps+1)
    list = temp_array.tolist()
    return list


def round_up(x, dp=0):
    import math
    multiplier = 10*dp
    return math.ceil(x*multiplier)/multiplier


def linear_extrapolate(x_new, x_known, y_known):
    from scipy import interpolate
    temp = interpolate.interp1d(x_known, y_known, fill_value = "extrapolate")
    return temp(x_new)


def format_sigfig(val, sigfigs):
    import math
    if val == 0:
        return f"{0:.{sigfigs - 1}f}"

    magnitude = math.floor(math.log10(abs(val)))
    decimals = sigfigs - magnitude - 1

    if decimals > 0:
        output = f"{val:.{decimals}f}"

        # Pad with trailing zeros if needed
        int_part, frac_part = output.split('.')
        if len(frac_part) < decimals:
            frac_part = frac_part.ljust(decimals, '0')
        output = f"{int_part}.{frac_part}"
    else:
        # Round to nearest significant digit without decimal places
        rounding_factor = 10 ** abs(decimals)
        rounded_val = round(val / rounding_factor) * rounding_factor
        output = f"{int(rounded_val)}"

    return output


def get_filename():
    from tkinter import Tk, filedialog

    # Opening then hiding the GUI window to be able to do the file-picking in a window next
    root = Tk()
    root.withdraw

    # Ask user for a filename to save
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Save your dictionary as a JSON file"
    )
    return file_path


def select_folder():
    from tkinter import Tk, filedialog

    # Opening then hiding the GUI window to be able to do the file-picking in a window next
    root = Tk()
    root.withdraw

    # Ask user for a filename to save
    file_path = filedialog.askdirectory(
        title="Select a location to create your results folder"
    )
    return file_path


def import_excel(sheet_name, file_path=None):
    import pandas as pd
    from tkinter import Tk, filedialog
    import warnings

    # Supressing a warning that happens everytime because the Excel sheet has drop downs for the distribution types
    warnings.filterwarnings("ignore", message="Data Validation extension is not supported and will be removed")

    if file_path == None: # used the first time when the file needs to be selected
        # Opening then hiding the GUI window to be able to do the file-picking in a window next
        root = Tk()
        root.withdraw

        # Opening the file dialog to select and excel input file
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")], title="Select input file")

    # Importing the data from selected excel file - full and as strings for searching for variable names
    input_data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    input_data_str = input_data.astype(str)

    return (input_data, input_data_str, file_path)


def import_json():
    import pandas as pd
    from tkinter import Tk, filedialog

    # Opening then hiding the GUI window to be able to do the file-picking in a window next
    root = Tk()
    root.withdraw

    # Opening the file dialog to select and json input file
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], title="Select input file")

    return file_path


def find_var_value(input_data, input_data_str, var_name):
    try:
        row_index = input_data_str[input_data_str[0] == var_name].index[0]
        var_value = input_data.loc[row_index, 1] # value in column B where column A matches the variable name
        return var_value
    
    except IndexError:
        raise ValueError(f"Variable '{var_name}' not found in column A")
    

def find_var_list(input_data, input_data_str, list_name):
    import pandas as pd
    try:
        row_index = input_data_str[input_data_str[0] == list_name].index[0]
        row = input_data.loc[row_index, 1:]
        values = []
        for val in row:
            if pd.isna(val) or str(val).strip() == "":
                break
            values.append(val)
        return values
    
    except IndexError:
        raise ValueError(f"Variable '{list_name}' not found in column A")


def read_columns(input_data, input_data_str, col_headings, data_type, start_heading):
    d = {} # Initialising empty dictionary
    type_map = dict(zip(col_headings, data_type)) # mapping data types with column headings for later when assigning into output dictionary

    # Find start of parameter block based on provided heading
    if start_heading == None: # Column heading are in the top row
        start_index = 1
    else:
        try:
            start_index = input_data_str[input_data_str.iloc[:,0].str.contains(start_heading, na=False)].index[0] + 2
        except IndexError:
            raise ValueError(f"Start heading '{start_heading}' not found in column A")

    # Find relevant column indices based on the column headings provided
    header_row = input_data_str.iloc[start_index - 1, :]
    col_indices = []
    for heading in col_headings:
        matches = header_row[header_row.str.contains(heading, case=True, regex=False, na=False)]
        if matches.empty:
            raise ValueError(f"No column heading found containing '{heading}'")
        col_indices.append(matches.index[0])

    # Compiling relevant rows and columns
    param_block = input_data.loc[start_index:, col_indices]
    param_block.columns = col_headings
    param_block = param_block.reset_index(drop=True)

    # Putting into dictionary format
    for heading in col_headings:
        dtype = type_map.get(heading, str)
        if dtype == float:
            d[heading] = param_block[heading].astype(float).to_numpy()
        else:
            d[heading] = param_block[heading].astype(dtype).tolist()
    
    return d


def restructure_col_to_row(d, column_headings):
    restructured_d = {}
    param_names = d[column_headings[0]]
    for i, name in enumerate(param_names):
        restructured_d[name] = {}
        for col in column_headings[1:]: # skips the first column as that was treated separately to name the rows
            restructured_d[name][col] = d[col][i]

    return restructured_d


def generate_rolls(d, No_rolls, results_path=None):
    rolls = {}
    fit_info = {}
    import numpy as np
    import Distributions
    from scipy.stats import uniform, truncnorm, lognorm, weibull_min, weibull_max, gamma, rayleigh

    for name, info in d.items():
        LE = info["LE"]
        BE = info["BE"]
        HE = info["HE"]
        Min = info["Min"]
        dist = info["Distribution to Fit"].lower()

        if dist == 'uniform':
            loc, scale = Distributions.fit_uniform(LE, BE, HE) # No 'Min' for uniform distribution as LE is Min in this case to avoid having to manually calculate the 5% value where Min and Max are known
            fit_info[name] = [dist, (loc,scale)]
            # For case where LE=BE=HE, manually making this the value as statistical distribution will otherwise make it vary marginally
            if np.isclose(LE, BE) and np.isclose(BE, HE):
                rolls[name] = np.full(No_rolls, BE)
            else:
                rolls[name] = uniform.rvs(loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0, 0.5, 1], uniform, (loc, scale), samples=rolls[name], param_name=name, dist_name='Uniform', results_path=results_path, type='input')

        elif dist == 'normal':
            mu, sigma = Distributions.fit_normal_to_percentiles(LE, BE, HE, Min)
            fit_info[name] = [dist, (mu, sigma)]
            rolls[name] = truncnorm.rvs(loc=mu, scale=sigma, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], truncnorm, (mu, sigma), samples=rolls[name], param_name=name, dist_name='Normal', results_path=results_path, type='input')

        elif dist == 'log-normal':
            s, loc, scale = Distributions.fit_lognormal_to_percentiles(LE, BE, HE, Min)
            fit_info[name] = [dist, (s,loc,scale)]
            rolls[name] = lognorm.rvs(s=s, loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], lognorm, (s, loc, scale), samples=rolls[name], param_name=name, dist_name='Log-normal', results_path=results_path, type='input')

        elif dist == 'weibull':
            c, loc, scale = Distributions.fit_weibull_to_percentiles(LE, BE, HE, Min)
            fit_info[name] = [dist, (c,loc,scale)]
            rolls[name] = weibull_min.rvs(c=c, loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], weibull_min, (c, loc, scale), samples=rolls[name], param_name=name, dist_name='Weibull', results_path=results_path, type='input')

        # elif dist == 'reverse-weibull':
        #     c, loc, scale = Distributions.fit_reverseweibull_to_percentiles(LE, BE, HE, Min)
        #     rolls[name] = weibull_max.rvs(c=c, loc=loc, scale=scale, size=No_rolls)
        #     #Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], weibull_max, (c, loc, scale), samples=rolls[name], param_name=name, dist_name='Reverse-weibull', results_path=results_path)

        elif dist == 'gamma':
            a, loc, scale = Distributions.fit_gamma_to_percentiles(LE, BE, HE, Min)
            fit_info[name] = [dist, (a,loc,scale)]
            rolls[name] = gamma.rvs(a=a, loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], gamma, (a, loc, scale), samples=rolls[name], param_name=name, dist_name='Gamma', results_path=results_path, type='input')

        elif dist == 'rayleigh':
            loc, scale = Distributions.fit_rayleigh_to_percentiles(LE, BE, HE, Min)
            fit_info[name] = [dist, (loc,scale)]
            rolls[name] = rayleigh.rvs(loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], rayleigh, (loc, scale), samples=rolls[name], param_name=name, dist_name='Rayleigh', results_path=results_path, type='input')

        elif dist == 'automated fit':
            dist_name, params = Distributions.fit_best_dist_to_percentiles(LE, BE, HE, Min)
            fit_info[name] = [dist_name.lower(), params]
            dist_obj = Distributions.dist_map(dist_name)
            rolls[name] = dist_obj.rvs(*params, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], dist_obj, params, samples=rolls[name], param_name=name, dist_name=dist_name, results_path=results_path, type='input')

    return rolls, fit_info


def process_results(results, dist_str, chosen, output_folder=None, Model_fct=1):
    import numpy as np
    import Distributions

    # dist = Distributions.dist_map(dist_str)
    output_fit_params = {}
    percentiles = [0.05, 0.5, 0.95]

    for name, info in results.items():
        if name in chosen:
            if dist_str == 'Automated Fit':
                dist_str, output_fit_params[name] = Distributions.fit_best_distribution_cdf(info)
                dist = Distributions.dist_map(dist_str)

            else:
                dist = Distributions.dist_map(dist_str)
                output_fit_params[name] = Distributions.fit_distribution_cdf(info, dist_str)

            x_percentiles = dist.ppf(percentiles, *output_fit_params[name])

            if np.isnan(x_percentiles).any():
                print(f"No or insufficient information to fit distribtuion for {name}")

            else:
                if Model_fct != 1: # If model factor is 1, no change to the results therefore no step here
                    BE = x_percentiles[1]
                    new_info = (info - BE)*Model_fct + BE
                    info = new_info
                    output_fit_params[name] = Distributions.fit_distribution_cdf(info, dist_str) # updating curve fit - don't refit best type if automated fit option, apply the one already selected
                    x_percentiles = dist.ppf(percentiles, *output_fit_params[name]) # updating values of 5%, 50% (should be unchanged) and 95% for plotting

                Distributions.plot_distribution_fit(x_percentiles, percentiles, dist, output_fit_params[name], info, name, dist_str, output_folder, type='output')
                print(f"{name} - LE: {x_percentiles[0]}, BE: {x_percentiles[1]}, HE: {x_percentiles[2]}")

                return info