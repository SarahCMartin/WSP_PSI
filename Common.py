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


def import_excel():
    import pandas as pd
    from tkinter import Tk, filedialog

    # Opening then hiding the GUI window to be able to do the file-picking in a window next
    root = Tk()
    root.withdraw

    # Opening the file dialog to select and excel input file
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")], title="Select input file")

    # Importing the data from selected excel file - full and as strings for searching for variable names
    input_data = pd.read_excel(file_path, sheet_name='Inputs', header=None)
    input_data_str = input_data.astype(str)

    return (input_data, input_data_str)


def find_var_value(input_data, input_data_str, var_name):
    try:
        row_index = input_data_str[input_data_str[0] == var_name].index[0]
        var_value = input_data.loc[row_index, 1] # value in column B where column A matches the variable name
        return var_value
    
    except IndexError:
        raise ValueError(f"Variable '{var_name}' not found in column A")
    

def read_columns(input_data, input_data_str, col_headings, data_type, start_heading):
    d = {} # Initialising empty dictionary
    type_map = dict(zip(col_headings, data_type)) # mapping data types with column headings for later when assigning into output dictionary

    # Find start of parameter block based on provided heading
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

def generate_rolls(d, No_rolls):
    rolls = {}
    import numpy as np
    import Distributions
    from scipy.stats import norm
    from scipy.stats import uniform
    from scipy.stats import lognorm
    from scipy.stats import weibull_min
    from scipy.stats import weibull_max
    from scipy.stats import gamma
    from scipy.stats import rayleigh

    for name, info in d.items():
        LE = info["LE"]
        BE = info["BE"]
        HE = info["HE"]
        dist = info["Distribution to Fit"].lower()

        if dist == 'uniform':
            loc, scale = Distributions.fit_uniform(LE, BE, HE)
            # For case where LE=BE=HE, manually making this the value as statistical distribution will otherwise make it vary marginally
            if np.isclose(LE, BE) and np.isclose(BE, HE):
                rolls[name] = np.full(No_rolls, BE)
            else:
                rolls[name] = uniform.rvs(loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0, 0.5, 1], uniform, (loc, scale), samples=rolls[name], dist_name='Uniform')

        elif dist == 'normal':
            mu, sigma = Distributions.fit_normal_to_percentiles(LE, BE, HE)
            rolls[name] = norm.rvs(loc=mu, scale=sigma, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], norm, (mu, sigma), samples=rolls[name], dist_name='Normal')

        elif dist == 'log-normal':
            s, loc, scale = Distributions.fit_lognormal_to_percentiles(LE, BE, HE)
            rolls[name] = lognorm.rvs(s=s, loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], lognorm, (s, loc, scale), samples=rolls[name], dist_name='Log-normal')

        elif dist == 'weibull':
            c, loc, scale = Distributions.fit_weibull_to_percentiles(LE, BE, HE)
            rolls[name] = weibull_min.rvs(c=c, loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], weibull_min, (c, loc, scale), samples=rolls[name], dist_name='Weibull')

        elif dist == 'reverse-weibull':
            c, loc, scale = Distributions.fit_reverseweibull_to_percentiles(LE, BE, HE)
            rolls[name] = weibull_max.rvs(c=c, loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], weibull_max, (c, loc, scale), samples=rolls[name], dist_name='Reverse-weibull')

        elif dist == 'gamma':
            a, loc, scale = Distributions.fit_gamma_to_percentiles(LE, BE, HE)
            rolls[name] = gamma.rvs(a=a, loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], gamma, (a, loc, scale), samples=rolls[name], dist_name='Gamma')

        elif dist == 'rayleigh':
            loc, scale = Distributions.fit_rayleigh_to_percentiles(LE, BE, HE)
            rolls[name] = rayleigh.rvs(loc=loc, scale=scale, size=No_rolls)
            Distributions.plot_distribution_fit([LE, BE, HE], [0.05, 0.5, 0.95], rayleigh, (loc, scale), samples=rolls[name], dist_name='Rayleigh')

    return rolls
