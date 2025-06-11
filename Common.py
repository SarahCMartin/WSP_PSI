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
        start_index = input_data_str[input_data_str[0] == start_heading].index[0] + 2
    except IndexError:
        raise ValueError(f"Start heading '{start_heading}' not found in column A")

    # Find relevant column indices based on the column headings provided
    col_indices = []
    for heading in col_headings:
        try:
            col_index = input_data_str.columns[(input_data_str == heading).any()].tolist()[0]
            col_indices.append(col_index)
        except IndexError:
            raise ValueError(f"Column heading '{heading}' not found in input data")

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
