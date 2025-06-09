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