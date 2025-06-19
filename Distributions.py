import numpy as np
from scipy.stats import uniform, norm, lognorm, weibull_min, weibull_max, gamma, rayleigh
from scipy.optimize import minimize, curve_fit


def fit_uniform(LE, BE, HE):
    """Fit a uniform distribution to match the min, mean and max.
    Returns estimated loc and scale"""

    # Target percentiles (0%, 50%, 100%)
    target_percentiles = [LE, BE, HE]
    probs = [0, 0.5, 1] # different for normal than for others as this is only used where actual min and max are better defined, e.g. pipe weight

    # Objective function: sum of squared differences between target and candidate percentiles
    def objective(params):
        loc, scale = params
        if scale <= 0:
            return np.inf
        predicted = uniform.ppf(probs, loc=loc, scale=scale)
        return np.sum((np.array(predicted) - np.array(target_percentiles))**2)
    
    initial_guess = [LE, HE - LE]
    result = minimize(objective, initial_guess, method='Nelder-Mead')

    loc_opt, scale_opt = result.x
    return loc_opt, scale_opt


def fit_normal_to_percentiles(LE, BE, HE):
    """Fit a normal distribution to match the 5th, 50th, and 95th percentiles.
    Returns estimated mu and sigma."""

    # Target percentiles (5%, 50%, 95%)
    target_percentiles = [LE, BE, HE]
    probs = [0.05, 0.5, 0.95]

    # Objective function: sum of squared differences between target and candidate percentiles
    def objective(params):
        mu, sigma = params
        predicted = norm.ppf(probs, loc=mu, scale=sigma)
        return np.sum((np.array(predicted) - np.array(target_percentiles))**2)

    # Initial guess
    initial_guess = [BE, (HE - LE) / 4]

    # Minimize the objective
    result = minimize(objective, initial_guess, bounds=[(None, None), (1e-6, None)], method='Nelder-Mead')
    mu_opt, sigma_opt = result.x

    return mu_opt, sigma_opt


def fit_lognormal_to_percentiles(LE, BE, HE):
    """Fit a log-normal distribution to match the 5th, 50th, and 95th percentiles.
    Returns estimated s, loc and scale."""

    # Target percentiles (5%, 50%, 95%)
    LE = max(LE, 1e-6)
    BE = max(BE, 1e-6)
    HE = max(HE, 1e-6)
    target_percentiles = [LE, BE, HE]
    probs = [0.05, 0.5, 0.95]

    # Objective function: sum of squared differences between target and candidate percentiles
    def objective(params):
        s, loc, scale = params
        predicted = lognorm.ppf(probs, s=s, loc=loc, scale=scale)
        return np.sum((np.array(predicted) - np.array(target_percentiles))**2)

    # Initial guess
    sigma_guess = (np.log(HE) - np.log(LE))/4
    mu_guess = np.log(BE)
    initial_guess = [sigma_guess, 0, np.exp(mu_guess)]

    # Minimize the objective
    result = minimize(objective, initial_guess, bounds=[(1e-6, None), (0, None), (1e-6, None)], method='L-BFGS-B')
    s_opt, loc_opt, scale_opt = result.x

    return s_opt, loc_opt, scale_opt


def fit_weibull_to_percentiles(LE, BE, HE):
    """Fit a weibull distribution to match the 5th, 50th, and 95th percentiles.
    Returns estimated c, loc and scale."""

    # Target percentiles (5%, 50%, 95%)
    target_percentiles = [LE, BE, HE]
    probs = [0.05, 0.5, 0.95]

    # Objective function: sum of squared differences between target and candidate percentiles
    def objective(params):
        c, loc, scale = params
        predicted = weibull_min.ppf(probs, c=c, loc=loc, scale=scale)
        return np.sum((np.array(predicted) - np.array(target_percentiles))**2)

    # Initial guess
    initial_guess = [1.5, 0, HE-LE] # initial guesses from chatGPT, adjust if needed

    # Minimize the objective
    result = minimize(objective, initial_guess, bounds=[(1e-3, None), (0, None), (1e-6, None)], method='Nelder-Mead') # limits from chatGPT, adjust if needed
    c_opt, loc_opt, scale_opt = result.x

    return c_opt, loc_opt, scale_opt


def fit_reverseweibull_to_percentiles(LE, BE, HE):
    """Fit a reverse-weibull distribution to match the 5th, 50th, and 95th percentiles.
    Returns estimated c, loc and scale."""

    # Target percentiles (5%, 50%, 95%)
    target_percentiles = [LE, BE, HE]
    probs = [0.05, 0.5, 0.95]

    # Objective function: sum of squared differences between target and candidate percentiles
    def objective(params):
        c, loc, scale = params
        predicted = weibull_max.ppf(probs, c=c, loc=loc, scale=scale)
        return np.sum((np.array(predicted) - np.array(target_percentiles))**2)

    # Initial guess
    initial_guess = [1.5, HE, HE-LE] # initial guesses from chatGPT, adjust if needed

    # Minimize the objective
    result = minimize(objective, initial_guess, bounds=[(1e-3, None), (0, None), (1e-6, None)], method='Nelder-Mead') # limits from chatGPT, adjust if needed
    c_opt, loc_opt, scale_opt = result.x

    return c_opt, loc_opt, scale_opt


def fit_gamma_to_percentiles(LE, BE, HE):
    """Fit a gamma distribution to match the 5th, 50th, and 95th percentiles.
    Returns estimated a, loc and scale."""

    # Target percentiles (5%, 50%, 95%)
    LE = max(LE, 1e-6)
    BE = max(BE, 1e-6)
    HE = max(HE, 1e-6)
    target_percentiles = [LE, BE, HE]
    probs = [0.05, 0.5, 0.95]

    # Objective function: sum of squared differences between target and candidate percentiles
    def objective(params):
        a, loc, scale = params
        predicted = gamma.ppf(probs, a=a, loc=loc, scale=scale)
        return np.sum((np.array(predicted) - np.array(target_percentiles))**2)

    # Initial guess
    shape_guess = 2
    loc_guess = 0
    scale_guess = (HE-LE)/shape_guess
    initial_guess = [shape_guess, loc_guess, scale_guess] # initial guesses from chatGPT, adjust if needed

    # Minimize the objective
    result = minimize(objective, initial_guess, bounds=[(1e-3, None), (0, None), (1e-6, None)], method='L-BFGS-B') # limits from chatGPT, adjust if needed
    a_opt, loc_opt, scale_opt = result.x

    return a_opt, loc_opt, scale_opt


def fit_rayleigh_to_percentiles(LE, BE, HE):
    """Fit a rayleigh distribution to match the 5th, 50th, and 95th percentiles.
    Returns estimated loc and scale."""

    # Target percentiles (5%, 50%, 95%)
    LE = max(LE, 1e-6)
    BE = max(BE, 1e-6)
    HE = max(HE, 1e-6)
    target_percentiles = [LE, BE, HE]
    probs = [0.05, 0.5, 0.95]

    # Objective function: sum of squared differences between target and candidate percentiles
    def objective(params):
        loc, scale = params
        predicted = rayleigh.ppf(probs, loc=loc, scale=scale)
        return np.sum((np.array(predicted) - np.array(target_percentiles))**2)

    # Initial guess
    loc_guess = 0
    scale_guess = np.sqrt((BE - loc_guess)**2 / (4 - np.pi))
    initial_guess = [loc_guess, scale_guess] # initial guesses from chatGPT (scale guess apparently derived from Rayleigh mean), adjust if needed

    # Minimize the objective
    result = minimize(objective, initial_guess, bounds=[(0, None), (1e-6, None)], method='L-BFGS-B') # limits from chatGPT, adjust if needed
    loc_opt, scale_opt = result.x

    return loc_opt, scale_opt


def fit_distribution_cdf(data, dist_name):
    """Fit a distribution CDF to empirical data by minimizing squared error of CDF values.
    Enforces positive shape/scale and loc >= 0. Returns optimised parameters."""

    # Get distribution object and parameter names
    dist, param_names = dist_map(dist_name, return_params=True)

    # Sort data and compute empirical CDF values
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)

    # For positive-support dists, filter data <= 0
    positive_support = ['Log-normal', 'Weibull', 'Reverse-weibull', 'Gamma', 'Rayleigh']
    if dist_name in positive_support:
        mask = sorted_data > 0
        sorted_data = sorted_data[mask]
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Get initial guess using scipy.stats fit (with loc fixed to 0 if positive support)
    try:
        if 'loc' in param_names and dist_name in positive_support:
            init_params = dist.fit(sorted_data, floc=0)
        else:
            init_params = dist.fit(sorted_data)
    except Exception:
        # fallback initial guess if fit fails
        init_params = np.ones(len(param_names))

    # Define bounds: loc >=0; shape/scale >0; others unbounded
    bounds = []
    for param in param_names:
        if param == 'loc':
            bounds.append((0, None))  # loc ≥ 0
        elif param in ['scale', 's', 'c', 'a']:
            bounds.append((1e-6, None))  # shape/scale > 0
        else:
            bounds.append((None, None))  # no bounds

    # Objective function: sum squared error between empirical CDF and model CDF
    def objective(params):
        try:
            cdf_vals = dist.cdf(sorted_data, *params)
            return np.sum((cdf_vals - ecdf) ** 2)
        except Exception:
            return 1e10  # penalty for invalid params

    result = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B')

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x


def plot_distribution_fit(x_percentiles, percentiles, dist, params, samples=None, param_name=None, dist_name=''):
    """Plots the fitted distribution's CDF and PDF along with input percentiles, inputs:
        x_percentiles (list or np.array): The x-values at given percentiles (e.g., [LE, BE, HE]).
        percentiles (list or np.array): Percentiles corresponding to x_percentiles (e.g., [0.05, 0.5, 0.95]).
        dist (scipy.stats distribution): A scipy.stats distribution object (e.g., scipy.stats.norm).
        params (tuple): Parameters for the distribution (e.g., (mu, sigma)).
        samples (list): Randomly generated values based on the distribution.
        dist_name (str): Optional name for labeling plots."""

    import matplotlib.pyplot as plt
    from statsmodels.distributions.empirical_distribution import ECDF
    import PSI_resultformat

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # x = np.array(x_percentiles)
    # probs = np.array(percentiles)

    # Create x-range for plotting
    if x_percentiles is not None: # plotting distribution for inputs
        x = np.array(x_percentiles)
        probs = np.array(percentiles)
        x_min, x_max = get_plot_bounds(x)
    else: # plotting distribution for outputs
        x = None
        probs = None
        x_min, x_max = get_plot_bounds(samples)

    x_vals = np.linspace(x_min, x_max, 1000)

    # Check for degenerate case (all percentiles equal)
    if dist == uniform and np.allclose(x, x[0]):
        # Compute CDF and PDF
        cdf = np.where(x_vals >= x[0], 1.0, 0.0)
        pdf = np.zeros_like(x_vals)
        spike_width = (x_vals[-1] - x_vals[0]) * 0.001  # spike allocated width of 0.1 % of x-range to be large enough to see
        bin_width = spike_width*20 # making bin wider than spike so it's visible
        pdf[(x_vals>=x[0] - spike_width) & (x_vals<=x[0] + spike_width)] = 1/bin_width # setting height arbitrarily based on histogram so they match visually; theoretically the spike would be infinite
        pdf[np.abs(x_vals - x[0]) < 1e-6] = 1.0  # visual spike

        # Plotting
        ax[0].plot(x_vals, cdf, 'r-', label=f'{dist_name.title()} CDF')
        ax[1].plot(x_vals, pdf, 'r-', label=f'{dist_name.title()} PDF')
        ax[0].axvline(x[0], color='blue', linestyle='--', label='LE/BE/HE')
        ax[1].axvline(x[0], color='blue', linestyle='--', label='LE/BE/HE')

        # Adjusting for degenerative case only not to show numbers on the y-axis where they have been arbirtraily adjusted based on arbirary spike/bar widths (theoretically would be infinite)
        ax[1].tick_params(axis='y', left=False, labelleft=False)
        ax[1].set_ylabel("Density (arbitrary scale)")

        # Adding histogram of the generated values to visually confirm they correspond to the function
        if samples is not None and len(samples) > 0:
            bins=[min(samples)-bin_width/2, max(samples)+bin_width/2]
            # Setting width and height arbirtarily for visualisation purposes
            ax[1].hist(samples, bins=bins, alpha=0.5, density=True, color='gray', label='Histogram of Random Values')

    else:
        # Compute CDF and PDF
        fitted_dist = dist(*params)
        cdf = fitted_dist.cdf(x_vals)
        pdf = fitted_dist.pdf(x_vals)

        # Plotting
        ax[0].plot(x_vals, cdf, 'r-', label=f'{dist_name.title()} CDF')
        ax[1].plot(x_vals, pdf, 'r-', label=f'{dist_name.title()} PDF')
        
        # Adding input percentiles for input fitting, not relevant for results fitting
        if x is not None and len(x) > 0:
            ax[0].scatter(x, probs, color='blue', label='Input Percentiles')
            ax[1].scatter(x, fitted_dist.pdf(x), color='blue', label='Input Percentiles')

        # Defining PDF axis label here as it is different for the degenerative case vs others
        ax[1].set_ylabel("Density")

        # Adding histogram of the randomly generated inputs / calculated results from those inputs to visually confirm they correspond to the function
        if samples is not None and len(samples) > 0:
            ax[1].hist(samples, bins=50, density=True, alpha=0.5, color='gray', label='Histogram of Actual Values')

    # Set y-limits for plots
    ax[0].set_ylim(0, 1.05)  # CDF always between 0–1
    ax[1].set_ylim(0, 1.1 * pdf.max() if np.any(pdf > 0) else 1)  # PDF adaptive

    # Setting other aesthetic details for plots
    ax[0].set_title("Cumulative Distribution Function")
    #ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Probability")
    ax[0].set_xlim([max(0,x_vals[0]), x_vals[-1]])
    ax[0].legend()

    ax[1].set_title("Probability Density Function")
    #ax[1].set_xlabel("Value")
    ax[1].set_xlim([max(0,x_vals[0]), x_vals[-1]])
    ax[1].legend()
    
    PSI_resultformat.hard_coded_headings(fig, ax, param_name)

    #plt.tight_layout()
    plt.show()

def get_plot_bounds(x, pad_fraction=0.5, default_pad=0.05):
    """Returns (x_min, x_max) for plotting. Handles cases where all x are equal."""
    x_min = np.min(x)
    x_max = np.max(x)
    if np.isclose(x_min, x_max):
        pad = default_pad * abs(x_min) if x_min != 0 else default_pad
        return x_min - pad, x_max + pad
    else:
        pad = (x_max - x_min)*pad_fraction
        plot_min = x_min-pad
        plot_max = x_max+pad
        return plot_min, plot_max
    

def dist_map(dist_name, return_params=False):
    # Map distribution names and parameters to scipy.stats objects
    dist_map = {
        "Uniform": (uniform, ['loc', 'scale']),
        "Normal": (norm, ['loc', 'scale']),
        "Log-normal": (lognorm, ['s', 'loc', 'scale']),
        "Weibull": (weibull_min, ['c', 'loc', 'scale']),
        "Reverse-weibull": (weibull_max, ['c', 'loc', 'scale']),
        "Gamma": (gamma, ['a', 'loc', 'scale']),
        "Rayleigh": (rayleigh, ['loc', 'scale'])
    }
    entry = dist_map.get(dist_name)

    if entry is None:
        raise ValueError(f"Unsupported distribution: {dist_name}")
    if return_params:
        return entry # returns dist_obj and param_names
    else:
        return entry[0] # returns dist_obj only