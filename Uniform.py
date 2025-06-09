import numpy as np
import scipy

n=int(n)
f = lambda x,loc,scale: scipy.stats.uniform(loc,scale).cdf(x)

# y = (0, 1)
loc, scale = scipy.optimize.curve_fit(f,x,y)[0]
r = scipy.stats.uniform.rvs(loc,scale, size=n)

plotmax = x[-1]*1.5
plotmin = x[0]-x[-1]*0.5
x1 = np.linspace(plotmin,plotmax,100)
CDF = scipy.stats.uniform(loc,scale).cdf(x1)
PDF = scipy.stats.uniform(loc,scale).pdf(x1)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 2)
# ax[0].scatter(x, y)
# ax[0].plot(x1, CDF,'r-', lw=5, alpha=0.6)

# ax[1].scatter(x, scipy.stats.uniform(loc,scale).pdf(x))
# ax[1].plot(x1, PDF,'r-', lw=5, alpha=0.6)
# plt.show()