import numpy as np
import scipy
import matplotlib.pyplot as plt

x = (2, 3.8, 8)
f = lambda x,mu,sigma: scipy.stats.norm(mu,sigma).cdf(x)

data = (0.05, 0.5, 0.95)
mu,sigma = scipy.optimize.curve_fit(f,x,data)[0]
print(mu,sigma)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(x, data)
n = np.linspace(0,10,100)
ax[0].plot(n, scipy.stats.norm(mu,sigma).cdf(n),'r-', lw=5, alpha=0.6)

ax[1].scatter(x, scipy.stats.norm(mu,sigma).pdf(x))
ax[1].plot(n, scipy.stats.norm(mu,sigma).pdf(n),'r-', lw=5, alpha=0.6)
plt.show()