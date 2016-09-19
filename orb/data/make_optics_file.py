import numpy as np
import scipy.interpolate

lmax = 484
lmin = 389

t = np.zeros(2000, dtype=float)
w = np.linspace(200, 1000, 2000)
x = np.arange(2000, dtype=float)
xf = scipy.interpolate.UnivariateSpline(w, x, s=0, k=1)
lminp, lmaxp = xf((lmin, lmax))
t[int(lminp):int(lmaxp)] = 0.8

w = w[::-1]
t = t[::-1]


with open('optics_C1.orb', 'w') as f:
    for i in range(len(x)):
        f.write('{} {}\n'.format(
            w[i], t[i]))


