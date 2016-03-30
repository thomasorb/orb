import numpy as np
import scipy.interpolate

lmax = 386.006200207
lmin = 362.218740625

t = np.zeros(2000, dtype=float)
w = np.linspace(200, 1000, 2000)
x = np.arange(2000, dtype=float)
xf = scipy.interpolate.UnivariateSpline(w, x, s=0, k=1)
lminp, lmaxp = xf((lmin, lmax))
t[int(lminp):int(lmaxp)] = 0.7661

w = w[::-1]
t = t[::-1]


with open('optics_SN1.orb', 'w') as f:
    for i in range(len(x)):
        f.write('{} {}\n'.format(
            w[i], t[i]))


