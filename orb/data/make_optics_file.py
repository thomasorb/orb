import numpy as np
import scipy.interpolate

lmax = 685.882862762
lmin = 646.334877829

t = np.zeros(2000, dtype=float)
w = np.linspace(200, 1000, 2000)
x = np.arange(2000, dtype=float)
xf = scipy.interpolate.UnivariateSpline(w, x, s=0, k=1)
lminp, lmaxp = xf((lmin, lmax))
t[int(lminp):int(lmaxp)] = 0.7661

w = w[::-1]
t = t[::-1]


with open('optics_SN3.orb', 'w') as f:
    for i in range(len(x)):
        f.write('{} {}\n'.format(
            w[i], t[i]))


