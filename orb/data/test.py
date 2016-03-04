import numpy as np
import scipy.interpolate
wl = list()
trans = list()
with open('optics_SN2.orb') as f:
    for line in f:
        if '#' not in line and len(line) > 2:
            line = np.array(line.strip().split(), dtype=float)
            wl.append(line[0])
            trans.append(line[1])

sp = scipy.interpolate.UnivariateSpline(wl[::-1], trans[::-1], s=0, k=1)



x = np.linspace(min(wl), max(wl), 2000)
trans = sp(x)
x = x[::-1]
trans = trans[::-1]

with open('optics_SN2.orb.ok', 'w') as f:
    for i in range(len(x)):
        f.write('{} {}\n'.format(
            x[i], trans[i]))

            
