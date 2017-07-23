import numpy as np
import scipy.interpolate

with open('filter_C3.raw') as f:
    w = list()
    t = list()
    for line in f:
        if len(line) > 2:
            line = line.strip().split()
            w.append(float(line[0]))
            t.append(float(line[1]))

tf = scipy.interpolate.UnivariateSpline(w,t,k=1,s=0,ext=1)
ww = np.linspace(200,1000, 30000)

tt = tf(ww)[::-1]
ww = ww[::-1]

with open('filter_C3.new.orb', 'w') as f:
    for i in range(tt.shape[0]):
        f.write('{} {}\n'.format(ww[i], tt[i]))
    

import pylab as pl
pl.plot(ww,tt)
pl.show()
        
