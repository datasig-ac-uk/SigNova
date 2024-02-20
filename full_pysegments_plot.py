import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import plot_lib


insfile0 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset0.npy'
insfile1 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset1.npy'
insfile2 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset2.npy'
insfile3 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Datase3.npy'
insfile4 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Datase4.npy'
insfile5 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset5.npy'
insfile6 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset6.npy'  
insfile7 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset7.npy'
insfile8 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset8.npy'



outdir = 'Pysegments_plots/plots'

with open(insfile0, 'rb') as w:
    z = np.load(w)

with open(insfile1, 'rb') as f:
    a = np.load(f)

with open(insfile2, 'rb') as g:
    b = np.load(g)

with open(insfile3, 'rb') as h:
    c = np.load(h)

with open(insfile4, 'rb') as j:
    d = np.load(j)

with open(insfile5, 'rb') as k:
    e = np.load(k)

with open(insfile6, 'rb') as l:
    o = np.load(l)

with open(insfile7, 'rb') as m:
    p = np.load(m)

with open(insfile8, 'rb') as n:
    q = np.load(n)
#
#
tog = np.vstack((z,a,b,c,d,e,o,p,q))

xticks = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
xticklabels = [167, 170, 173, 177, 180, 183, 186, 189, 193, 196]

array = tog[:,:]
array_t = np.transpose(array)
print('Array', array.shape)
print('Array T', array_t.shape)
y_ticks = xticks
y_ticklabels = xticklabels
x_ticks = np.arange(0, 450, 50)
x_ticklabels = np.arange(0, 450, 50)
cmap = colors.ListedColormap(['navy', 'yellow'])
bounds=[0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
plot_lib.image_plot(fig, ax, array_t,
                    title='Pysegments', xticks=x_ticks,
                    xticklabels=x_ticklabels, yticks = y_ticks, yticklabels = y_ticklabels, mask_color='yellow', xlabel='Time (2s)', ylabel='Frequency (MHz)', cbar=False, two_colors=True, cbar_ticks=[0, 0.5, 1], cmap=cmap, norm=norm, boundaries=bounds, vmin=0, vmax=1, cbar_ticks_labels=['Non-Flagged', '', 'Flagged'], cbar_label='', xline=True, xlineticks=[50,100, 150, 200, 250, 300, 350, 400])
prefix = '%s/Pysegments_Fullplot_PerFrCh_GenExtreme_Alpha0005_Dataset' % (outdir)

fig.savefig('%s.pdf' % prefix)
plt.show()
