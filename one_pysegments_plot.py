import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import plot_lib


insfile0 = 'npy/Pysegments_PerFrCh_GenExtreme_Alpha0005_Dataset0.npy'

outdir = 'Pysegments_plots/plots'

with open(insfile0, 'rb') as w:
    z = np.load(w)


xticks = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
xticklabels = [167, 170, 173, 177, 180, 183, 186, 189, 193, 196]

array_t = np.transpose(z)
print('Array', array.shape)
print('Array T', array_t.shape)

y_ticks = xticks
y_ticklabels = xticklabels
x_ticks = np.arange(0, z.shape[0], 10)
x_ticklabels = np.arange(0, z.shape[0], 10)
cmap = colors.ListedColormap(['navy', 'yellow'])
bounds=[0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
plot_lib.image_plot(fig, ax, array_t,
                    title='Pysegments', xticks=x_ticks,
                    xticklabels=x_ticklabels, yticks = y_ticks, yticklabels = y_ticklabels, mask_color='yellow', xlabel='Time (2s)', ylabel='Frequency (MHz)', cbar=False, two_colors=True, cbar_ticks=[0, 0.5, 1], cmap=cmap, norm=norm, boundaries=bounds, vmin=0, vmax=1, cbar_ticks_labels=['Non-Flagged', '', 'Flagged'], cbar_label='', xline=False, xlineticks=[50,100, 150, 200, 250, 300, 350, 400])
prefix = '%s/Pysegments_Oneplot_PerFrCh_GenExtreme_Alpha0005_Dataset' % (outdir)

fig.savefig('%s.pdf' % prefix)
plt.show()
