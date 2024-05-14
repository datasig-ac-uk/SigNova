import numpy as np
import warnings

def image_plot(fig, ax, data, cmap=None, vmin=None, vmax=None, title='', xlabel='', ylabel='', midpoint=False, aspect='auto', cbar_label=None, xticks=None, yticks=None, log=False, xticklabels=None, yticklabels=None, vline=None, rotate=None, annotate=None, ymin=None, mask_color='white', cbar_ticks=None, cbar=True, font_size=10, extend='neither', alpha=None, two_colors=None, norm=None, boundaries=None, cbar_ticks_labels=None, xline=False, xlineticks=None, extent=None):


    from matplotlib import colors, cm

    if cmap is None:
        cmap = cm.plasma

    class MidpointNormalize(colors.Normalize):
        

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
           
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):  
        
            result, is_scalar = self.process_value(value)
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)

    if two_colors:
        cax = ax.imshow(data, cmap=cmap, interpolation='none', norm=norm, aspect=aspect,
                         alpha=alpha)
    else:
        cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect,
                        interpolation='none', extent=extent, alpha=alpha)

    cmap.set_bad(color=mask_color)

    if cbar is not False:
        cbar = fig.colorbar(cax, ax=ax, ticks=cbar_ticks, extend=extend)
        cbar.set_label(cbar_label, fontsize=font_size)
    else:
        cbar = fig.colorbar(cax, ax=ax, ticks=cbar_ticks, cmap=cmap, norm=norm, boundaries=boundaries)
        cbar.set_label(cbar_label, fontsize=font_size)
        cbar.ax.set_yticklabels(cbar_ticks_labels) 


    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    if xline is True:
        for xc in xlineticks:
            ax.axvline(x=xc-0.5, ls = '--', color = 'black')

    if extent is None:
        set_ticks_labels(ax, xticks, yticks, xticklabels, yticklabels, vline, ymin, rotate)
    elif (xticks is not None) or (yticks is not None):
        set_ticks_labels(ax, xticks, yticks, xticklabels, yticklabels)

    if annotate is True:
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                text = ax.text(j, i, data[i, j].round(5), ha="center", va="center", color="black")

    cbar.ax.tick_params(labelsize=font_size)
    ax.tick_params(labelsize=font_size)


def set_ticks_labels(ax, xticks, yticks, xticklabels, yticklabels, vline, ymin, rotate):
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=rotate)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if vline is not None:
        ax.vlines(vline, ymin=ymin, ymax=-0.5, color='black')
