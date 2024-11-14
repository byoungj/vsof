
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

def plot_boundaries(surfaces, labels=None, nphi=4, ntheta=51, ax=None, return_data=False, gif_path=None, **kwargs):
    """
    Plot stellarator boundaries at multiple toroidal coordinates with separate subplots for each φ.

    Parameters
    ----------
    surfaces : array-like of Surface
        Surfaces to plot.
    labels : array-like, optional
        Array of the same length as surfaces of labels to apply to each surface.
    nphi : int, optional
        Value of nphi to plot the boundary surface at.
        plot nphi many contours linearly spaced in [0, 2 * np.pi / surf.nfp).
    ntheta : int, optional
        Value of ntheta to plot on the boundary surface at.
        Determines the resolution of the curve.
    ax : matplotlib AxesSubplot, optional
        Axis to plot on. If None, a new figure and axis are created.
    return_data : bool, optional
        If True, return the data plotted as well as fig, ax.
    gif_path : str, optional
        Path to save a GIF of the plot with varying surfaces.
    **kwargs : dict, optional
        Specify figure, axis, and plot appearance properties.

        Valid keyword arguments are:
            * figsize: tuple (width, height)
            * xlabel_fontsize: float
            * ylabel_fontsize: float
            * legend: bool, whether to display legend or not
            * legend_kw: dict, keyword arguments passed to ax.legend()
            * cmap: colormap to generate uniform colors for each surface.
            * ls: list of line styles (e.g., ['-', '--', ':', '-.']) to use for surfaces.
            * lw: list of line widths to use for surfaces.
            * marker: str, marker style for axis points
            * size: float, marker size for axis points
            * font_family: str, font family for the plot
            * dpi: int, DPI setting for the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure being plotted to.
    ax : matplotlib.axes.Axes or ndarray of Axes
        Axes being plotted to.
    plot_data : dict
        Dictionary of the data plotted, only returned if return_data=True.

    Examples
    --------
    fig, ax = plot_boundaries([surf1, surf2, surf3], labels=["Surface 1", "Surface 2", "Surface 3"])
    """
    
    dpi = kwargs.get('dpi', 100) 
    font_family = kwargs.get('font_family', 'sans-serif')
    figsize = kwargs.get('figsize', (10, 3))  
    xlabel_fontsize = kwargs.get('xlabel_fontsize', 12)
    ylabel_fontsize = kwargs.get('ylabel_fontsize', 12)
    legend = kwargs.get('legend', True)
    legend_kw = kwargs.get('legend_kw', {})
    lw = kwargs.get('lw', 1.5)
    
    line_styles = kwargs.get('ls', ['-', '--', ':', '-.'])

    cmap = kwargs.get('cmap', plt.get_cmap('plasma'))
    n_surfaces = len(surfaces)
    colors = [cmap(i / n_surfaces) for i in range(n_surfaces)]
    
    fig, axs = plt.subplots(1, nphi, dpi=dpi, figsize=figsize, sharex=True, sharey=True)
    
    if labels is None:
        labels = [f"Surface {i+1}" for i in range(len(surfaces))]
    # Collect global min/max values for fixing axes
    all_R, all_Z = [], []
    for surf in surfaces:
        for iphi in range(nphi):
            theta = np.linspace(0, 2 * np.pi, num=ntheta)
            R, Z = np.zeros(ntheta), np.zeros(ntheta)
            for itheta in range(ntheta):
                for imode in range(len(surf.n)):
                    m, n = surf.m[imode], surf.n[imode]
                    angle = m * theta[itheta] - surf.nfp * n * iphi * 2 * np.pi / nphi
                    R[itheta] += surf.get_rc(m, n) * np.cos(angle)
                    Z[itheta] += surf.get_zs(m, n) * np.sin(angle)
            all_R.append(R)
            all_Z.append(Z)
    
    R_min, R_max = np.min(all_R), np.max(all_R)
    Z_min, Z_max = np.min(all_Z), np.max(all_Z)
    sqr_min, sqr_max = -(R_max-R_min),R_max-R_min 

    if gif_path:
        images = []  
    plot_data = {}

    for idx, surf in enumerate(surfaces):
        if gif_path:
            lam_values = np.linspace(-0.3, 1.3, len(surfaces))
            fig, axs = plt.subplots(1, nphi, dpi=dpi, figsize=figsize, sharex=True, sharey=True)
            fig.suptitle(rf'$RC_{{0,1}} = {lam_values[idx]:.2f}$', fontsize=16)

        # Loop over each phi angle to create separate subplots
        for iphi in range(nphi):
            ax = axs[iphi] if nphi > 1 else axs
            label_str_phi = f'$\phi={iphi}/{nphi}$ period'

            nfp = surf.nfp
            nmodes = len(surf.n)
            current_color = colors[idx]
            current_ls = line_styles[idx % len(line_styles)] 

            theta = np.linspace(0, 2 * np.pi, num=ntheta)
            R = np.zeros(ntheta)
            Z = np.zeros(ntheta)

            # Compute R and Z using Fourier series for the current surface and phi
            for itheta in range(ntheta):
                for imode in range(nmodes):
                    m = surf.m[imode]
                    n = surf.n[imode]
                    angle = m * theta[itheta] - nfp * n * iphi * 2 * np.pi / nphi
                    R[itheta] += surf.get_rc(m, n) * np.cos(angle)
                    Z[itheta] += surf.get_zs(m, n) * np.sin(angle)

            plot_data[f'surf_{idx+1}_phi_{iphi}'] = {'R': R, 'Z': Z, 'theta': theta, 'phi': iphi}

            # Plot the boundary for the current surface on this subplot
            ax.plot(R, Z, ls=current_ls, lw=lw, label=labels[idx] if labels else f"Surface {idx+1}", color=current_color)
            ax.set_xlim([R_min, R_max])
            ax.set_ylim([Z_min, Z_max])
            ax.set_aspect('equal')
            ax.set_title(label_str_phi, fontsize=12)
            if iphi == 0:
                ax.set_ylabel('Z', fontsize=ylabel_fontsize)
            ax.set_xlabel('R', fontsize=xlabel_fontsize)

            if legend:
                ax.legend(**legend_kw)

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f'${val:.1f}$'))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f'${val:.1f}$'))

        plt.tight_layout()
        plt.rcParams["font.family"] = font_family
        plt.rcParams["mathtext.fontset"] = "dejavusans"

        if gif_path:
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image_from_plot)
            plt.close(fig)  

    if gif_path:
        images += images[::-1][1:-1]  # Append reversed images to create back-and-forth effect
        imageio.mimsave(gif_path, images, fps=10, loop = 0) 
        
    if return_data:
        return plot_data
    else:
        return fig, axs


