"""
Ordinary Kriging Example
========================

First we will create a 2D dataset together with the associated x, y grids.

"""
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import pickle
import numpy as np
import pykrige.kriging_tools as kt
from pykrige.uk import UniversalKriging
import matplotlib.pyplot as plt
import os

import deal_with_files
from deal_with_files import get_kriging_suffix
import constants as c
import binning

if not os.path.exists(f"figures{c.bin_sizes}/"):
    os.mkdir(f"figures{c.bin_sizes}/")

def plot_variance(vmesh, NTag, show=False, uk_kwargs=None):
    """
    vmesh: matrix, variance at each mesh point
    """
    xmesh, ymesh = np.meshgrid(c.xbins, c.ybins)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(xmesh, ymesh, vmesh, shading='auto')
    fig.colorbar(im, ax=ax)
    suffix = get_kriging_suffix(uk_kwargs)
    plt.title(f'GP estimator variance, NTag={NTag}, {suffix}')
    plt.savefig(f"figures{c.bin_sizes}/kriging_variance_{suffix}_{NTag}b.png")
    if show:
        plt.show()
    plt.close()

def plot_predictions(xbins, ybins, zpred_grid, NTag, show=False, uk_kwargs=None):
    fig, ax = plt.subplots()
    gridx, gridy = np.meshgrid(xbins, ybins)
    im = ax.pcolormesh(gridx, gridy, zpred_grid, shading='auto')
    fig.colorbar(im, ax=ax)

    suffix = get_kriging_suffix(uk_kwargs)
    fig_title = f'GPR Output, NTag={NTag}, {suffix}'
    file_title = f"kriging_{suffix}_{NTag}b"
    if n_indices is not None:
        fig_title += f" n_indices={n_indices}"
        file_title += f"_n{n_indices}"

    plt.title(fig_title)
    plt.savefig(f"figures{c.bin_sizes}/{file_title}.png")
    if show:
        plt.show()
    plt.close()


def krige(NTag, x, y, z, n_indices=None, show=False, uk_kwargs=None):
    """
    Runs universal kriging on the x y z data generated earlier

    Returns matrix of predictions and matrix of variances at each point.

    NTag:
        2 or 4, which data to use
    x, y, z:
        data to use for the fit, 1d arrays
    n_indices:
        integer, for a sampling of 10 points, say n_indices = 10
    show:
        boolean, whether to display plot of predictions
    """


    if n_indices is not None:
        print("sampling", n_indices, "indices")
        indices = np.random.randint(0,len(x), n_indices)
        sampled_x = x[indices]
        sampled_y = y[indices]
        sampled_z = z[indices]
    else:
        sampled_x = x
        sampled_y = y
        sampled_z = z

    if NTag == 4:
        print('removing SR')
        in_SR = binning.binInSR(sampled_x, sampled_y)
        filtered_x = sampled_x[np.logical_not(in_SR)]
        filtered_y = sampled_y[np.logical_not(in_SR)]
        filtered_z = sampled_z[np.logical_not(in_SR)]
    else:
        filtered_x = sampled_x
        filtered_y = sampled_y
        filtered_z = sampled_z

    ###########################################################################
    # Create the kriging object. Required inputs are the X-coordinates of
    # the data points, the Y-coordinates of the data points, and the Z-values
    # of the data points. If no variogram model is specified, defaults to a
    # linear variogram model. If no variogram model parameters are specified,
    # then the code automatically calculates the parameters by fitting the
    # variogram model to the binned experimental semivariogram. The verbose
    # kwarg controls code talk-back, and the enable_plotting kwarg controls
    # the display of the semivariogram.

    """
    Variables to play with:

    x, y, z: inputs, don't mess with those

    variogram_model = linear, power, gaussian, spherical, exponential, hole-effect

    variogram_parameters = 
            # linear
               {'slope': slope, 'nugget': nugget}
            # power
               {'scale': scale, 'exponent': exponent, 'nugget': nugget}
            # gaussian, spherical, exponential and hole-effect:
               {'sill': s, 'range': r, 'nugget': n}
               # OR
               {'psill': p, 'range': r, 'nugget': n}
    nlags = integer, default 6
    weight = bool, default False
    drift_terms : list of strings, optional
        List of drift terms to include in universal kriging. Supported drift
        terms are currently 'regional_linear', 'point_log', 'external_Z',
        'specified', and 'functional'.
    exact_values : bool, default True
    """
    UK = UniversalKriging(
        filtered_x,
        filtered_y,
        filtered_z,
        verbose=True,
        enable_plotting=False,
        **uk_kwargs
    )

    ###########################################################################
    # Creates the kriged grid and the variance grid. Allows for kriging on a
    # rectangular grid of points, on a masked rectangular grid of points, or
    # with arbitrary points. (See UniversalKriging.__doc__ for more info)

    # inputs on which to evaluete the kriged model
    # can be any values really, e.g. these
    #gridx = np.arange(np.min(original_x), np.max(original_x)+1, x_grid_res)
    #gridy = np.arange(np.min(original_y), np.max(original_y)+1, y_grid_res)
    # or evaluate on the original points

    xbins = list(sorted(set(x)))
    ybins = list(sorted(set(y)))

    print(len(x))
    print(len(xbins))

    # evaluate
    zpred_grid, variance_grid = UK.execute("grid", xbins, ybins)

    ###########################################################################
    # might as well plot while we're at it
    plot_predictions(xbins, ybins, zpred_grid, NTag, show=show, uk_kwargs=uk_kwargs)
    plot_variance(variance_grid, NTag, show=show, uk_kwargs=uk_kwargs)

    ###########################################################################
    # save z preds and variance so we can play with them later
    deal_with_files.save_kriging(NTag, uk_kwargs, zpred_grid, variance_grid)

    return zpred_grid, variance_grid

def get_kriging_prediction(NTag, x, y, z, uk_kwargs, n_indices=None):
    zpred_grid, var_grid = deal_with_files.load_kriging(NTag, uk_kwargs)
    if zpred_grid is None:
        zpred_grid, var_grid = krige(
            NTag, x, y, z, n_indices=n_indices, uk_kwargs=uk_kwargs)
    return zpred_grid, var_grid

if __name__ == "__main__":
    """
    variogram_parameters = 
            # linear
               {'slope': slope, 'nugget': nugget}
            # power
               {'scale': scale, 'exponent': exponent, 'nugget': nugget}
            # gaussian, spherical, exponential and hole-effect:
               {'sill': s, 'range': r, 'nugget': n}
               # OR
               {'psill': p, 'range': r, 'nugget': n}
    nlags = integer, default 6
    weight = bool, default False
    drift_terms : list of strings, optional
        List of drift terms to include in universal kriging. Supported drift
        terms are currently 'regional_linear', 'point_log', 'external_Z',
        'specified', and 'functional'.
    exact_values : bool, default True
    """
    print('loading x y z inputs')

    NTag = 4
    n_indices = None
    x, y, z = deal_with_files.load_1d(NTag=NTag)
    
    for ev in [True, False]:
        for vm in ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]:
            uk_kwargs = {
                "variogram_model": vm,
                'exact_values': ev
            }
            zpred, variance = get_kriging_prediction(NTag, x, y, z, uk_kwargs, n_indices=n_indices)