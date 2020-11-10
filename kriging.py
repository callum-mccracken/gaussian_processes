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
from pykrige.uk3d import UniversalKriging3D
import matplotlib.pyplot as plt
import os

import deal_with_files
from deal_with_files import get_kriging_suffix
import constants as c
import binning

if not os.path.exists(f"figures{c.bin_sizes}/"):
    os.mkdir(f"figures{c.bin_sizes}/")
if not os.path.exists(f"PG_figures{c.bin_sizes}/"):
    os.mkdir(f"PG_figures{c.bin_sizes}/")

def plot_variance(xbins, ybins, vmesh, NTag, show=False, uk_kwargs=None, pairagraph=False):
    """
    vmesh: matrix, variance at each mesh point
    """
    pg = "PG_" if pairagraph else ""
    xmesh, ymesh = np.meshgrid(xbins, ybins)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(xmesh, ymesh, vmesh, shading='auto')
    fig.colorbar(im, ax=ax)
    suffix = get_kriging_suffix(uk_kwargs)
    plt.title(f'GP estimator variance, NTag={NTag}, {suffix}')
    plt.savefig(f"{pg}figures{c.bin_sizes}/kriging_variance_{suffix}_{NTag}b.png")
    if show:
        plt.show()
    plt.close()

def plot_predictions(xbins, ybins, zpred_grid, NTag, show=False, uk_kwargs=None, pairagraph=False):
    pg = "PG_" if pairagraph else ""

    fig, ax = plt.subplots()
    gridx, gridy = np.meshgrid(xbins, ybins)
    im = ax.pcolormesh(gridx, gridy, zpred_grid, shading='auto')
    fig.colorbar(im, ax=ax)

    suffix = get_kriging_suffix(uk_kwargs)
    fig_title = f'GPR Output, NTag={NTag}, {suffix}'
    file_title = f"{pg}_kriging_{suffix}_{NTag}b"
    if n_indices is not None:
        fig_title += f" n_indices={n_indices}"
        file_title += f"_n{n_indices}"

    plt.title(fig_title)
    plt.savefig(f"figures{c.bin_sizes}/{file_title}.png")
    if show:
        plt.show()
    plt.close()


def krige_2d(NTag, mh1, mh2, pdf, n_indices=None, show=False, uk_kwargs=None, pairagraph=False):
    """
    Runs universal kriging on the mh1 mh2 pdf data generated earlier

    Returns matrix of predictions and matrix of variances at each point.

    Parameters::
    - NTag: 2 or 4, which data to use
    - mh1, mh2, pdf: data to use for the fit, 1d arrays
    - n_indices: integer, for a sampling of 10 points, say n_indices = 10
    - show: boolean, whether to display plot of predictions
    - uk_kwargs: dict, kwargs for pykrige
    - pairagraph: bool, did we use pairagraph data?
    """


    if n_indices is not None:
        print("sampling", n_indices, "indices")
        indices = np.random.randint(0,len(mh1), n_indices)
        sampled_mh1 = mh1[indices]
        sampled_mh2 = mh2[indices]
        sampled_pdf = pdf[indices]
    else:
        sampled_mh1 = mh1
        sampled_mh2 = mh2
        sampled_pdf = pdf

    if NTag == 4:
        print('removing SR')
        in_SR = binning.binInSR(sampled_mh1, sampled_mh2)
        filtered_mh1 = sampled_mh1[np.logical_not(in_SR)]
        filtered_mh2 = sampled_mh2[np.logical_not(in_SR)]
        filtered_pdf = sampled_pdf[np.logical_not(in_SR)]
    else:
        filtered_mh1 = sampled_mh1
        filtered_mh2 = sampled_mh2
        filtered_pdf = sampled_pdf

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
        filtered_mh1,
        filtered_mh2,
        filtered_pdf,
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

    mh1_bins = np.linspace(min(mh1), max(mh1), 200)  # list(sorted(set(x)))
    mh2_bins = np.linspace(min(mh2), max(mh2), 200)  # list(sorted(set(y)))

    #print(len(mh1))
    #print(len(mh1_bins))

    # evaluate
    pdf_pred_grid, variance_grid = UK.execute("grid", mh1_bins, mh2_bins)

    ###########################################################################
    # might as well plot while we're at it
    plot_predictions(
        mh1_bins, mh2_bins, pdf_pred_grid, NTag,
        show=show, uk_kwargs=uk_kwargs, pairagraph=pairagraph)
    plot_variance(
        mh1_bins, mh2_bins, variance_grid, NTag,
        show=show, uk_kwargs=uk_kwargs, pairagraph=pairagraph)

    ###########################################################################
    # save z preds and variance so we can play with them later
    deal_with_files.save_kriging(
        NTag, uk_kwargs, pdf_pred_grid, variance_grid, dim=2, pairagraph=pairagraph)

    return pdf_pred_grid, variance_grid

def krige_3d(NTag, mh1, mh2, mhh, pdf, n_indices=None, show=False, uk_kwargs=None, pairagraph=False):
    """
    Runs universal kriging on the mh1 mh2 pdf data generated earlier

    Returns matrix of predictions and matrix of variances at each point.

    Parameters::
    - NTag: 2 or 4, which data to use
    - mh1, mh2, mhh, pdf: data to use for the fit, 1d arrays
    - n_indices: integer, for a sampling of 10 points, say n_indices = 10
    - show: boolean, whether to display plot of predictions
    - uk_kwargs: dict, kwargs for pykrige
    - pairagraph: bool, did we use pairagraph data?
    """


    if n_indices is not None:
        print("sampling", n_indices, "indices")
        indices = np.random.randint(0,len(mh1), n_indices)
        sampled_mh1 = mh1[indices]
        sampled_mh2 = mh2[indices]
        sampled_mhh = mhh[indices]
        sampled_pdf = pdf[indices]
    else:
        sampled_mh1 = mh1
        sampled_mh2 = mh2
        sampled_mhh = mhh
        sampled_pdf = pdf

    if NTag == 4:
        print('removing SR')
        in_SR = binning.binInSR(sampled_mh1, sampled_mh2)
        filtered_mh1 = sampled_mh1[np.logical_not(in_SR)]
        filtered_mh2 = sampled_mh2[np.logical_not(in_SR)]
        filtered_mhh = sampled_mhh[np.logical_not(in_SR)]
        filtered_pdf = sampled_pdf[np.logical_not(in_SR)]
    else:
        filtered_mh1 = sampled_mh1
        filtered_mh2 = sampled_mh2
        filtered_mhh = sampled_mhh
        filtered_pdf = sampled_pdf

    UK = UniversalKriging3D(
        filtered_mh1,
        filtered_mh2,
        filtered_mhh,
        filtered_pdf,
        verbose=True,
        enable_plotting=False,
        **uk_kwargs
    )

    mh1_bins = np.linspace(min(mh1), max(mh1), 200)
    mh2_bins = np.linspace(min(mh2), max(mh2), 200)
    mhh_bins = np.linspace(min(mhh), max(mhh), 20)

    # evaluate
    pdf_pred_grid, variance_grid = UK.execute("grid", mh1_bins, mh2_bins, mhh_bins)

    # convert to 2d for export
    print(pdf_pred_grid.shape)
    pdf_pred_grid = np.sum(pdf_pred_grid, axis=3)
    print(pdf_pred_grid.shape)

    print(variance_grid.shape)
    variance_grid = np.sum(variance_grid, axis=3) # TODO: do we just sum these or do we sum with some kind of factor?
    print(variance_grid.shape)

    ###########################################################################
    # might as well plot while we're at it
    plot_predictions(
        mh1_bins, mh2_bins, pdf_pred_grid, NTag,
        show=show, uk_kwargs=uk_kwargs, pairagraph=pairagraph)
    plot_variance(
        mh1_bins, mh2_bins, variance_grid, NTag,
        show=show, uk_kwargs=uk_kwargs, pairagraph=pairagraph)

    ###########################################################################
    # save z preds and variance so we can play with them later
    deal_with_files.save_kriging(
        NTag, uk_kwargs, pdf_pred_grid, variance_grid, dim=3, pairagraph=pairagraph)

    return pdf_pred_grid, variance_grid

def get_kriging_prediction_2d(NTag, mh1, mh2, pdf, uk_kwargs, n_indices=None, pairagraph=False):
    zpred_grid, var_grid = deal_with_files.load_kriging(NTag, uk_kwargs, pairagraph=pairagraph, dim=2)
    if zpred_grid is None:
        zpred_grid, var_grid = krige_2d(
            NTag, mh1, mh2, pdf, n_indices=n_indices,
            uk_kwargs=uk_kwargs, pairagraph=pairagraph)
    return zpred_grid, var_grid

def get_kriging_prediction_3d(NTag, mh1, mh2, mhh, pdf, uk_kwargs, n_indices=None, pairagraph=False):
    zpred_grid, var_grid = deal_with_files.load_kriging(NTag, uk_kwargs, pairagraph=pairagraph, dim=3)
    if zpred_grid is None:
        zpred_grid, var_grid = krige_3d(
            NTag, mh1, mh2, mhh, pdf, n_indices=n_indices,
            uk_kwargs=uk_kwargs, pairagraph=pairagraph)
    return zpred_grid, var_grid


if __name__ == "__main__":
    """
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
    dim = 3
    pairagraph = False
    n_indices = None
    mh1, mh2, mhh, pdf = deal_with_files.load_flattened(NTag=NTag, pairagraph=pairagraph, dim=dim)
    #for s in [800,900,1000,1100]:
    #    for r in [20,40,60,80,100,120,140,160]:
    #        for n in [1e-12, 1e-11,1e-10,1e-9,1e-8]:
    #s = 800
    #r = 160
    #n = 1e-8
    uk_kwargs = {
        "variogram_model": "gaussian",
        'exact_values': True,
    #    'variogram_parameters': {'sill': s, 'range': r, 'nugget': n}
    }
    zpred, variance = get_kriging_prediction_3d(NTag, mh1, mh2, mhh, pdf, uk_kwargs, n_indices=n_indices, pairagraph=pairagraph)
