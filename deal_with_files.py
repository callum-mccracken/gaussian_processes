import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import constants as c
import plot_functions
import binning


suffix = f"{c.NxbinsInSig}_{c.NybinsInSig}_{c.n_mhhbins}"


def get_kriging_suffix(uk_kwargs):
    suffix = ""
    if uk_kwargs is None:
        return "none"
    else:
        for var in sorted(uk_kwargs.keys()):
            suffix += str(uk_kwargs[var])
        return suffix

def integrate_pdf_2d(df):
    """given a dataframe, sum up pdf with (mh1, mh2) in each bin"""
    row_list = []
    for xi in tqdm(c.xbins):
        for yi in c.ybins:
            row_list.append({"mh1":xi,"mh2":yi,
                "pdf":sum(df.loc[ (df["mh1"]==xi) 
                                & (df["mh2"]==yi),"pdf"])})
    return pandas.DataFrame(row_list)

def integrate_pdf_3d(df):
    """given a dataframe, sum up pdf with (mh1, mh2, mhh) in each bin"""
    row_list = []
    for xi in tqdm(c.xbins):
        for yi in c.ybins:
            for zi in c.mhhbins:
                row_list.append({"mh1":xi,"mh2":yi,"mhh":zi,
                    "pdf":sum(df.loc[ (df["mh1"]==xi)
                                    & (df["mh2"]==yi)
                                    & (df["mhh"]==zi),"pdf"])})
    return pandas.DataFrame(row_list)


# Integrates the fullmassplane to get slices of mhh
def integrate_fmp(df, mhhbins):
    row_list = []
    for mhh in mhhbins[:-1]:
        row_list.append({"mhh":mhh,"pdf":sum(df.loc[df["mhh"]==mhh,"pdf"])})
    return pandas.DataFrame(row_list)

def create_mesh_2d(NTag=4, pairagraph=False):
    print('creating mesh')
    pg = "PG_" if pairagraph else ""
    # pandas df with 3 columns: m_h1, m_h2, and m_hh
    df = pandas.read_pickle(f"data/{pg}data_{NTag}tag.p")
    print(len(df),"Events")

    # Now make the 3D histogram
    coord_array = np.array(df[["m_h1","m_h2","m_hh"]])
    hist3d,[xbins,ybins,mhhbins] = np.histogramdd(
        coord_array,[c.xbins,c.ybins,c.mhhbins])
    xv,yv,zv = np.meshgrid(
        xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')
    data_df = pandas.DataFrame()
    data_df["mh1"] = xv.flatten()
    data_df["mh2"] = yv.flatten()
    data_df["mhh"] = zv.flatten()
    data_df["pdf"] = hist3d.flatten()

    modeldffmp = integrate_pdf_2d(data_df)

    shape = (c.n_xbins,c.n_ybins)
    mh1_mesh = np.array(modeldffmp["mh1"]).reshape(shape).transpose()
    mh2_mesh = np.array(modeldffmp["mh2"]).reshape(shape).transpose()
    pdf_mesh = np.array(modeldffmp["pdf"]).reshape(shape).transpose()
    if not os.path.exists(f"{pg}data{c.bin_sizes}"):
        os.mkdir(f"{pg}data{c.bin_sizes}")
    with open(f"{pg}data{c.bin_sizes}/{pg}2d_mh1_mesh_{suffix}.p",'wb') as mh1_file:
        pickle.dump(mh1_mesh,mh1_file)
    with open(f"{pg}data{c.bin_sizes}/{pg}2d_mh2_mesh_{suffix}.p",'wb') as mh2_file:
        pickle.dump(mh2_mesh,mh2_file)
    with open(f"{pg}data{c.bin_sizes}/{pg}2d_pdf_mesh_{NTag}tag_{suffix}.p",'wb') as pdf_file:
        pickle.dump(pdf_mesh,pdf_file)


def create_mesh_3d(NTag=4, pairagraph=False):
    print('creating mesh')
    pg = "PG_" if pairagraph else ""
    # pandas df with 3 columns: m_h1, m_h2, and m_hh
    df = pandas.read_pickle(f"data/{pg}data_{NTag}tag.p")
    print(len(df),"Events")

    # Now make the 3D histogram
    coord_array = np.array(df[["m_h1","m_h2","m_hh"]])
    hist3d,[xbins,ybins,mhhbins] = np.histogramdd(
        coord_array,[c.xbins,c.ybins,c.mhhbins])
    xv,yv,zv = np.meshgrid(
        xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')
    data_df = pandas.DataFrame()
    data_df["mh1"] = xv.flatten()
    data_df["mh2"] = yv.flatten()
    data_df["mhh"] = zv.flatten()
    data_df["pdf"] = hist3d.flatten()

    modeldffmp = integrate_pdf_3d(data_df)

    shape = (c.n_xbins,c.n_ybins, c.n_mhhbins)
    mh1_mesh = np.array(modeldffmp["mh1"]).reshape(shape).transpose()
    mh2_mesh = np.array(modeldffmp["mh2"]).reshape(shape).transpose()
    mhh_mesh = np.array(modeldffmp["mhh"]).reshape(shape).transpose()
    pdf_mesh = np.array(modeldffmp["pdf"]).reshape(shape).transpose()
    if not os.path.exists(f"{pg}data{c.bin_sizes}"):
        os.mkdir(f"{pg}data{c.bin_sizes}")
    with open(f"{pg}data{c.bin_sizes}/{pg}3d_mh1_mesh_{suffix}.p",'wb') as mh1_file:
        pickle.dump(mh1_mesh,mh1_file)
    with open(f"{pg}data{c.bin_sizes}/{pg}3d_mh2_mesh_{suffix}.p",'wb') as mh2_file:
        pickle.dump(mh2_mesh,mh2_file)
    with open(f"{pg}data{c.bin_sizes}/{pg}3d_mhh_mesh_{suffix}.p",'wb') as mhh_file:
        pickle.dump(mhh_mesh,mhh_file)
    with open(f"{pg}data{c.bin_sizes}/{pg}3d_pdf_mesh_{NTag}tag_{suffix}.p",'wb') as pdf_file:
        pickle.dump(pdf_mesh,pdf_file)

def load_mesh(NTag=4, bins=None, pairagraph=False, dim=2):
    pg = "PG_" if pairagraph else ""
    if dim == 2:
        if not os.path.exists(f'{pg}data{c.bin_sizes}/{pg}{dim}d_count_mesh_{NTag}tag_{suffix}.p'):
            create_mesh_2d(NTag, pairagraph=pairagraph)
        with open(f'{pg}data{c.bin_sizes}/{pg}{dim}d_mh1_mesh_{suffix}.p', 'rb') as mh1_file:
            mh1_mesh = pickle.load(mh1_file)
        with open(f'{pg}data{c.bin_sizes}/{pg}{dim}d_mh2_mesh_{suffix}.p', 'rb') as mh2_file:
            mh2_mesh = pickle.load(mh2_file)
        with open(f"{pg}data{c.bin_sizes}/{pg}{dim}d_pdf_mesh_{NTag}tag_{suffix}.p",'rb') as pdf_file:
            pdf_mesh = pickle.load(pdf_file)
        return mh1_mesh, mh2_mesh, pdf_mesh
    elif dim == 3:
        if not os.path.exists(f'{pg}data{c.bin_sizes}/{pg}{dim}d_count_mesh_{NTag}tag_{suffix}.p'):
            create_mesh_3d(NTag, pairagraph=pairagraph)
        with open(f'{pg}data{c.bin_sizes}/{pg}{dim}d_mh1_mesh_{suffix}.p', 'rb') as mh1_file:
            mh1_mesh = pickle.load(mh1_file)
        with open(f'{pg}data{c.bin_sizes}/{pg}{dim}d_mh2_mesh_{suffix}.p', 'rb') as mh2_file:
            mh2_mesh = pickle.load(mh2_file)
        with open(f'{pg}data{c.bin_sizes}/{pg}{dim}d_mhh_mesh_{suffix}.p', 'rb') as mhh_file:
            mhh_mesh = pickle.load(mhh_file)
        with open(f"{pg}data{c.bin_sizes}/{pg}{dim}d_pdf_mesh_{NTag}tag_{suffix}.p",'rb') as pdf_file:
            pdf_mesh = pickle.load(pdf_file)
        return mh1_mesh, mh2_mesh, mhh_mesh, pdf_mesh
    else:
        raise ValueError('invalid dim')

def load_kriging(NTag, uk_kwargs, dim, pairagraph=False):
    """Load kriging files if they exist

    NTag: int, 2 or 4
    uk_kwargs: dict, kwargs for the pykrige module
    pairagraph: bool, whether or not to use pairagraph data
    dim: int, 2 or 3, for (mh1, mh2) vs (mh1, mh2, mhh) kriging
    """
    pg = "PG_" if pairagraph else ""
    suffix = get_kriging_suffix(uk_kwargs)
    # predictions
    zfilename = f"{pg}data{c.bin_sizes}/{pg}{dim}d_kriging_{suffix}_{NTag}b_z.p"
    # variances
    vfilename = f"{pg}data{c.bin_sizes}/{pg}{dim}d_kriging_{suffix}_{NTag}b_v.p"
    if os.path.exists(zfilename) and os.path.exists(vfilename):
        with open(zfilename, 'rb') as zfile:
            zpred_grid = pickle.load(zfile)
        with open(vfilename, 'rb') as vfile:
            var_grid = pickle.load(vfile)
        print('successfully loaded files from')
        print(zfilename)
        print(vfilename)
        return zpred_grid, var_grid
    else:
        print('kriging files not found')
        return None, None

def save_kriging(NTag, uk_kwargs, pdf_pred_grid, variance_grid, dim,
                 n_indices=None, pairagraph=False):
    """
    Save data from kriging. Most parameters are just for file naming purposes

    Parameters::
    - NTag: int, 2 or 4
    - uk_kwargs: dict, kwargs for kriging
    - pdf_pred_grid: array, predictions
    - variance_grid: array, estimator variances
    - dim: int, 2 or 3
    - n_indices: None or int, number of samples to take
    - pairagraph: bool, did we use pairagraph or not
    """
    pg = "PG_" if pairagraph else ""
    suffix = get_kriging_suffix(uk_kwargs)
    file_title = f"{pg}{dim}d_kriging_{suffix}_{NTag}b"
    if n_indices is not None:
        file_title += f"_n{n_indices}"
    with open(f"{pg}data{c.bin_sizes}/{file_title}_z.p", 'wb') as pdf_file:
        pickle.dump(pdf_pred_grid, pdf_file)
    with open(f"{pg}data{c.bin_sizes}/{file_title}_v.p", 'wb') as vfile:
        pickle.dump(variance_grid, vfile)

def load_flattened(NTag=4, pairagraph=False, dim=2):
    """
    loads flattened arrays containing data needed for kriging fits

    Parameters:
    - NTag: int, 2 or 4
    - pairagraph: bool, did we use pairagraph data?
    - dim: int, 2 or 3, type of kriging fit
    """
    if dim==2:
        mh1_mesh, mh2_mesh, count_mesh = load_mesh(NTag, pairagraph=pairagraph, dim=2)
        return mh1_mesh.flatten(), mh2_mesh.flatten(), count_mesh.flatten()
    elif dim==3:
        mh1_mesh, mh2_mesh, mhh_mesh, count_mesh = load_mesh(NTag, pairagraph=pairagraph, dim=3)
        return mh1_mesh.flatten(), mh2_mesh.flatten(), mhh_mesh.flatten(), count_mesh.flatten()


if __name__ == "__main__":
    load_flattened(NTag=2, dim=3)