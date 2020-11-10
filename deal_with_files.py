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

# Have function to integrate mhh in each bin of the fullmassplane
def integrate_mhh(df):
    row_list = []
    for xi in tqdm(c.xbins):
        for yi in c.ybins:
            row_list.append({"mh1":xi,"mh2":yi,
                "pdf":sum(df.loc[ (df["mh1"]==xi) & (df["mh2"]==yi),"pdf"])})
    return pandas.DataFrame(row_list)

# Integrates the fullmassplane to get slices of mhh
def integrate_fmp(df, mhhbins):
    row_list = []
    for mhh in mhhbins[:-1]:
        row_list.append({"mhh":mhh,"pdf":sum(df.loc[df["mhh"]==mhh,"pdf"])})
    return pandas.DataFrame(row_list)

def create_mesh(NTag=4, pairagraph=False):
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

    modeldffmp = integrate_mhh(data_df)

    shape = (c.n_xbins,c.n_ybins)
    xmesh = np.array(modeldffmp["mh1"]).reshape(shape).transpose()
    ymesh = np.array(modeldffmp["mh2"]).reshape(shape).transpose()
    hmesh = np.array(modeldffmp["pdf"]).reshape(shape).transpose()
    if not os.path.exists(f"{pg}data{c.bin_sizes}"):
        os.mkdir(f"{pg}data{c.bin_sizes}")
    with open(f"{pg}data{c.bin_sizes}/{pg}xmesh_{suffix}.p",'wb') as xfile:
        pickle.dump(xmesh,xfile)
    with open(f"{pg}data{c.bin_sizes}/{pg}ymesh_{suffix}.p",'wb') as yfile:
        pickle.dump(ymesh,yfile)
    with open(f"{pg}data{c.bin_sizes}/{pg}hmesh_{NTag}tag_{suffix}.p",'wb') as hfile:
        pickle.dump(hmesh,hfile)

def load_mesh(NTag=4, bins=None, pairagraph=False):
    pg = "PG_" if pairagraph else ""
    if not os.path.exists(f'{pg}data{c.bin_sizes}/{pg}hmesh_{NTag}tag_{suffix}.p'):
        create_mesh(NTag, pairagraph=pairagraph)
    with open(f'{pg}data{c.bin_sizes}/{pg}xmesh_{suffix}.p', 'rb') as xfile:
        xmesh = pickle.load(xfile)
    with open(f'{pg}data{c.bin_sizes}/{pg}ymesh_{suffix}.p', 'rb') as yfile:
        ymesh = pickle.load(yfile)
    with open(f"{pg}data{c.bin_sizes}/{pg}hmesh_{NTag}tag_{suffix}.p",'rb') as hfile:
        hmesh = pickle.load(hfile)
    return xmesh, ymesh, hmesh

def load_kriging(NTag, uk_kwargs, pairagraph=False):
    pg = "PG_" if pairagraph else ""
    suffix = get_kriging_suffix(uk_kwargs)
    zfilename = f"{pg}data{c.bin_sizes}/{pg}kriging_{suffix}_{NTag}b_z.p"
    vfilename = f"{pg}data{c.bin_sizes}/{pg}kriging_{suffix}_{NTag}b_v.p"
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
        return None, None

def save_kriging(NTag, uk_kwargs, zpred_grid, variance_grid, n_indices=None, pairagraph=False):
    pg = "PG_" if pairagraph else ""
    suffix = get_kriging_suffix(uk_kwargs)
    file_title = f"{pg}kriging_{suffix}_{NTag}b"
    if n_indices is not None:
        file_title += f"_n{n_indices}"
    with open(f"{pg}data{c.bin_sizes}/{file_title}_z.p", 'wb') as zfile:
        pickle.dump(zpred_grid, zfile)
    with open(f"{pg}data{c.bin_sizes}/{file_title}_v.p", 'wb') as vfile:
        pickle.dump(variance_grid, vfile)

def load_1d(NTag=4, pairagraph=False):
    xmesh, ymesh, hmesh = load_mesh(NTag, pairagraph=pairagraph)
    return xmesh.flatten(), ymesh.flatten(), hmesh.flatten()

if __name__ == "__main__":
    load_1d(NTag=2)