import pandas
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import sys
import os
from tqdm import tqdm
import constants as c
import plot_functions

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

import constants as c
import binning

suffix = f"{c.NxbinsInSig}_{c.NybinsInSig}_{c.n_mhhbins}"


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

def create_mesh(NTag=4):
    print('creating mesh')
    
    # pandas df with 3 columns: m_h1, m_h2, and m_hh
    df = pandas.read_pickle(f"data/data_{NTag}tag.p")
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

    with open(f"data/xmesh_{suffix}.p",'wb') as xfile:
        pickle.dump(xmesh,xfile)
    with open(f"data/ymesh_{suffix}.p",'wb') as yfile:
        pickle.dump(ymesh,yfile)
    with open(f"data/hmesh_{NTag}tag_{suffix}.p",'wb') as hfile:
        pickle.dump(hmesh,hfile)
    with open(f"data/hmesh_{NTag}tag_{suffix}.p",'rb') as hfile:
        hmesh = pickle.load(hfile)

def load_mesh(NTag=4, bins=None):
    if not os.path.exists(f'data/hmesh_{NTag}tag_{suffix}.p'):
        create_mesh(NTag)
    with open(f'data/xmesh_{suffix}.p', 'rb') as xfile:
        xmesh = pickle.load(xfile)
    with open(f'data/ymesh_{suffix}.p', 'rb') as yfile:
        ymesh = pickle.load(yfile)
    with open(f"data/hmesh_{NTag}tag_{suffix}.p",'rb') as hfile:
        hmesh = pickle.load(hfile)
    return xmesh, ymesh, hmesh

def load_1d(NTag=4):
    xmesh, ymesh, hmesh = load_mesh(NTag)
    return xmesh.flatten(), ymesh.flatten(), hmesh.flatten()

if __name__ == "__main__":
    load_1d(NTag=2)