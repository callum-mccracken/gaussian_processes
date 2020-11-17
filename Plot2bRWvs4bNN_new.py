import pandas
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import sys
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

from binning import binInSR, binInVR
from plot_functions import plotXhh
import constants as c
import kriging
import deal_with_files

# Have function to integrate mhh in each bin of the fullmassplane
def integrate_mhh(df, xbins, ybins):
    row_list = []
    for xi in tqdm(xbins):
        for yi in ybins:
            row_list.append({"mh1":xi,"mh2":yi,"pdf":sum(df.loc[ (df["mh1"]==xi) & (df["mh2"]==yi),"pdf"])})
    return pandas.DataFrame(row_list)

# Integrates the fullmassplane to get slices of mhh
def integrate_fmp(df):
    row_list = []
    for mhh in c.mhhbins[:-1]:
        row_list.append({"mhh":mhh,"pdf":sum(df.loc[df["mhh"]==mhh,"pdf"])})
    return pandas.DataFrame(row_list)

def plot(x, y, h, name=None, pairagraph=False):
    """plot a pcolormesh of h, at points given by x, y"""
    pg = "_PG" if pairagraph else ""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.pcolormesh(x,y,h, shading='auto')
    plotXhh()
    plt.xlabel("$m_{h1}$")
    plt.ylabel("$m_{h2}$")
    if name is not None:
        plt.savefig(pg + name)
    #plt.show()
    plt.close()
    return h

def make_all_plots(method, ModelName=None, uk_kwargs=None, pairagraph=False, dim=2):
    pg = "PG_" if pairagraph else ""
    if method == "GP":
        suffix = deal_with_files.get_kriging_suffix(uk_kwargs)
        ModelName = f"{pg}figures{c.bin_sizes}/{pg}{dim}d_kriging_{suffix}"
    else:
        assert ModelName is not None

    df = pandas.read_pickle(f"data/{pg}data_2tag_full.p")
    coord_array = np.array(df[["m_h1","m_h2","m_hh"]])
    NORM = 1.0246291
    weights = NORM*np.array(df["NN_d24_weight_bstrap_med_17"])
    xbins = np.linspace(min(c.xbins), max(c.xbins), 200)
    ybins = np.linspace(min(c.ybins), max(c.ybins), 200)
    hist3d,[xbins,ybins,mhhbins] = np.histogramdd(
        coord_array,[xbins,ybins,c.mhhbins],weights=weights)
    mh1,mh2,mhh = np.meshgrid(xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')
    grid_shape = (len(xbins),len(ybins))

    data_df = pandas.DataFrame()
    data_df["mh1"] = mh1.flatten()
    data_df["mh2"] = mh2.flatten()
    data_df["mhh"] = mhh.flatten()
    data_df["pdf"] = hist3d.flatten()

    GridBins = data_df[["mh1","mh2","mhh"]]

    data_df_SR = data_df.loc[binInSR(data_df["mh1"],data_df["mh2"])]
    data_mhh = list(integrate_fmp(data_df_SR)["pdf"])

    if method == "NN":
        # OK 2b reweighted is loaded
        # Now load model and make prediction df over GridBins
        model = keras.models.load_model(ModelName)
        scaler = pickle.load(open("MinMaxScaler4b.p",'rb'))
        if "2b4b" in ModelName:
            # we want to get predictions as if this were 4b data
            data_df["ntag"] = np.array([4]*len(data_df))
            GridBins = data_df[["mh1","mh2","mhh", 'ntag']]
            scaler = pickle.load(open("MinMaxScaler2b4b.p",'rb'))

        # even if 2b4b model, we're only simulating NTag=4 at this point
        # and only considering points within the SR
        predicted_df = GridBins
        predicted_df["pdf"] = model.predict(scaler.transform(GridBins), verbose=1)
        predicted_df_SR = predicted_df.loc[
            binInSR(predicted_df["mh1"], predicted_df["mh2"])]
        predicted_mhh = list(integrate_fmp(predicted_df_SR)["pdf"])

        predicted_fmp = integrate_mhh(predicted_df, xbins, ybins)
        xmesh = np.array(predicted_fmp["mh1"]).reshape(grid_shape).transpose()
        ymesh = np.array(predicted_fmp["mh2"]).reshape(grid_shape).transpose()
        hmesh = np.array(predicted_fmp["pdf"]).reshape(grid_shape).transpose()
    elif method == "GP":
        mh1_flat = mh1.flatten()
        mh2_flat = mh2.flatten()
        mhh_flat = mhh.flatten()
        pdf_flat = hist3d.flatten()
        if dim == 2:
            hmesh, _ = kriging.get_kriging_prediction_2d(4, mh1_flat, mh2_flat, pdf_flat, uk_kwargs=uk_kwargs, pairagraph=pairagraph)
        if dim == 3:
            hmesh, _ = kriging.get_kriging_prediction_3d(4, mh1_flat, mh2_flat, mhh_flat, pdf_flat, uk_kwargs=uk_kwargs, pairagraph=pairagraph)

        xmesh = mh1[:,:,0]
        ymesh = mh2[:,:,0]

        hmesh_resized = np.empty((*hmesh.transpose().shape, c.n_mhhbins))
        mhh_counts = []
        for i in range(c.n_mhhbins-1):
            hmesh_resized[:,:,i] = hmesh.transpose()
            pdf = hmesh_resized[:-1,:-1,i]
            pdf = pdf / np.sum(pdf)
            count = np.sum(pdf * hist3d[:,:,i])
            mhh_counts.append(count)
        mhh_counts = np.array(mhh_counts)
        predicted_df = pandas.DataFrame()
        predicted_df["mh1"] = mh1_flat
        predicted_df["mh2"] = mh2_flat
        predicted_df["mhh"] = mhh_flat
        predicted_df["pdf"] = hmesh_resized[:-1,:-1,:-1].flatten()
        predicted_df_SR = predicted_df.loc[
            binInSR(predicted_df["mh1"], predicted_df["mh2"])]

        # for predicted massplane, scale
        predicted_mhh = mhh_counts 



    # Plot predicted massplane
    plot(xmesh, ymesh, hmesh.transpose()[:-1,:-1], name=ModelName+"_fullmassplane_4b_pred.png")

    # Plot 2b reweighted massplane
    hmesh_2brw = np.array(integrate_mhh(
        data_df, xbins, ybins)["pdf"]).reshape((len(xbins),len(ybins))).transpose()

    plot(xmesh, ymesh, hmesh_2brw.transpose()[:-1,:-1], name=ModelName+"_fullmassplane_2brw.png")

    # Plot the ratio
    massplane_scale_factor = np.sum(hmesh_2brw) / np.sum(hmesh)
    hmesh *= massplane_scale_factor
    with np.errstate(divide='ignore', invalid='ignore'):
        hmesh_ratio = hmesh / hmesh_2brw
    hmesh_ratio[np.isnan(hmesh_ratio)] = 0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(xmesh,ymesh,hmesh_ratio.transpose()[:-1,:-1],
                    vmin=0.8,vmax=1.4, cmap='bwr', shading='auto')
    fig.colorbar(im, ax=ax)
    plotXhh()
    plt.xlabel("$m_{h1}$")
    plt.ylabel("$m_{h2}$")
    plt.title("Ratio of (4b prediction)/2bRW")
    plt.savefig(ModelName+"_fullmassplane_NNOver2bRW.png")
    #plt.show()
    plt.close()

    # Plot mhh
    mhh_scale_factor = sum(data_mhh) / sum(mhh_counts)
    predicted_mhh *= mhh_scale_factor

    fig,_ = plt.subplots(2,1)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    gs.update(hspace=0)

    ax = plt.subplot(gs[0])
    ax.step(mhhbins,list(predicted_mhh) + [predicted_mhh[-1]],'r',linewidth=2,where='post')
    XData = mhhbins[:-1]+(mhhbins[1]-mhhbins[0])/2
    ax.errorbar(XData,data_mhh,yerr=np.sqrt(data_mhh),fmt='k.')
    ax.set_ylabel("Counts")
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.legend([f"4b SR {method} Regression","2b Reweighted"])

    ratio = [m/d          if d>0 else 100 for m,d in zip(predicted_mhh,data_mhh)]
    err =   [r/np.sqrt(d) if d>0 else 0   for r,d in zip(ratio,data_mhh)]
    ax = plt.subplot(gs[1])
    ax.errorbar(XData,ratio,yerr=err,fmt='k.')
    ax.plot([mhhbins[0],mhhbins[-1]],[1,1],'k--',linewidth=1)
    ax.set_ylim(0.75,1.25)
    #ax.set_ylim(0.9,1.1)
    ax.set_xlabel("$m_{hh}$"+" (GeV)")
    ax.set_ylabel("$\\frac{Regression}{Reweighting}$")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    plt.savefig(ModelName+"_mhhSR.png")
    plt.close()
    #plt.show()

if __name__ == "__main__":
    method = 'GP'
    # model path for NN models, naming string for figures for kriging
    #ModelName = "models/model_20_288_384_416_512_192_30e_25x25_poisson"
    #for s in [800,900,1000,1100]:
    #    for r in [20,40,60,80,100,120,140,160]:
    #        for n in [1e-12, 1e-11,1e-10,1e-9,1e-8]:
    s = 800
    r = 160
    n = 1e-8
    pairagraph=False
    dim=2
    uk_kwargs = {
        "variogram_model": "gaussian",
        'exact_values': True,
        'variogram_parameters': {'sill': s, 'range': r, 'nugget': n}
    }
    make_all_plots(method, uk_kwargs=uk_kwargs, pairagraph=pairagraph, dim=dim)
