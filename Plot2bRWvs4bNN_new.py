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
def integrate_mhh(df):
    row_list = []
    for xi in tqdm(c.xbins):
        for yi in c.ybins:
            row_list.append({"mh1":xi,"mh2":yi,"pdf":sum(df.loc[ (df["mh1"]==xi) & (df["mh2"]==yi),"pdf"])})
    return pandas.DataFrame(row_list)

# Integrates the fullmassplane to get slices of mhh
def integrate_fmp(df):
    row_list = []
    for mhh in c.mhhbins[:-1]:
        row_list.append({"mhh":mhh,"pdf":sum(df.loc[df["mhh"]==mhh,"pdf"])})
    return pandas.DataFrame(row_list)

def plot(x, y, h, name=None):
    """plot a pcolormesh of h, at points given by x, y"""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.pcolormesh(x,y,h, shading='auto')
    plotXhh()
    plt.xlabel("$m_{h1}$")
    plt.ylabel("$m_{h2}$")
    if name is not None:
        plt.savefig(name)
    #plt.show()
    plt.close()
    return h

def make_all_plots(method, ModelName=None, uk_kwargs=None):

    if method == "GP":
        suffix = deal_with_files.get_kriging_suffix(uk_kwargs)
        ModelName = f"figures{c.bin_sizes}/kriging_{suffix}"
    else:
        assert ModelName is not None

    df = pandas.read_pickle("data/data_2tag_full.p")
    coord_array = np.array(df[["m_h1","m_h2","m_hh"]])
    NORM = 1.0246291
    weights = NORM*np.array(df["NN_d24_weight_bstrap_med_17"])
    hist3d,[xbins,ybins,mhhbins] = np.histogramdd(
        coord_array,[c.xbins,c.ybins,c.mhhbins],weights=weights)
    xv,yv,zv = np.meshgrid(xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')
    grid_shape = (len(xbins),len(ybins))

    data_df = pandas.DataFrame()
    data_df["mh1"] = xv.flatten()
    data_df["mh2"] = yv.flatten()
    data_df["mhh"] = zv.flatten()
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

        predicted_df = GridBins
        predicted_df["pdf"] = model.predict(scaler.transform(GridBins), verbose=1)
        predicted_df_SR = predicted_df.loc[
            binInSR(predicted_df["mh1"], predicted_df["mh2"])]
        predicted_mhh = list(integrate_fmp(predicted_df_SR)["pdf"])

        predicted_fmp = integrate_mhh(predicted_df)
        xmesh = np.array(predicted_fmp["mh1"]).reshape(grid_shape).transpose()
        ymesh = np.array(predicted_fmp["mh2"]).reshape(grid_shape).transpose()
        hmesh = np.array(predicted_fmp["pdf"]).reshape(grid_shape).transpose()
    elif method == "GP":
        x = xv.flatten()
        y = yv.flatten()
        z = hist3d.flatten()
        hmesh, _ = kriging.get_kriging_prediction(4, x, y, z, uk_kwargs=uk_kwargs)
        xmesh = xv[:,:,0]
        ymesh = yv[:,:,0]

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
        predicted_df["mh1"] = xv.flatten()
        predicted_df["mh2"] = yv.flatten()
        predicted_df["mhh"] = zv.flatten()
        predicted_df["pdf"] = hmesh_resized[:-1,:-1,:-1].flatten()
        predicted_df_SR = predicted_df.loc[
            binInSR(predicted_df["mh1"], predicted_df["mh2"])]

        predicted_mhh = mhh_counts*150

    # Plot predicted massplane
    hT = hmesh.transpose()[:-1,:-1]
    plot(xmesh, ymesh, hT, name=ModelName+"_fullmassplane_4b_pred.png")

    # Plot 2b reweighted massplane
    hmesh_2brw = np.array(integrate_mhh(data_df)["pdf"]).reshape((len(xbins),len(ybins))).transpose()
    plot(xmesh, ymesh, hmesh_2brw.transpose()[:-1,:-1], name=ModelName+"_fullmassplane_2brw.png")

    # Plot the ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        hmesh_ratio = hmesh/hmesh_2brw
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
    for ev in [True, False]:
        for vm in ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]:
            uk_kwargs = {
                "variogram_model": vm,
                'exact_values': ev
            }
            make_all_plots(method, uk_kwargs=uk_kwargs)
