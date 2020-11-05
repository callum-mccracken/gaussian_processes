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
import binning
import plot_functions

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

# Have function to integrate mhh in each bin of the fullmassplane
def integrate_mhh(df):
    row_list = []
    for xi in tqdm(xbins):
        for yi in ybins:
            row_list.append({"mh1":xi,"mh2":yi,"pdf":sum(df.loc[ (df["mh1"]==xi) & (df["mh2"]==yi),"pdf"])})
    return pandas.DataFrame(row_list)
# Integrates the fullmassplane to get slices of mhh
def integrate_fmp(df):
    row_list = []
    for mhh in mhhbins[:-1]:
        row_list.append({"mhh":mhh,"pdf":sum(df.loc[df["mhh"]==mhh,"pdf"])})
    return pandas.DataFrame(row_list)

NTag = 2
epochs = 30
training = True
n_mhhbins = 20

# Size of the signal region, in bins
# The main tunable bin size parameters
# Fit is pretty sensitive to this. Worth experimenting.
# Hypothesis: NN should do well with a very large number of bins
for NxbinsInSig in [25]: #range(25,251,5):
    for NybinsInSig in [NxbinsInSig]:  # consider only square bins for now
        # The differing bin sizes did terribly with the regression
        #mhhbins = np.array([150, 250, 262, 275, 288, 302, 317, 332, 348, 365, 383, 402, 422, 443, 465, 488, 512, 
        #                  537, 563, 591, 620, 651, 683, 717, 752, 789, 828, 869, 912, 957,1004])#, 1054, 1106, 1161, 
        #                  1219, 1279, 1342, 1409, 1479, 1552, 1629, 1710, 1795, 1884, 1978, 2076])
        mhhbins = np.linspace(200,1000,n_mhhbins)

        # pandas df with 3 columns: m_h1, m_h2, and m_hh
        df = pandas.read_pickle("data/data_"+str(NTag)+"tag.p")
        print(len(df),"Events")

        """ This is just a bunch of stuff to ensure that no bins are partly in the
        signal region. This is necessary so the fit isn't biased right at the edge of the SR.
        That would be terrible."""

        # First, make histogram with signal region box:
        # The exact edges of our signal region, computed analytically
        sxmin,sxmax=103.45,142.86
        symin,symax=94.82,130.95

        # Approximate min/max values in the massplane. Chooses closest bin edge to this boundary
        # Based on extent of the background. Might be able to tune a little
        xmin,xmax = 50,250 
        ymin,ymax = 40,200

        xbinSize = (sxmax-sxmin)/NxbinsInSig
        ybinSize = (symax-symin)/NybinsInSig
        xbins = np.arange(sxmin-int((sxmin-xmin)/xbinSize)*xbinSize,int(xmax/xbinSize)*xbinSize,xbinSize)
        ybins = np.arange(symin-int((symin-ymin)/ybinSize)*ybinSize,int(ymax/ybinSize)*ybinSize,ybinSize)

        # Now make the 3D histogram
        coord_array = np.array(df[["m_h1","m_h2","m_hh"]])

        hist3d,[xbins,ybins,mhhbins] = np.histogramdd(coord_array,[xbins,ybins,mhhbins])
        xv,yv,zv = np.meshgrid(xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')

        data_df = pandas.DataFrame()
        data_df["mh1"] = xv.flatten()
        data_df["mh2"] = yv.flatten()
        data_df["mhh"] = zv.flatten()
        data_df["pdf"] = hist3d.flatten()


        # Checks if any corners of the bin are in the SR
        # Assumes giving the lower left corner of the bin
        GridBins = data_df[["mh1","mh2","mhh"]]

        # Filter out the SR bins
        if NTag == 2:
            data_dfSR = data_df.loc[binning.binInSR(data_df["mh1"],data_df["mh2"])]
        data_df = data_df.loc[~binning.binInSR(data_df["mh1"],data_df["mh2"])]
        print(len(data_df),"data points")

        ####################
        # Moving on to the ML parts
        ####################


        ModelName = f"models/model_{NTag}b_20_288_384_416_512_192_{epochs}e_{NxbinsInSig}x{NybinsInSig}_poisson_{n_mhhbins}mhh"

        # Now lets make the regression model and train
        def build_model():
            model = Sequential()

            # tuned to be better
            model.add(Dense(20, input_dim=3, activation='relu'))
            model.add(Dense(288, activation='relu'))
            model.add(Dense(384, activation='relu'))
            model.add(Dense(416, activation='relu'))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(192, activation='relu'))

            # works well enough, runs quickly
            #model.add(Dense(10, input_dim=3, activation='relu'))
            #model.add(Dense(50, activation='relu'))
            #model.add(Dense(50, activation='relu'))
            #model.add(Dense(50, activation='relu'))

            model.add(Dense(1, activation='exponential')) # Poisson loss requires output > 0 always
            model.compile(loss='poisson', optimizer='adam')
            print('built model')
            return model

        # Shuffle data points, so no training biasing/validation biasing
        np.random.seed(1234)
        #data_df = data_df.sample(frac=1).reset_index(drop=True)

        # separate into training and validation
        df_no_VR = data_df.loc[~binning.binInVR(data_df["mh1"], data_df["mh2"])]
        df_VR = data_df.loc[binning.binInVR(data_df["mh1"], data_df["mh2"])]
        print(df_VR.shape)

        X = df_no_VR[["mh1","mh2","mhh"]]
        Y = df_no_VR["pdf"]

        X_val = df_VR[["mh1","mh2","mhh"]]
        Y_val = df_VR["pdf"]


        # Plot the massplane without SR before training
        if False:
            modeldffmp = integrate_mhh(data_df)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xmesh = np.array(modeldffmp["mh1"]).reshape((len(xbins),len(ybins))).transpose()
            ymesh = np.array(modeldffmp["mh2"]).reshape((len(xbins),len(ybins))).transpose()
            hmesh = np.array(modeldffmp["pdf"]).reshape((len(xbins),len(ybins))).transpose()

            pickle.dump(xmesh,open(f"xmesh.p",'wb'))
            pickle.dump(ymesh,open(f"ymesh.p",'wb'))
            pickle.dump(hmesh,open(f"hmesh.p",'wb'))

            ax.pcolormesh(xmesh,ymesh,hmesh, cmap='bwr')
            plot_functions.plotXhh()
            plt.xlabel("$m_{h1}$")
            plt.ylabel("$m_{h2}$")
            plt.title(f"Massplane, using data, NTag={NTag}")
            plt.savefig(f"fullmassplane_{NTag}tag_data.png")
            plt.cla();plt.clf()
            plt.close()
            #plt.show()


        # Scale data to make NN easier
        scaler = MinMaxScaler()
        scaler.fit(X)
        print(X.shape, X_val.shape)
        X_scaled = scaler.transform(X)
        X_val_scaled = scaler.transform(X_val)
        pickle.dump(scaler,open("MinMaxScaler4b.p",'wb'))
        print('data ready for nn')

        if training:
            model = build_model()
            print(model.summary())
            history = model.fit(
                X_scaled, Y, validation_data = [X_val_scaled, Y_val],
                epochs=epochs, validation_split = 0.1, verbose=1, shuffle=True)
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.legend(["Training Set","Validation"])
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.savefig(ModelName+"_Loss.png")
            plt.cla();plt.clf()
            #plt.show()
            plt.close()

            model.save(ModelName)
        else:
            model = keras.models.load_model(ModelName)
            print(model.summary())
            print('model opened')


        modeldf = GridBins
        modeldf["pdf"] = model.predict(scaler.transform(GridBins), verbose=1)
        pickle.dump(modeldf["pdf"],open(f"model_output_{NTag}b.p",'wb'))
        print('predictions made')


        # Plot the predicted massplane
        modeldffmp = integrate_mhh(modeldf)
        print('m_hh integrated')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xmesh = np.array(modeldffmp["mh1"]).reshape((len(xbins),len(ybins))).transpose()
        ymesh = np.array(modeldffmp["mh2"]).reshape((len(xbins),len(ybins))).transpose()
        hmesh = np.array(modeldffmp["pdf"]).reshape((len(xbins),len(ybins))).transpose()
        ax.pcolormesh(xmesh,ymesh,hmesh)
        plot_functions.plotXhh()
        plt.xlabel("$m_{h1}$")
        plt.ylabel("$m_{h2}$")
        plt.title(f"{ModelName} Model-predicted massplane, NTag={NTag}")
        plt.savefig(ModelName+"_fullmassplane.png")
        plt.cla();plt.clf()
        #plt.show()
        plt.close()
        print('plotted predicted massplane')

        # Plot the predicted SR mhh
        if NTag==2:
            modeldfSR = modeldf.loc[binning.binInSR(modeldf["mh1"],modeldf["mh2"])]
            modelmhh = list(integrate_fmp(modeldfSR)["pdf"])
            datamhh = list(integrate_fmp(data_dfSR)["pdf"])

            fig,axs = plt.subplots(2,1)
            gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
            gs.update(hspace=0)

            ax = plt.subplot(gs[0])
            ax.step(mhhbins,modelmhh+[modelmhh[-1]],'r',linewidth=2,where='post')
            XData = mhhbins[:-1]+(mhhbins[1]-mhhbins[0])/2
            ax.errorbar(XData,datamhh,yerr=np.sqrt(datamhh),fmt='k.')
            ax.set_ylabel("Counts")
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.legend(["2b SR NN Regression","2b SR Data"])

            ratio = [m/d          if d>0 else 100 for m,d in zip(modelmhh,datamhh)]
            err =   [r/np.sqrt(d) if d>0 else 0   for r,d in zip(ratio,datamhh)]
            ax = plt.subplot(gs[1])
            print(ratio)
            ax.errorbar(XData,ratio,yerr=err,fmt='k.')
            ax.plot([mhhbins[0],mhhbins[-1]],[1,1],'k--',linewidth=1)
            ax.set_ylim(0.75,1.25)
            ax.set_ylim(0.9,1.1)
            ax.set_xlabel("$m_{hh}$"+" (GeV)")
            ax.set_ylabel("Pred./Data")
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(25))
            plt.savefig(ModelName+"mhhSR.png")
            #plt.show()
            plt.close()
        else:
            modeldfSR = modeldf.loc[binning.binInSR(modeldf["mh1"],modeldf["mh2"])]
            modelmhh = list(integrate_fmp(modeldfSR)["pdf"])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.step(mhhbins,modelmhh+[modelmhh[-1]],'r',linewidth=2,where='post')
            ax.set_xlabel("$m_{hh}$"+" (GeV)")
            ax.set_ylabel("Counts")
            ax.legend(["4b SR NN Regression"])
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(25))
            plt.savefig(ModelName+"mhhSR.png")
            plt.cla();plt.clf()
            #plt.show()
            plt.close()


        # Plot massplane and 3D histogram
        extra_plotting=False
        if extra_plotting:
            # Just plot the massplane from data
            plt.hist2d(df["m_h1"],df["m_h2"],[xbins,ybins],[[0,300],[0,300]])
            plot_functions.plotXhh()
            plt.savefig("Fullmassplane.png")
            plt.close()

            # Make a 3D histogram of mh1,mh2,mhh
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Make custom colormap so alpha value is 0 at min of range
            cmap = plt.cm.get_cmap("Greys")
            alpha_cmap = cmap(np.arange(cmap.N))
            alpha_cmap[:,-1] = np.linspace(0, 1, cmap.N)
            alpha_cmap = ListedColormap(alpha_cmap)
            colors = [x**(0.5) for x in hist3d.flatten()] # Nonlinear scaling to colors so nonzero regions more visible
            img = ax.scatter(xv, yv, zv, c=colors, cmap=alpha_cmap, marker='.')

            ax.set_xlabel("$m_{h1}$")
            ax.set_ylabel("$m_{h2}$")
            ax.set_zlabel("$m_{hh}$")
            plt.show()
            plt.close()
