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

def plotXhh(mh1_0, mh2_0, r, Xhh_cut, mh1_min, mh1_max, mh2_min, mh2_max, color):
    mh1, mh2 = sp.symbols('mh1 mh2')
    sg_expr = ((mh1-mh1_0)/(r*mh1))**2 + ((mh2-mh2_0)/(r*mh2))**2
    sg_eq = sp.Eq(sg_expr, Xhh_cut**2)
    plot = sp.plot_implicit(sg_eq, 
                            x_var = (mh1,mh1_min,mh1_max),
                            y_var = (mh2,mh2_min,mh2_max),
                            show = False,
                            axis_center = (mh1_min,mh2_min))
    x,y = zip(*[(x_int.mid, y_int.mid) for x_int, y_int in plot[0].get_points()[0]])
    x,y = list(x),list(y)
    plt.plot(x,y,'.',markersize=0.5,color=color)
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
mh10,mh20=120,110
def binInSR(x,y):
    xs = x + xbinSize
    ys = y + ybinSize
    return ((0.0256 > (x -mh10)**2/x**2 +(y-mh20)**2/y**2) |
            (0.0256 > (xs-mh10)**2/xs**2+(y-mh20)**2/y**2) |
            (0.0256 > (x-mh10)**2/x**2+(ys-mh20)**2/ys**2) |
            (0.0256 > (xs-mh10)**2/xs**2+(ys-mh20)**2/ys**2))

# In VR if no corner of a bin is in the SR, and at least one corner is in the VR
def binInVR(x,y):
    xs = x + xbinSize
    ys = y + ybinSize
    return (~binInSR(x,y) &
            (((x-123.6)**2 +(y-113.3)**2<900) |
             ((xs-123.6)**2+(y-113.3)**2<900) |
             ((x-123.6)**2+(ys-113.3)**2<900) | 
             ((xs-123.6)**2+(ys-113.3)**2<900))
           )
    

# Shenanigans to set up bins as close as possible to the SR
NxbinsInSig=25
NybinsInSig=25
mhhbins = np.linspace(200,1000,50)
sxmin,sxmax=103.45,142.86
symin,symax=94.82,130.95
xmin,xmax = 50,250 
ymin,ymax = 40,200
xbinSize = (sxmax-sxmin)/NxbinsInSig
ybinSize = (symax-symin)/NybinsInSig
xbins = np.arange(sxmin-int((sxmin-xmin)/xbinSize)*xbinSize,int(xmax/xbinSize)*xbinSize,xbinSize)
ybins = np.arange(symin-int((symin-ymin)/ybinSize)*ybinSize,int(ymax/ybinSize)*ybinSize,ybinSize)

df = pandas.read_pickle("data/data_2tag_full.p")
coord_array = np.array(df[["m_h1","m_h2","m_hh"]])
NORM = 1.0246291
weights = NORM*np.array(df["NN_d24_weight_bstrap_med_17"])
hist3d,[xbins,ybins,mhhbins] = np.histogramdd(coord_array,[xbins,ybins,mhhbins],weights=weights)
xv,yv,zv = np.meshgrid(xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')
data_df = pandas.DataFrame()
data_df["mh1"] = xv.flatten()
data_df["mh2"] = yv.flatten()
data_df["mhh"] = zv.flatten()
data_df["pdf"] = hist3d.flatten()

# OK 2b reweighted is loaded
# Now load model and make prediction df over GridBins
ModelName = "models/model_20_288_384_416_512_192_30e_25x25_poisson"
model = keras.models.load_model(ModelName)
GridBins = data_df[["mh1","mh2","mhh"]]
scaler = pickle.load(open("MinMaxScaler4b.p",'rb'))
if "2b4b" in ModelName:
    # we want to get predictions of 4b data
    data_df["ntag"] = np.array([4]*len(data_df))
    GridBins = data_df[["mh1","mh2","mhh", 'ntag']]
    scaler = pickle.load(open("MinMaxScaler2b4b.p",'rb'))

# even if 2b4b model, we're only simulating NTag=4 at this point

modeldf = GridBins
modeldf["pdf"] = model.predict(scaler.transform(GridBins), verbose=1)

modeldfSR = modeldf.loc[binInSR(modeldf["mh1"],modeldf["mh2"])]
modelmhh = list(integrate_fmp(modeldfSR)["pdf"])
data_dfSR = data_df.loc[binInSR(data_df["mh1"],data_df["mh2"])]
datamhh = list(integrate_fmp(data_dfSR)["pdf"])

# Plot predicted massplane
modeldffmp = integrate_mhh(modeldf)
fig = plt.figure()
ax = fig.add_subplot(111)
xmesh = np.array(modeldffmp["mh1"]).reshape((len(xbins),len(ybins))).transpose()
ymesh = np.array(modeldffmp["mh2"]).reshape((len(xbins),len(ybins))).transpose()
hmesh = np.array(modeldffmp["pdf"]).reshape((len(xbins),len(ybins))).transpose()
hmesh_2brw = hmesh
ax.pcolormesh(xmesh,ymesh,hmesh)
plotXhh(120,110,0.1,1.6,sxmin,sxmax,symin,symax,'r')
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.savefig(ModelName+"_fullmassplane_4bNN.png")
#plt.show()
plt.close()

# Plot 2b reweighted massplane
modeldffmp = integrate_mhh(data_df)
fig = plt.figure()
ax = fig.add_subplot(111)
xmesh = np.array(modeldffmp["mh1"]).reshape((len(xbins),len(ybins))).transpose()
ymesh = np.array(modeldffmp["mh2"]).reshape((len(xbins),len(ybins))).transpose()
hmesh = np.array(modeldffmp["pdf"]).reshape((len(xbins),len(ybins))).transpose()
ax.pcolormesh(xmesh,ymesh,hmesh)
plotXhh(120,110,0.1,1.6,sxmin,sxmax,symin,symax,'r')
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.savefig(ModelName+"_fullmassplane_2brw.png")
#plt.show()
plt.close()

# Plot the ratio
with np.errstate(divide='ignore', invalid='ignore'):
    hmesh_ratio = hmesh/hmesh_2brw
hmesh_ratio[np.isnan(hmesh_ratio)] = 0
fig = plt.figure()
ax = fig.add_subplot(111)
im=ax.pcolormesh(xmesh,ymesh,hmesh_ratio,vmin=0.8,vmax=1.4, cmap='bwr')
fig.colorbar(im, ax=ax)
plotXhh(120,110,0.1,1.6,sxmin,sxmax,symin,symax,'r')
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.savefig(ModelName+"_fullmassplane_NNOver2bRW.png")
#plt.show()
plt.close()

# Plot mhh
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
ax.legend(["4b SR NN Regression","2b Reweighted"])

ratio = [m/d          if d>0 else 100 for m,d in zip(modelmhh,datamhh)]
err =   [r/np.sqrt(d) if d>0 else 0   for r,d in zip(ratio,datamhh)]
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
