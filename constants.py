import numpy as np

# mass plane edges (by default)
xmin, xmax = 50, 250 
ymin, ymax = 40, 200

# parameters for signal region shape
m_h1_0 = 120
m_h2_0 = 110
r = 0.1
Xhh_cut = 1.6
m_h1_min = 103.45
m_h1_max = 142.86
m_h2_min = 94.82
m_h2_max = 130.95
sr_color = 'r'
vr_color = 'm'
cr_color = 'y'

# x,y boundaries of signal region
sxmin, sxmax = 103.45, 142.86
symin, symax = 94.82, 130.95

# Size of the signal region, in bins
# The main tunable bin size parameters
# Fit is pretty sensitive to this. Worth experimenting.
# Hypothesis: NN should do well with a very large number of bins
NxbinsInSig = 25
NybinsInSig = 25
n_mhhbins = 20

# bins for m_HH histogram
mhhbins = np.linspace(200,1000,n_mhhbins)

# bins for massplane
xbinSize = (sxmax - sxmin) / NxbinsInSig
ybinSize = (symax - symin) / NybinsInSig
xbins = np.arange(sxmin-int((sxmin - xmin) / xbinSize) * xbinSize,
                  int(xmax / xbinSize) * xbinSize, xbinSize)
n_xbins = len(xbins)
ybins = np.arange(symin-int((symin - ymin) / ybinSize) * ybinSize,
                  int(ymax / ybinSize) * ybinSize, ybinSize)
n_ybins = len(ybins)
