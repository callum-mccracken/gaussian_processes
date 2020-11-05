import pandas as pd
import numpy as np
import constants as c

variables = ["m_h1", "m_h2", "m_hh"]


def binInSR(x,y):
    """
    Function to check if a bin's x,y is in the Signal Region.
    Also works if x and y are arrays.
    """
    xs = x + c.xbinSize
    ys = y + c.ybinSize
    return ((0.0256 > (x - c.m_h1_0)**2/x**2 + (y - c.m_h2_0)**2/y**2) |
            (0.0256 > (xs - c.m_h1_0)**2/xs**2 + (y - c.m_h2_0)**2/y**2) |
            (0.0256 > (x - c.m_h1_0)**2/x**2 + (ys - c.m_h2_0)**2/ys**2) |
            (0.0256 > (xs - c.m_h1_0)**2/xs**2 + (ys - c.m_h2_0)**2/ys**2))

def binInVR(x,y):
    """
    In VR if no corner of a bin is in the SR,
    and at least one corner is in the VR
    """
    xs = x + c.xbinSize
    ys = y + c.ybinSize
    return (~binInSR(x,y) &
            (((x-123.6)**2 +(y-113.3)**2<900) |
             ((xs-123.6)**2+(y-113.3)**2<900) |
             ((x-123.6)**2+(ys-113.3)**2<900) | 
             ((xs-123.6)**2+(ys-113.3)**2<900))
           )


def make_histogram(pickle_file, NTag):
    """get histogram data"""

    # pandas df containg all variables in 'variables' list
    original_df = pd.read_pickle(pickle_file)

    # cast as array
    coord_array = np.array(original_df[variables])

    # make multi-dimensional histogram using data as first arg and bins as 2nd
    histNd, [xbins, ybins, mhhbins] = np.histogramdd(
        coord_array, [c.xbins, c.ybins, c.mhhbins])

    # meshgrid to get all points of the Nd histogram array
    xv, yv, zv = np.meshgrid(
        xbins[:-1], ybins[:-1], mhhbins[:-1], indexing='ij')

    df = pd.DataFrame()
    df["m_h1"] = xv.flatten()
    df["m_h2"] = yv.flatten()
    df["m_hh"] = zv.flatten()
    df["pdf"] = histNd.flatten()

    #df_no_pdf = df[["m_h1","m_h2","m_hh"]]

    # separate df into SR and non-SR DataFrames
    if NTag == 2:
        df_SR = df.loc[binInSR(df["m_h1"], df["m_h2"])]
    elif NTag == 4:
        df_SR = None
    df_no_SR = df.loc[~binInSR(df["m_h1"], df["m_h2"])]

    return df, df_no_SR, df_SR


if __name__ == "__main__":
    import plot_functions
    df, df_no_SR, df_SR = make_histogram("data_2tag.p", 2)
    df_no_VR = df.loc[~binInVR(df["m_h1"], df["m_h2"])]
    plot_functions.plot_fullmassplane_from_df(
        df_no_VR, savename='notsaved.png',
            save=False, show=True, vr=True)