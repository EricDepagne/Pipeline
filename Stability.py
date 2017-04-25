#!/usr/bin/env python
# -*- coding: utf-8 -*-

# sys imports

# python imports
from glob import glob

# numpy imports
import numpy as np

# astropy imports
from astropy.io import fits
from astropy.modeling import fitting, models

# scipy imports
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt

# matplotlib imports
import matplotlib.pylab as plt


def find_thar_files():
    l = []
    ffile = glob('*.fits')
    for f in ffile:
        with fits.open(f) as fh:
            if 'STABLE' in fh[0].header['PROPID']:
                l.append(f)
    return l


def fit_orders_pair(arcfile):
    plt.clf()
    print('Arc file : {file}'.format(file=arcfile))
    ff = fits.open(arcfile)
    center = ff[0].header['NAXIS1']/2
    chip = ff[0].header['DETNAM']
    parameters = {
            'HBDET': {
                'Level': 800,
                'Distance': 30
                    },
            'HRDET': {
                'Level': 1300,
                'Distance': 40
                    }
                }
    cut = ff[0].data[:, int(center)]
    cutfiltered = savgol_filter(cut, 11, 3)
    peaks = find_peaks_cwt(cutfiltered, widths=np.arange(1, 20))
    goodpeaks = []
    for i in range(peaks.shape[0]-1, 0, -1):
        if cutfiltered[peaks[i]] < parameters[chip]['Level'] or (peaks[i]-peaks)[i-1] > parameters[chip]['Distance']:  # We find doublets of peaks, in order to fit both sky and object at the same time
            # print('Wrong !{0}, {1}, {2}'.format(i, peaks[i], peaks[i-1]))
            continue
        # print(i, peaks[i], cutfiltered[peaks[i]])
        goodpeaks.append(i)
    print('peaks : {goodpeaks}'.format(goodpeaks=goodpeaks))
    plt.plot(cutfiltered)
    plt.scatter(peaks[goodpeaks], cutfiltered[peaks[goodpeaks]], c='green')
    for i in range(len(goodpeaks)):
        xg = np.arange(peaks[goodpeaks][i]-50, peaks[goodpeaks][i]+20)
        plt.plot(xg, cutfiltered[peaks[goodpeaks][i]-50:peaks[goodpeaks][i]+20])
        g1 = models.Gaussian1D(amplitude=1., mean=peaks[goodpeaks[i]], stddev=5)
        g2 = models.Gaussian1D(amplitude=1., mean=peaks[goodpeaks[i]]-30, stddev=5)
        gg_init = g1 + g2
        fitter = fitting.SLSQPLSQFitter()
    # Pour fitter les deux gaussiennes, il faut normaliser les flux à 1
        gg_fit = fitter(gg_init, xg, cutfiltered[xg]/cutfiltered[peaks[goodpeaks[i]]], verblevel=0)
        plt.plot(xg, cutfiltered[peaks[goodpeaks[i]]]*gg_fit(xg))

    # return goodpeaks


def fit_dual_orders(arcfile):
    pass


def find_order(cut, center_order):
    i = 0
# TODO : toruver mieux que 725 écrit en dur!
    while cut[i+center_order] > 725:
        i -= 1
    inf = i
    i = 0
    while cut[i+center_order] > 725:
        i += 1
    sup = i

    print('bornes : {inf}, {sup}, {i}'.format(inf=inf, sup=sup, i=i))


def assess_stability():
    arclist = find_thar_files()
    print(arclist)

if __name__ == "__main__":
    assess_stability()
