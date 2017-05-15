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


def classify_files():
    files = {}
    thar = []
    bias = []
    flat = []

    ffile = glob('*.fits')
    for f in ffile:
        with fits.open(f) as fh:
            h = fh[0].header['PROPID']
            if 'STABLE' in h:
                thar.append(f)
            if 'CAL_FLAT' in h:
                flat.append(f)
            if 'BIAS' in h:
                bias.append(f)
    files.update({'ThAr': thar, 'Bias': bias, 'Flat': flat})
    return files


def find_peaks(arc):
    goodpeaks = []
    # print('dimensions : {dim}, parameters :{param}'.format(dim=arc.shape, param=parameters))
    cutfiltered = savgol_filter(arc, 11, 3)
    peaks = find_peaks_cwt(cutfiltered, widths=np.arange(1, 20))
    for i in range(peaks.shape[0]-1, 0, -1):
        if cutfiltered[peaks[i]] < parameters[parameters['chip']]['Level'] or (peaks[i]-peaks)[i-1] > parameters[parameters['chip']]['Distance']:  # We find doublets of peaks, in order to fit both sky and object at the same time
            # print('Wrong !{0}, {1}, {2}'.format(i, peaks[i], peaks[i-1]))
            continue
        # print(i, peaks[i], cutfiltered[peaks[i]])
        goodpeaks.append(i)
    return peaks[goodpeaks]


def fit_orders_pair(arcdata):
    # plt.clf()
    cut = arcdata[:, parameters['center']]
    order = {}
    # print('peaks : {goodpeaks}'.format(goodpeaks=goodpeaks))
    # plt.plot(cutfiltered)
    # plt.scatter(peaks[goodpeaks], cutfiltered[peaks[goodpeaks]], c='green')
    goodpeaks = find_peaks(cut)[::-1]
    lmax = 0
    # goodpeaks = find_peaks(cut)
# We invert the lists to start from the top of the frame.
    # goodpeaks = gps[::-1]
    print('Orders found at the following pixels {goodpeaks} using the center column of the chip'.format(goodpeaks=goodpeaks))
# Cette boucle est probablement vectorisable. ON doit pouvoir parser tous les ordres en même temps, car ils sont tous indépendants.
#
# yyg contains the pixels at the center of the chip where the orders are.
# This is the origin of the gaussian fits.
    # yyg = np.asarray([np.arange(goodpeaks[i]-50, goodpeaks[i]+20) for i in range(2, len(goodpeaks))])
    # for i in range(1,4):
    for i in range(1, len(goodpeaks)-1):
        center = parameters['center']
        nbpixperstep = parameters['nbpixperstep']  # how far from a fit do we go to fit the next.
# Computing how many steps are needed to parse the orders.
        steps = np.int((parameters['X2']-parameters['X1']-center)/nbpixperstep)
        skyorder = []
        scienceorder = []
        positions = []
        fit = []
        ordernumber = i + parameters[parameters['chip']]['OrderShift']
        print('Detecting order number {order}.'.format(order=ordernumber))
        for direction in [-1, 1]:
            yg = np.arange(goodpeaks[i]-50, goodpeaks[i]+20)
            # plt.plot(yg, cutfiltered[peaks[goodpeaks][i]-50:peaks[goodpeaks][i]+20])
            g1 = models.Gaussian1D(amplitude=1., mean=goodpeaks[i], stddev=5)
            g2 = models.Gaussian1D(amplitude=1., mean=goodpeaks[i]-30, stddev=5)
            gg_init = g1 + g2
# In the compound results, parameters with _1 are relative to the science fibre, and those with _0 are the sky fibre.
            fitter = fitting.SLSQPLSQFitter()
        # Pour fitter les deux gaussiennes, il faut normaliser les flux à 1
            gg_fit = fitter(gg_init, yg, cut[yg]/cut[yg].max(), verblevel=0)
            sci = gg_fit.mean_0
            sky = gg_fit.mean_1
            for index in range(steps+1):
                if direction == 1 and index == 0:
                    # We do not need to redo the point at the center.
                    continue
                try:
                    y = center+nbpixperstep*index*direction
                except IndexError as e:
                    print('Out of bounds')
                    break
                # print('y : {y}, step {step}, direction {direction}'.format(y=y, step=index, direction=direction))
                # print(sci.value, sky.value)
                if sci.value+20 > parameters['Y2']-1:
                    # print('Overflow at {sci} for order {order}'.format(sci=sci.value, order=i))
                    continue
                # print(sci.value, sky.value)
                xmobile = np.arange(sky.value-20, sci.value+20, dtype=np.int)
                # print('xmobile : {xmobile}'.format(xmobile=xmobile.shape))
                if xmobile.shape[0] == 0:
                    print('Not enough points to find the order. Skipping')
                    continue
                ymobile = arcdata[xmobile, y]
                g1 = models.Gaussian1D(amplitude=1., mean=sci, stddev=5)
                g2 = models.Gaussian1D(amplitude=1., mean=sky, stddev=5)
                g = g1 + g2
                gfit = fitter(g, xmobile, ymobile/ymobile.max(), verblevel=0)
                if gfit.mean_0 > parameters['Y'] or gfit.mean_1 > parameters['Y']:
                    print('Out of bounds')
                    break
                sci = gfit.mean_0
                sky = gfit.mean_1
                # print('Center of fibres at position {p} : sky : {sky}, sci : {sci}.'.format(p=y, sci=sci.value, sky=sky.value))
                skyorder.append(sky.value)
                scienceorder.append(sci.value)
                positions.append(y)
                fit.append(gfit)
            if direction == -1:
                fit = fit[::-1]
                positions = positions[::-1]
            if len(positions) > lmax:
                lmax = len(positions)
                p = positions
            # print('nb de points : {positions}'.format(positions=len(positions)))
            ysky = []
            yscience = []
            pfit = {}
            polyorder = 7
            for f in range(len(fit)):
                ysky.append(fit[f].mean_0.value)
                yscience.append(fit[f].mean_1.value)
            pfit.update(
                    {
                        'yscience': np.poly1d(np.polyfit(positions, yscience, polyorder)),
                        'ysky': np.poly1d(np.polyfit(positions, ysky, polyorder)),
                        'fit': fit
                    }
                    )
        order.update(
                {
                    str(ordernumber): pfit
                    }
                )
        order.update({'X': p})

    return order


def assess_stability():
    arclist = classify_files()
    # print(arclist)
    return arclist


def set_parameters(arcfile):
    print('extracting information from file {arcfile}'.format(arcfile=arcfile))
    ff = fits.open(arcfile)
    parameters = {
            'HBDET': {
                'Level': 15,
                'Distance': 30,
                'OrderShift': 85
                    },
            'HRDET': {
                'Level': 15,
                'Distance': 40,
                'OrderShift': 53
                    },
            'X': ff[0].header['NAXIS1'],
            'Y': ff[0].header['NAXIS2'],
            'center': int(ff[0].header['NAXIS1']/2),
            'chip': ff[0].header['DETNAM'],
            'data': ff[0].data,
            'X1': int(ff[0].header['DATASEC'][1:8].split(':')[0]),
            'X2': int(ff[0].header['DATASEC'][1:8].split(':')[1]),
            'Y1': int(ff[0].header['DATASEC'][9:15].split(':')[0]),
            'Y2': int(ff[0].header['DATASEC'][9:15].split(':')[1]),
            'nbpixperstep': 11
                }
    return parameters


def prepare_data(data):
    obs = fits.open(data)
    if parameters['chip'] == 'HRDET':
        bias = fits.open('R201704150021.fits')
        d = obs[0].data
    else:
        bias = fits.open('H201704150021.fits')
        d = obs[0].data[::-1, :]

    dt = d[np.int(parameters['Y1']):np.int(parameters['Y2']), np.int(parameters['X1']):np.int(parameters['X2'])] - bias[0].data.mean()
# We crudely remove the cosmics by moving all pixels in the highest bin of a 50-bin histogram to the second lowest.
    hist, bins = np.histogram(dt, bins=50)
    dt[np.where(dt >= bins[-2])] = bins[2]

    return dt


def plot_orders(orderframe, orderpositions):
    from astropy.visualization import ZScaleInterval
    (vmin, vmax) = ZScaleInterval().get_limits(orderframe)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for o in orderpositions.keys():
        print('Plotting order {o}'.format(o=o))
        if 'X' in o:
            continue
        ysky = []
        yscience = []
        x = []
        if 'X' in o:
            continue
        # for i in range(len(orderpositions[o]['fit'])):
        for pixel in range(orderframe.shape[1]):

            if pixel < orderpositions['X'][0]:
                continue
            try:
                i = orderpositions['X'].index(pixel)
                keepi = i
            except ValueError:
                i = keepi
            if i > len(orderpositions[o]['fit'])-1:
                continue
            x.append(pixel)
            # We append the center of the order, the lower and the upper limits to ysky and yscience.

            ysky.append(orderpositions[o]['yscience'](pixel))
            ysky.append(orderpositions[o]['yscience'](pixel) - 2.5*orderpositions[o]['fit'][i].stddev_1)
            ysky.append(orderpositions[o]['yscience'](pixel) + 2.5*orderpositions[o]['fit'][i].stddev_1)
            yscience.append(orderpositions[o]['ysky'](pixel))
            yscience.append(orderpositions[o]['ysky'](pixel) - 2.5*orderpositions[o]['fit'][i].stddev_0)
            yscience.append(orderpositions[o]['ysky'](pixel) + 2.5*orderpositions[o]['fit'][i].stddev_0)
        ax.annotate(o, xy=(x[0], ysky[0]), color='white')
        ax.annotate(o, xy=(x[parameters['center']], ysky[parameters['center']]), color='white')
        ax.annotate(o, xy=(x[-1], ysky[-1]), color='white')
        # print(x, y1)
        # plt.plot(x, y1c, 'blue')  # , x, y1b, 'blue', x, y1t, 'blue')
        ax.plot(x, ysky[::3], 'green',  x, ysky[1::3], 'blue', x, ysky[2::3], 'blue')
        ax.plot(x, yscience[::3], 'green',  x, yscience[1::3], 'red', x, yscience[2::3], 'red')
    ax.imshow(orderframe, vmin=vmin, vmax=vmax)
    # return y1c, x


def extract_order(data, orderpositions, order):
    print('Extracting order {order}'.format(order=order))
    import pandas as pd
    from astropy.convolution import Gaussian2DKernel, convolve
    # We try first on one particular order.
    o = str(order)
    X = orderpositions['X']
    extracted = []
    orderex = []
    for pixel in np.arange(parameters['X1'], min(data.shape[1], parameters['X2'])):
        if pixel < X[0]:
            continue
        try:
            i = X.index(pixel)
            keepi = i
        except ValueError:
            i = keepi
        skb = np.int(orderpositions[o]['yscience'](pixel) - 2.5*orderpositions[o]['fit'][i].stddev_1)+1
        skt = np.int(orderpositions[o]['yscience'](pixel) + 2.5*orderpositions[o]['fit'][i].stddev_1)
        extracted.append(data[skb:skt, pixel].sum())
        scb = np.int(orderpositions[o]['ysky'](pixel) - 2.5*orderpositions[o]['fit'][i].stddev_0)+1
        sct = np.int(orderpositions[o]['ysky'](pixel) + 2.5*orderpositions[o]['fit'][i].stddev_0)
        # print('Order size at pixel {pixel}: \nScience: {science} pixels\nSky: {sky}'.format(pixel=pixel, science=sct-scb, sky=skt-skb))
        # print(data[scb:sct, pixel].sum(), data[scb:sct, pixel].std())
        orderex.append(data[scb:sct, pixel])
        extracted.append(data[scb:sct, pixel].sum())
    df = pd.DataFrame(orderex)
    gaussk = Gaussian2DKernel(stddev=4)
    oc = convolve(df.T.values, gaussk)
    orderconvolved = oc.sum(axis=0)

    return extracted, orderconvolved


if __name__ == "__main__":
    arcfiles = assess_stability()
    parameters = set_parameters(arcfiles['Flat'][-1])
    # tp = fits.open('H201704120017.fits')
    tp = 'H201704120017.fits'
    data = prepare_data(arcfiles['Flat'][-1])
    order = fit_orders_pair(data)
