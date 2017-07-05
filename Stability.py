#!/usr/bin/env python

# -*- coding: utf-8 -*-

# sys imports

# python imports
from glob import glob

# numpy imports
import numpy as np

# astropy imports
from astropy.io import fits

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


def find_peaks():
    pixelstart = 50
    pixelstop = parameters['X2']
    step = 50
    xb = np.arange(pixelstart, pixelstop, step)
    temp = []
    for pixel in xb:
        xp = find_peaks_cwt(savgol_filter(parameters['data'][:, pixel], 11, 5), widths=np.arange(1, 20))
# The wavelet transform sometimes picks noise. Let's remove it now.
        # m = np.isclose(parameters['data'][:, pixel][xp], np.zeros_like(parameters['data'][:, pixel][xp]), atol=20)
        # print(m)
        # xxp = xp[np.invert(m)]
        plt.scatter(pixel*np.ones(len(xp)), xp, s=30)
        temp.append(xp)
    # Storing the location of the peaks in a numpy array
    size = max([len(i) for i in temp])
    peaks = np.ones((size, len(temp)), dtype=np.int)
    for index in range(len(temp)):
        temp[index].resize(size, refcheck=False)
        peaks[:, index] = temp[index]
# We need to remove the zeros.

    return peaks


def identify_orders(pts):
    """
    This function extracts the real location of the orders
    The input parameter is a numpy array containing the probable location of the orders. It has been filtered
    to remove the false detection of the algorithm.

    """
    pp = []
    o = np.zeros_like(pts)
    for i in range(0, 30):
        # we find where there is a discontinuity in the position of the orders
        # We only use the first 15 orders, since we know that it's the same for all orders
        # and they are better defined than the other.
        # If there is more than a 10 pixel shift between two consecutive peaks, then we have moved to the next order.
        # Except if the value is zero, which indicates it's the first order.
        po = np.where((pts[i, 1:] - pts[i, :-1]) > 5)[0]
        pp.append(po)
        # We first find the shortest list that describes the break. It's likely found for the best orders
    m = min([len(p) for p in pp])
    # Then we find which is this list of indices, and we use it as the places where the orders break
    for t in range(len(pp)):
        if len(pp[t]) == m:
            p = pp[t] + 1
            break

    print('changement à', p, len(p))
# The indices will allow us to know when to switch row in order to follow the orders.
# The first one has to be zero and the last one the size of the orders, so that the automatic procedure picks them properly
    indices = [0] + list(p) + [pts.shape[1]]
    print(indices)
    for i in range(73):
        # The orders come in three section, so we coalesce them
        print('indice', i)
        ind = np.arange(i, i-(len(p)+1), -1) + 1
        ind[np.where(ind <= 0)] = 0
        a = ind > 0
        a = a*1
        for j in range(len(a)):
            print(j)
            print(indices[j]*50, indices[j+1]*50)
            # For the first two orders, there are two discontinuities, but only one for the ones after.
            diff = j
            if i >= 2:
                if j >= 2:
                    diff = 1
            arr1 = pts[i-diff, indices[j]:indices[j+1]] * a[j]
            o[i, indices[j]:indices[j+1]] = arr1
    return o


# Un moyen d'aller plus vite, c'est de vectoriser le calcul des fits. Cela se fait avec np.vectorize une fois qu'on a défini des fonctions qui vont faire un calcul sur un élément des tableaux. C'est dans find_orders, vgf et vadd.
def gaussian_fit(a, k):
    from astropy.modeling import fitting, models
    fitter = fitting.SLSQPLSQFitter()
    gaus = models.Gaussian1D(amplitude=1., mean=a, stddev=5.)
    # print(gaus)
    # print(a, k)
    y1 = a-25
    y2 = a+25
    y = np.arange(y1, y2)
    gfit = fitter(gaus, y, parameters['data'][y, 50*(k+1)]/parameters['data'][y, 50*(k+1)].max(), verblevel=0)
    return(gfit)


def add_gaussian(a):
    """ Computes the size of the orders by adding/substracting two times the standard dev of the gaussian
    fit to the mean of the same fit.
    Returns the lower limit, the center and the upper limit.
    """
    return(a.mean.value-2.7*a.stddev.value, a.mean.value, a.mean.value+2.7*a.stddev.value)


def find_orders(op):
    """ Computes the location of the orders
    Returns a 3D numpy array
    """
    vgf = np.vectorize(gaussian_fit)
    vadd = np.vectorize(add_gaussian)
    fit = np.zeros_like(op, dtype=object)
    positions = np.zeros((op.shape[0], op.shape[1], 3))
    for i in range(op.shape[1]):
        tt = vgf(op[:, i], i)
        fit[:, i] = tt
    positions[:, :, 0], positions[:, :, 1], positions[:, :, 2] = vadd(fit)
    return positions, fit


def extract_orders(positions, data):
    """ positions est un array à 3 dimensions représentant pour chaque point des ordres detectes la limite inférieure, le centre et la limite supérieure des ordres.
    [:,:,0] est la limite inférieure
    [:,:,1] le centre,
    [:,:,2] la limite supérieure
    """
# TODO penser à mettre l'array en fortran, vu qu'on travaille par colonnes, ça ira plus vite.

    # data = parameters['data']
    orders = np.zeros((positions.shape[0], data.shape[1]))
    nborder = orders.shape[1]
    x = [i for i in range(nborder)]
    for o in range(2, orders.shape[0]):
        X = [50*(i+1) for i in range(positions[o, :, 0].shape[0])]
        foinf = np.poly1d(np.polyfit(X, positions[o, :, 0], 7))
        fosup = np.poly1d(np.polyfit(X, positions[o, :, 2], 7))
        orderwidth = np.ceil(np.mean(fosup(x)-foinf(x))).astype(int)
        for i in x:
            orders[o, i] = data[np.int(foinf(i)):np.int(foinf(i))+orderwidth, i].sum()
    return orders


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
                'OrderShift': 85,
                'XPix': 2048,
                'BiasLevel': 690
                    },
            'HRDET': {
                'Level': 15,
                'Distance': 40,
                'OrderShift': 53,
                'XPix': 4096,
                'BiasLevel': 920
                    },
            'X': ff[0].header['NAXIS1'],
            'Y': ff[0].header['NAXIS2'],
            'center': int(ff[0].header['NAXIS1']/2),
            'chip': ff[0].header['DETNAM'],
            'data': ff[0].data,
            'mode': ff[0].header['OBSMODE'],
            'X1': int(ff[0].header['DATASEC'][1:8].split(':')[0]),
            'X2': int(ff[0].header['DATASEC'][1:8].split(':')[1]),
            'Y1': int(ff[0].header['DATASEC'][9:15].split(':')[0]),
            'Y2': int(ff[0].header['DATASEC'][9:15].split(':')[1]),
            'nbpixperstep': 11,
            'ccdtype': ff[0].header['CCDTYPE']
                }
    return parameters


def prepare_data(data):
    obs = fits.open(data)
    if parameters['chip'] == 'HRDET':
        bias = fits.open('R201704150021.fits')
        d = obs[0].data  # - bias[0].data
    else:
        bias = fits.open('H201704150021.fits')
        d = obs[0].data[::-1, :]  # - bias[0].data

    dt = d[np.int(parameters['Y1']):np.int(parameters['Y2']), np.int(parameters['X1']):np.int(parameters['X2'])]  # - bias[0].data.mean()
# We crudely remove the cosmics by moving all pixels in the highest bin of a 50-bin histogram to the second lowest.
    hist, bins = np.histogram(dt, bins=50)
    d = d-bias[0].data.mean()
    if 'Flat' in parameters['ccdtype']:
        print('Flatfield : removing cosmics')
        dt[np.where(dt >= bins[-2])] = bins[2]
    else:
        print('Not a flat, not doing anything')

    # d[np.where(d<=0)] = 0

    return d


def plot_orders(data):
    orderframe = data['data']
    orderpositions = data['order']
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
        xlabelleft = 0.25*(x[-1]+x[0])
        xlabelcentre = 0.5*(x[-1]+x[0])
        xlabelright = 0.75*(x[-1]+x[0])
        ylabelleft = orderpositions[o]['yscience'](xlabelleft)
        ylabelcentre = orderpositions[o]['yscience'](xlabelcentre)
        ylabelright = orderpositions[o]['yscience'](xlabelright)
        ax.annotate(o, xy=(xlabelleft, ylabelleft), color='peachpuff')
        ax.annotate(o, xy=(xlabelcentre, ylabelcentre), color='turquoise')
        ax.annotate(o, xy=(xlabelright, ylabelright), color='orange')
        # print(x, y1)
        # plt.plot(x, y1c, 'blue')  # , x, y1b, 'blue', x, y1t, 'blue')
        ax.plot(x, ysky[::3], 'green',  x, ysky[1::3], 'blue', x, ysky[2::3], 'blue')
        ax.plot(x, yscience[::3], 'green',  x, yscience[1::3], 'red', x, yscience[2::3], 'red')
    ax.imshow(orderframe, vmin=vmin, vmax=vmax)


def old_extract_order(data, orderpositions, order=None):
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
        # print(pixel, scb, sct)
        # print(data[scb:sct, pixel].sum(), data[scb:sct, pixel].std())
        orderex.append(data[scb:sct, pixel])
        extracted.append(data[scb:sct, pixel].sum())
    df = pd.DataFrame(orderex)
    gaussk = Gaussian2DKernel(stddev=4)
    oc = convolve(df.T.values, gaussk)
    orderconvolved = oc.sum(axis=0)

    return np.array(extracted), np.array(orderconvolved)


def wavelength(orders):
    # We load the spectral format of the chips"
    # fmt = np.loadtxt('HRS_Spectral_Format.dat', delimiter=',', dtype={'names':('order', 'centralwl', 'wlrange'),
    #                                                                 'formats' :('S3', 'f4', 'f4')})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    wl = {}
    with open('HRS_Spectral_Format.dat') as f:
        for l in f:
            (o, lc, ra) = l.split(',')
            wl[str(o)] = {'λcen': lc, 'range': ra.strip('\n'), 'step': np.float(ra.strip())/parameters[parameters['chip']]['XPix']}
    npix = parameters['X2'] - parameters['center']
    x = np.arange(4095)
    l = []
    o = []
    lcen = []
    step = []
    lrange = []
    for k in wl.keys():
        # print(k, wl[k]['λcen'])
        o.append(np.float(k))
        lcen.append(np.float(wl[k]['λcen']))
        step.append(np.float(wl[k]['step']))
        lrange.append(np.float(wl[k]['range']))
    os = o.copy()
    # plt.scatter(o, lcen)
    wl0 = np.poly1d(np.polyfit(o, lcen, 5))
    wl1 = np.poly1d(np.polyfit(o, step, 5))
    wl2 = np.poly1d(np.polyfit(o, lrange, 5))
    os.sort()
    # plt.plot(os, wl0(os))
    # Now the missing orders
    xo = []
    for xt in range(int(min(os)), int(max(os))):
        if not xt % 2:
            xo.append(xt)
    # print(xo, wl0(xo), wl1(xo), wl2(xo))
    # plt.scatter(xo, wl1(xo))
    t = {}
    for index, order in enumerate(xo):
        t.update({str(order): {'range': wl2(xo)[index], 'step': wl1(xo)[index], 'λcen': wl0(xo)[index]}})
    wl.update(t)
    for order in wl.keys():
        if order in orders.keys():
            lc = np.float(wl[order]['λcen'])
            st = wl[order]['step']
            xlm = [lc - i*st for i in range(npix)]
            xlp = [lc + i*st for i in range(npix)]
            xlm.sort()
            xlp.sort()
            xl = xlm + xlp
            print(len(xl))
            ylsc = orders[order]['yscience']
            ylsk = orders[order]['ysky']
            ax.plot(xl[parameters['X1']:parameters['X2']], ylsc(x), 'green', xl[parameters['X1']:parameters['X2']], ylsk(x), 'orange')
            print(lc, type(lc), ylsk(lc), order)
            ax.annotate(order, xy=(lc,  ylsk(lc)))
        else:
            continue
    return wl


def getshape(orderinf, ordersup):
    """
    In order to derive the shape of a given order, we use one order after and one order before
    We suppose that the variations are continuous.
    Thus approximating order n using orders n+1 and n-1 is close enough from reality
    """
    # Problems when the boundary order has some intense line, like order 65 and Hα.
    from scipy.signal import butter, filtfilt
    # Now we find the shape, it needs several steps and various fitting/smoothing
    b, a = butter(10, 0.025)
    # the shape of the orders does not vary much between orders, but there can be cosmic rays or emission lines
    # Since it's unlikely that the same pixel is affected on two non contiguous orders, we pick the minimum of the
    # two orders, and we create an artificial order between them and it's the one that we'll fit.
    shape = np.minimum(orderinf, ordersup)
    ysh2 = filtfilt(b, a, shape)
    x = np.arange(ysh2.shape[0])
    ysh3 = np.poly1d(np.polyfit(x, ysh2, 11))(x)
    ysh4 = np.minimum(ysh2, ysh3)
    ysh5 = np.poly1d(np.polyfit(x, ysh4, 11))(x)
    # ysh5 fits now the shape of the ThAr order quite well, and we can start from there to identify the lines.
    return ysh5

if __name__ == "__main__":
    arcfiles = assess_stability()
    # f = 'R201510210012.fits'
    f = arcfiles['Flat'][-1]
    parameters = set_parameters(f)
    if 'HBD'in parameters['chip']:
        print('Blue detector')
        parameters['data'] = parameters['data'][::-1, :]
    # tp = fits.open('H201704120017.fits')
    tp = 'H201704120017.fits'
    parameters['data'] = prepare_data(f)
    # parameters['order'] = fit_orders_pair(parameters['data'])
    # wavelength(order)
