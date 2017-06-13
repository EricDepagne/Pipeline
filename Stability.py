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
    return peaks[goodpeaks], peaks


def find_peaks3():
    pixelstart = 50
    pixelstop = parameters['X2']
    step = 50
    xb = np.arange(pixelstart, pixelstop, step)
    print(xb)
    temp = []
    for pixel in xb:
        xp = find_peaks_cwt(savgol_filter(parameters['data'][:, pixel], 11, 5), widths=np.arange(1, 20))
# The wavelet transform sometimes picks noise. Let's remove it now.
        m = np.isclose(parameters['data'][:, pixel][xp], np.zeros_like(parameters['data'][:, pixel][xp]), atol=20)
        xxp = xp[np.invert(m)]
        plt.scatter(pixel*np.ones(len(xxp)), xxp, s=3)
        temp.append(xxp)
    # Storing the location of the peaks in a numpy array
    size = max([len(i) for i in temp])
    peaks = np.ones((size, len(temp)), dtype=np.int)
    for index in range(len(temp)):
        temp[index].resize(size, refcheck=False)
        peaks[:, index] = temp[index]
# We need to remove the zeros.

    return peaks


def find_peaks2():
    # Instead of fitting a gaussian, we will parse the chip column by column and see if the find_peaks_cwt can do it better, since it seems to be detecting the orders much better on the edges of the chip.
    peaks = []
    gpeaks = []
    pixelstart = 50
    pixelstop = parameters['X2']
    step = 50
    for pixel in np.arange(pixelstart, pixelstop, step):
        cutfiltered = savgol_filter(parameters['data'][:, pixel], 11, 7)
        p = find_peaks_cwt(cutfiltered, widths=np.arange(1, 20))
        if pixel == 1950:
            plt.scatter(p, parameters['data'][:, pixel][p])

        peaks.append(p)

    # Sometimes, peaks are found between orders, we need to remove them.
    # that's easy : the intensity of the pixel between orders is equal to the bias value : 920 for red, 690 for blue.
    p4 = np.array(peaks)
    for index in range(1, len(peaks)):
        # print(parameters['data'][:, 50*index][p4[index]], index)
        # print(step*index)
        # We find the location of the peaks found, which correspond to a fluctuation in the background. Those are defined as being up 20 counts above the value of the bias frame at the same location.
        m = np.isclose(parameters['data'][:, step*index][p4[index]], np.zeros_like(parameters['data'][:, step*index][p4[index]]), rtol=30, atol=30)
        # once we have found all the pixels close to the background, we invert the array, which gives us all the pixels that are _not_ a fluctuation of the background
        gm = np.invert(m)
        # We add to the good peaks list those who are not this fluctuation.
        gpeaks.append(p4[index][gm])
    # Working with numpy arrays is easier, so now, we transform the list of arrays into a proper array
    size = max([len(i) for i in gpeaks])
    print(size)
    temparray = np.empty((size, len(gpeaks)))
    for index in range(len(gpeaks)):
        gpeaks[index].resize(size)
        temparray[:, index] = gpeaks[index]
    plt.show()

    return peaks, gpeaks, temparray


def identify_orders(pts):
    """
    This function extracts the real location of the orders
    The input parameter is a numpy array containing the probable location of the orders. It has been filtered
    to remove the false detection of the algorithm

    """
    pp = []
    o = np.zeros_like(pts)
    for i in range(1, 73):
        # we find where there is a discontinuity in the position of the orders
        po = np.where((pts[i, 1:] - pts[i, :-1]) > 0)[0]
        print(po)
        pp.append(po)
        # We first find the shortest list that describes the break. It's likely found for the best orders
    m = min([len(p) for p in pp])
    # Then we find which is this list of indices, and we use it as the places where the orders break
    for t in range(len(pp)):
        if len(pp[t]) == m and 0 not in pp[t]:
            # We want to avoid catching the first pixel, in case it's a noisy one.
            p = pp[t] + 1
            break

    print('changement à', p, len(p))
# The indices will allow us to know when to swith row in order to follow the orders.
# The first one has to be zero and the last one the size of the orders, so that the automatic procedure picks them properly
    indices = [0] + list(p) + [pts.shape[1]]
    print(indices)
    for i in range(1, 73):
        # The orders come in three section, so we coalesce them
        ind = np.arange(i, i-(len(p)+1), -1)
        ind[np.where(ind <= 0)] = 0
        a = ind > 0
        a = a*1
        print(ind, a)
        for j in range(len(a)):
            print(indices[j], indices[j+1])
            arr1 = pts[ind[j], indices[j]:indices[j+1]] * a[j]
            print('arr1', arr1)
            o[i, indices[j]:indices[j+1]] = arr1
    return o


def fit_orders_pair(arcdata):
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
    # for i in range(20, 28):
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

# TODO le fit rate quand les ordres ne sont pas bien fittés à cause du faible signal.
# Cela fout le bordel, trouver comment faire pour que ça marche!

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
            # very rough filtering of the data, in order to get a proper fitting.
            yska = np.array(ysky)
            ysca = np.array(yscience)
            outsc = np.where(ysca > ysca.mean() + 3*ysca.std())
            outsk = np.where(yska > yska.mean() + 3*yska.std())
            print('outliers : sky {sky}, science {science}'.format(sky=outsk, science=outsc))

            if outsk[0].shape[0]:
                yska[np.where(yska > yska.mean() + 3*yska.std())] = np.mean(yska[outsk[0][0]:5])
            if outsc[0].shape[0]:
                ysca[np.where(ysca > ysca.mean() + 3*ysca.std())] = np.mean(ysca[outsc[0][0]:5])
            pfit.update(
                    {
                        'yscience': np.poly1d(np.polyfit(positions, yscience, polyorder)),
                        'ysky': np.poly1d(np.polyfit(positions, ysky, polyorder)),
                        'fit': fit,
                        'yscdata': yscience,
                        'yskdata': ysky
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

    dt = d[np.int(parameters['Y1']):np.int(parameters['Y2']), np.int(parameters['X1']):np.int(parameters['X2'])] - bias[0].data.mean()
# We crudely remove the cosmics by moving all pixels in the highest bin of a 50-bin histogram to the second lowest.
    hist, bins = np.histogram(dt, bins=50)
    if 'Flat' in parameters['ccdtype']:
        print('Flatfield : removing cosmics')
        dt[np.where(dt >= bins[-2])] = bins[2]
    else:
        print('Not a flat, not doing anything')
    return d-bias[0].data.mean()


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


def extract_order(data, orderpositions, order=None):
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
