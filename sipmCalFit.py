import sys
import numpy as np
import tables as tb
from scipy.signal import find_peaks_cwt
from pandas import DataFrame
import matplotlib.pyplot as plt
from functools import partial

from invisible_cities.icaro.hst_functions import display_matrix
import invisible_cities.reco.spe_response as speR
import invisible_cities.core.fit_functions as fitf
from invisible_cities.database import load_db as DB
import invisible_cities.io.channel_param_io as pIO

from pmtCalFit import weighted_av_std

useSavedSeeds = True
GainSeeds = []
SigSeeds  = []
scalerChis = []


## Probably shite
darr = np.zeros(3)
def scaler(x, mu):
    global darr
    return mu * darr


def seeds_and_bounds(indx, func, bins, spec, ped_vals, ped_errs, lim_ped):

    global GainSeeds, SigSeeds, useSavedSeeds, scalerChis
    
    norm_seed = spec.sum()

    GSeed  = 0
    GSSeed = 0
    if useSavedSeeds:
        GSeed  = GainSeeds[indx]
        GSSeed = SigSeeds[indx]
    else:
        pDL = find_peaks_cwt(spec, np.arange(4, 20), min_snr=1, noise_perc=5)
        p1pe = pDL[(bins[pDL]>10) & (bins[pDL]<20)]
        if len(p1pe) == 0:
            p1pe = np.argwhere(bins==15)[0][0]
        else:
            p1pe = p1pe[spec[p1pe].argmax()]
        p1 = fitf.fit(fitf.gauss, bins[p1pe-5:p1pe+5], spec[p1pe-5:p1pe+5], seed=(spec[p1pe], bins[p1pe], 3.))
        GSeed = p1.values[1] - ped_vals[1]
        if p1.values[2] <= ped_vals[2]:
            GSSeed = 0.5
        else:
            GSSeed = np.sqrt(p1.values[2]**2 - ped_vals[2]**2)

    dscale = spec[bins<5].sum() / fitf.gauss(bins[bins<5], *ped_vals).sum()
    errs = np.sqrt(spec[bins<5])
    errs[errs==0] = 1
    fscale = fitf.fit(scaler, bins[bins<5], spec[bins<5], (dscale), sigma=errs)
    scalerChis.append(fscale.chi2)
    if scalerChis[-1] >= 500:
        print('Suspect channel index ', indx)
    muSeed = -np.log(fscale.values[0])
    if muSeed < 0: muSeed = 0.001

    if 'gau' in func:
        ped_seed = ped_vals[1]
        ped_min  = ped_seed - lim_ped * ped_errs[1]
        ped_max  = ped_seed + lim_ped * ped_errs[1]
        ped_sig_seed = ped_vals[2]
        ped_sig_min  = max(0.001, ped_sig_seed - lim_ped * ped_errs[2])
        ped_sig_max  = ped_sig_seed + lim_ped * ped_errs[2]
        sd0 = (norm_seed, muSeed, ped_seed, ped_sig_seed, GSeed, GSSeed)
        bd0 = [(0, 0, ped_min, ped_sig_min, 0, 0.001), (1e10, 10000, ped_max, ped_sig_max, 10000, 10000)]
        #print('Seed check: ', sd0)
        return sd0, bd0

    sd0 = (norm_seed, muSeed, GSeed, GSSeed)
    bd0 = [(0, 0, 0, 0.001), (1e10, 10000, 10000, 10000)]
    print('Seed check: ', sd0)
    return sd0, bd0


def fit_dataset(dataF=None, funcName=None, minStat=None, limitPed=None):

    """ Check new fit function on SiPM spectra """
    global useSavedSeeds, GainSeeds, SigSeeds

    file_name = dataF
    func_name = funcName
    min_stat = minStat
    limit_ped = limitPed
    optimise = True
    if not file_name:
        optimise = False
        file_name = sys.argv[1]
        func_name = sys.argv[2]
        min_stat  = 0
        limit_ped = 10000.
        if len(sys.argv) > 3:
            useSavedSeeds = True if 'true' in sys.argv[3] else False
            min_stat = int(sys.argv[4])
            limit_ped = int(sys.argv[5])

    run_no = file_name[file_name.find('R')+1:file_name.find('R')+5]
    run_no = int(run_no)
    chNos = DB.DataSiPM(run_no).SensorID.values
    if useSavedSeeds:
        dodgy = DB.DataSiPM(run_no).index[DB.DataSiPM(run_no).Active==0].values
        GainSeeds = DB.DataSiPM(run_no).adc_to_pes.values
        SigSeeds  = DB.DataSiPM(run_no).Sigma.values
        ## Give generic values to previously dead or dodgy channels
        GainSeeds[dodgy] = 15
        SigSeeds[dodgy] = 2

    sipmIn = tb.open_file(file_name, 'r')

    ## Bins are the same for dark and light, just use light for now
    bins = np.array(sipmIn.root.HIST.sipm_spe_bins)
    ## LED correlated and anticorrelated spectra:
    specsL = np.array(sipmIn.root.HIST.sipm_spe).sum(axis=0)
    specsD = np.array(sipmIn.root.HIST.sipm_dark).sum(axis=0)
    
    ffuncs = {'ngau':speR.poisson_scaled_gaussians(n_gaussians=7),
              'intgau':speR.poisson_scaled_gaussians(min_integral=100),
              'dfunc':partial(speR.scaled_dark_pedestal, min_integral=100),
              'conv':partial(speR.dark_convolution, min_integral=100)}

    ## Loop over the specra:
    outData = []
    outDict = {}
    llchans = []
    if not optimise:
        fnam = {'ngau':'poisson_scaled_gaussians_ngau', 'intgau':'poisson_scaled_gaussians_min', 'dfunc':'scaled_dark_pedestal', 'conv':'dark_convolution'}
        pOut = tb.open_file('sipmCalParOut_R'+str(run_no)+'_F'+func_name+'.h5', 'w')
        param_writer = pIO.channel_param_writer(pOut,
                                                sensor_type='sipm',
                                                func_name=fnam[func_name],
                                                param_names=pIO.generic_params)
    ## Extra protection since 3065 is weird
    knownDead = [ 3056, 11009, 12058, 14010, 22028, 22029, 25049 ]
    specialCheck = [1006, 1007, 3000, 3001, 5010, 7000, 22029, 28056, 28057]
    for ich, (led, dar) in enumerate(zip(specsL, specsD)):
        if chNos[ich] in knownDead:
            outData.append([chNos[ich], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 0, 0])
            if not optimise:
                for kname in pIO.generic_params:
                    outDict[kname] = (0, 0)
                param_writer(chNos[ich], outDict)
            print('no peaks in dark spectrum, spec ', ich)
            continue
        ## Limits for safe fit
        b1 = 0
        b2 = len(dar)
        if min_stat != 0:
            valid_bins = np.argwhere(led>=min_stat)
            b1 = valid_bins[0][0]
            b2 = valid_bins[-1][0]
        outDict[pIO.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)])
        # Seed finding
        pD = find_peaks_cwt(dar, np.arange(2, 20), min_snr=2)
        if len(pD) == 0:
            ## Try to salvage in case not a masked channel
            ## Masked channels have al entries in one bin.
            if led[led>0].size == 1:
                outData.append([0., 0., 0., 0., 0., 0., 0.])
                print('no peaks in dark spectrum, spec ', ich)
                continue
            else:
                pD = np.array([dar.argmax()])
        ## Fit the dark spectrum with a Gaussian (not really necessary for the conv option)
        gb0 = [(0, -100, 0), (1e99, 100, 10000)]
        sd0 = (dar.sum(), 0, 2)
        errs = np.sqrt(dar[pD[0]-5:pD[0]+5])
        errs[errs==0] = 0.1
        gfitRes = fitf.fit(fitf.gauss, bins[pD[0]-5:pD[0]+5], dar[pD[0]-5:pD[0]+5], sd0, sigma=errs, bounds=gb0)
        outDict[pIO.generic_params[2]] = (gfitRes.values[1], gfitRes.errors[1])
        outDict[pIO.generic_params[3]] = (gfitRes.values[2], gfitRes.errors[2])

        ## Scale just in case we lost a different amount of integrals in dark and led
        ## scale = led.sum() / dar.sum()
        scale = 1

        ## Take into account the scale in seed finding (could affect Poisson mu)????
        ped_vals = np.array([gfitRes.values[0] * scale, gfitRes.values[1], gfitRes.values[2]])

        binR = bins[b1:b2]
        global darr
        darr = dar[b1:b2] * scale
        darr = darr[binR<5]
        seeds, bounds = seeds_and_bounds(ich, func_name, bins[b1:b2], led[b1:b2],
                                         ped_vals, gfitRes.errors, limit_ped)      
        ## Protect low light channels
        if seeds[1] < 0.2:
            llchans.append(chNos[ich])
            ## Dodgy setting of high charge dark bins to zero
            dar[bins>gfitRes.values[1] + 3*gfitRes.values[2]] = 0
        ##
        if 'dfunc' in func_name:
            respF = ffuncs[func_name](dark_spectrum=dar[b1:b2] * scale,
                                     pedestal_mean=gfitRes.values[1],
                                     pedestal_sigma=gfitRes.values[2])
        elif 'conv' in func_name:
            respF = ffuncs[func_name](dark_spectrum=dar[b1:b2] * scale,
                                     bins=bins[b1:b2])
        else:
            respF = ffuncs[func_name]
        
        ## The fit
        errs = np.sqrt(led)
        if not 'gau' in func_name:
            errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dar)
        errs[errs==0] = 0.001
        print('About to fit channel ', chNos[ich])
        rfit = fitf.fit(respF, bins[b1:b2], led[b1:b2], seeds, sigma=errs[b1:b2], bounds=bounds)
        chi = rfit.chi2
        ## Attempt to catch bad fits and refit (currently only valid for dfunc and conv)
        if chi >= 7 or rfit.values[3] >= 2.5 or rfit.values[3] <= 1:
            ## The offending parameter seems to be the sigma in most cases
            nseed = rfit.values
            nseed[3] = 1.7
            nbound = [(bounds[0][0], bounds[0][1], bounds[0][2], 1),
                      (bounds[1][0], bounds[1][1], bounds[1][2], 2.5)]
            rfit = fitf.fit(respF, bins[b1:b2], led[b1:b2], nseed, sigma=errs[b1:b2], bounds=nbound)
            chi = rfit.chi2
        if not optimise:
            if chNos[ich] in specialCheck or chi >= 10 or rfit.values[2] < 12 or rfit.values[2] > 19 or rfit.values[3] > 3:
                if chNos[ich] in specialCheck: print('Special check channel '+str(chNos[ich]))
                print('Channel fit: ', rfit.values, chi)
                plt.errorbar(bins, led, xerr=0.5*np.diff(bins)[0], yerr=errs, fmt='b.')
                plt.plot(bins[b1:b2], respF(bins[b1:b2], *rfit.values), 'r')
                plt.plot(bins[b1:b2], respF(bins[b1:b2], *seeds), 'g')
                plt.title('Spe response fit to channel '+str(chNos[ich]))
                plt.xlabel('ADC')
                plt.ylabel('AU')
                plt.show()
        outData.append([chNos[ich], rfit.values, rfit.errors, respF.n_gaussians, chi])
        outDict[pIO.generic_params[0]] = (rfit.values[0], rfit.errors[0])
        outDict[pIO.generic_params[1]] = (rfit.values[1], rfit.errors[1])
        gIndx = 2
        if 'gau' in func_name:
            gaIndx = 4
        outDict[pIO.generic_params[4]] = (rfit.values[gIndx], rfit.errors[gIndx])
        outDict[pIO.generic_params[5]] = (rfit.values[gIndx+1], rfit.errors[gIndx+1])
        outDict[pIO.generic_params[-1]] = (respF.n_gaussians, rfit.chi2)
        if not optimise:
            param_writer(chNos[ich], outDict)

    ## Couple of plots
    gainIndx = 2
    if 'gau' in func_name:
        gainIndx = 4

    plot_names = ["Gain", "1pe sigma", "Poisson mu", "chi2"]
    pVals = [np.fromiter((ch[1][gainIndx] for ch in outData), np.float),
             np.fromiter((ch[1][gainIndx+1] for ch in outData), np.float),
             np.fromiter((ch[1][1] for ch in outData), np.float),
             np.fromiter((ch[4] for ch in outData), np.float)]
    if optimise:
        sipmIn.close()
        return pVals
    pOut.close()

    #global scalerChis
    pos_x = DB.DataSiPM(run_no).X.values
    pos_y = DB.DataSiPM(run_no).Y.values
    chNos = DB.DataSiPM(run_no).SensorID.values
    ## vals2D = np.zeros((int((pos_x.max()-pos_x.min())/10)+1, int((pos_y.max()-pos_y.min())/10)+1))
    ## print('shape: ', vals2D.shape)
    ## *_, chis = display_matrix(pos_x, pos_y, pVals[3])
    ## Trampa
    #pVals[3][pVals[3]>10] = 0
    plt.scatter(pos_x, pos_y, c=pVals[3])
    plt.title("Fit chi^2 map")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar()
    plt.show()
    #mask = np.argwhere((pVals[2]>=2) & (pVals[2]<8))
    #mask = np.argwhere((chNos<9000) | (chNos>=11000))
    #plt.scatter(pos_x[mask], pos_y[mask], c=pVals[2][mask])
    plt.scatter(pos_x, pos_y, c=pVals[2])
    plt.title("Fit poisson mu")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar()
    plt.show()
    ## fg2, ax2 = plt.subplots()
    ## p = ax2.pcolor(pos_x, pos_y, scalerChis, cmap=cm.Spectral, vmin=np.abs(scalerChis).min(), vmax=np.abs(scalerChis).max())
    ## plt.colorbar(p, ax=ax2)
    ## fg2.show()
    #plt.hist(scalerChis, bins=1000)
    #plt.show()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,6))
    chiVs = pVals[3]
    for ax, val, nm in zip(axes.flatten(), pVals, plot_names):
        ax.hist(val[(chiVs<10) & (chiVs!=0)], bins=100)
        ax.set_title(nm)
    fig.show()
    input('finished with plots?')

    print('Low light chans:', llchans)

    ## with open(file_name[:-3]+'_Fit_'+func_name+'.dat', 'w') as dbF:
    ##     dbF.write('Minimum statistics: '+str(min_stat)+'\n\n')
    ##     for vals in outData:
    ##         dbF.write(str(vals)+'\n')
        

if __name__ == '__main__':
    fit_dataset()
