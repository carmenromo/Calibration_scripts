
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from functools import partial
from cycler import cycler

from calutils import weighted_av_std

import invisible_cities.core.fit_functions        as fitf
import invisible_cities.reco.spe_response         as speR
from invisible_cities.database import load_db     as DB
from sipmCalFit                import fit_dataset as sipmF


GainSeeds = [21.3, 23.4, 26.0, 25.7, 30.0, 22.7, 25.1, 32.7, 23.1, 25.5, 20.8, 22.0]
SigSeeds  = [11.3, 11.5, 10.6, 11.9, 13.1, 9.9, 11.0, 14.7, 10.6, 10.4, 9.3, 10.0]

ffuncs = {'ngau':speR.poisson_scaled_gaussians(n_gaussians=7),
          'intgau':speR.poisson_scaled_gaussians(min_integral=100),
          'dfunc':partial(speR.scaled_dark_pedestal, min_integral=100),
          'conv':partial(speR.dark_convolution, min_integral=100)}


darr = np.zeros(3)
def scaler(x, mu):
    global darr
    return mu * darr


def seeds_and_bounds(indx, func, bins, spec, ped_vals, ped_errs, lim_ped):

    norm_seed = spec.sum()
    
    ped_seed = ped_vals[1]
    ped_min  = ped_seed - lim_ped * ped_errs[1]
    ped_max  = ped_seed + lim_ped * ped_errs[1]

    ped_sig_seed = ped_vals[2]
    ped_sig_min  = max(0.001, ped_sig_seed - lim_ped * ped_errs[2])
    ped_sig_max  = ped_sig_seed + lim_ped * ped_errs[2]

    ## Remove the ped prediction and check try to get seeds for 1pe
    # first scale the dark pedestal
    dscale = spec[bins<0].sum() / fitf.gauss(bins[bins<0], *ped_vals).sum()
    GSeed  = GainSeeds[indx]
    GSSeed = SigSeeds[indx]
        
    ## Test scale
    ftest = fitf.fit(scaler, bins[bins<0], spec[bins<0], (dscale))

    if 'gau' in func:
        # There are 6 variables: normalization, pedestal pos., spe mean, poisson mean, pedestal sigma, 1pe sigma
        sd0 = (norm_seed, -np.log(ftest.values[0]), ped_seed, ped_sig_seed, GSeed, GSSeed)
        bd0 = [(0, 0, ped_min, ped_sig_min, 0, 0.001), (1e10, 10000, ped_max, ped_sig_max, 10000, 10000)]
        return sd0, bd0
    ## The other functions only have four parameters: normalization, spe mean, poisson mean, 1pe sigma
    sd0 = (norm_seed, -np.log(ftest.values[0]), GSeed, GSSeed)
    bd0 = [(0, 0, 0, 0.001), (1e10, 10000, 10000, 10000)]
    return sd0, bd0


def fit_dataset(dataF_table, funcName, min_stat, limit_ped):

    bins = np.array(dataF_table.root.HIST.pmt_dark_bins)
    specsD = np.array(dataF_table.root.HIST.pmt_dark).sum(axis=0)
    specsL = np.array(dataF_table.root.HIST.pmt_spe).sum(axis=0)

    ## pedSig err poissonMu err gain err gSig err chi2
    fitVals = np.zeros((12, 9), dtype=np.float)
    for i, (dspec, lspec) in enumerate(zip(specsD, specsL)):
        #print('Channel: ', i)
        b1 = 0
        b2 = len(dspec)
        if min_stat != 0:
            valid_bins = np.argwhere(lspec>=min_stat)
            b1 = valid_bins[0][0]
            b2 = valid_bins[-1][0]

        ## Fit the dark spectrum with a Gaussian (not really necessary for the conv option)
        gb0 = [(0, -100, 0), (1e99, 100, 10000)]
        av, rms = weighted_av_std(bins[dspec>100], dspec[dspec>100])
        sd0 = (dspec.sum(), av, rms)
        errs = np.sqrt(dspec[dspec>100])
        errs[errs==0] = 0.0001
        gfitRes = fitf.fit(fitf.gauss, bins[dspec>100], dspec[dspec>100], sd0, sigma=errs, bounds=gb0)

        fitVals[i,0] = gfitRes.values[2]
        fitVals[i,1] = gfitRes.errors[2]

        scale = lspec.sum() / dspec.sum()
        
        if 'dfunc' in funcName:
            respF = ffuncs[funcName](dark_spectrum=dspec[b1:b2] * scale,
                                     pedestal_mean=gfitRes.values[1],
                                     pedestal_sigma=gfitRes.values[2])
        elif 'conv' in funcName:
            respF = ffuncs[funcName](dark_spectrum=dspec[b1:b2] * scale,
                                     bins=bins[b1:b2])
        else:
            respF = ffuncs[funcName]

        ped_vals = np.array([gfitRes.values[0] * scale, gfitRes.values[1], gfitRes.values[2]])

        binR = bins[b1:b2]
        global darr
        darr = dspec[b1:b2] * scale
        darr = darr[binR<0]
        seeds, bounds = seeds_and_bounds(i, funcName, bins[b1:b2], lspec[b1:b2],
                                         ped_vals, gfitRes.errors, limit_ped)

        ## The fit
        errs = np.sqrt(lspec[b1:b2])
        if not 'gau' in funcName:
            errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dspec[b1:b2])
        errs[errs==0] = 1

        rfit = fitf.fit(respF, bins[b1:b2], lspec[b1:b2], seeds, sigma=errs, bounds=bounds)

        fitVals[i,2] = rfit.values[1]
        fitVals[i,3] = rfit.errors[1]
        fitVals[i,8] = rfit.chi2
        if 'gau' in funcName:
            fitVals[i,4] = rfit.values[4]
            fitVals[i,5] = rfit.errors[4]
            fitVals[i,6] = rfit.values[5]
            fitVals[i,7] = rfit.errors[4]
        else:
            fitVals[i,4] = rfit.values[2]
            fitVals[i,5] = rfit.errors[2]
            fitVals[i,6] = rfit.values[3]
            fitVals[i,7] = rfit.errors[3]

    return fitVals


def optPMTCal(fileNames, intWidths, funcName, min_stat, limit_ped):

    fResults = []
    for i in range(len(fileNames)):

        with tb.open_file(fileNames[i], 'r') as dataF:
            fResults.append(fit_dataset(dataF, funcName, min_stat, limit_ped))

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20,6))
    fig.show()
    axis_titles = ['Pedestal sigma', 'Poisson mu', 'Gain', 'Gain sigma', 'chi2']
    for j in range(12):
        ## clear the axes first
        for k, ax in enumerate(axes.flatten()):
            ax.cla()

            if k < 5:
                vals = np.fromiter((pars[j][k*2] for pars in fResults), np.float)
                if k < 4:
                    errs = np.fromiter((pars[j][k*2+1] for pars in fResults), np.float)
                ax.errorbar(intWidths, vals, yerr=errs, fmt='r.', ecolor='r')
                ax.set_title(axis_titles[k]+' vs integral width for PMT '+str(j))
        plt.tight_layout()
        plt.draw()
        catcher = input("next plot? q to stop, s to save ")
        if catcher == 'q':
            exit()
        if catcher == 's':
            plt.savefig('pmtCalOptPlots_ch'+str(j)+'.png')
        plt.cla()


def comparison_plots(fileNames, funcName, min_stat, limit_ped):

    fResults = []
    for i in range(len(fileNames)):
        #print('File: ', fileNames[i])
        with tb.open_file(fileNames[i], 'r') as dataF:
            fResults.append(fit_dataset(dataF, funcName, min_stat, limit_ped))

    axistitles = ['Pedestal sigma vs. channel number',
                  'Normalised Poisson mu vs. channel number',
                  'Gain vs. channel number',
                  '1pe sigma vs. channel number',
                  'Fit chi^2', 'Legend']
    chNosAll = np.arange(12)
    chNos_temp = np.array([0, 1, 2, 3,4, 5, 6, 7, 8, 10, 11])
    run_nos = [f[f.find('R')+1:f.find('R')+5] for f in fileNames]
    run_nos = [run+'MAU' if f.find('Mau') != -1 else run for f, run in zip(fileNames, run_nos)]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20,6))
    #cm = plt.get_cmap('gist_rainbow')
    for j, (vals1, run) in enumerate(zip(fResults, run_nos)):
        chNos = chNosAll
        vals = vals1
        if '4819' in run:
            chNos = chNos_temp
            vals = vals1[:-1]
        for k, (ax, axtit) in enumerate(zip(axes.flatten(), axistitles)):
            if j == 0:
                #ax.set_prop_cycle(cycler('color', [cm(1.*i/vals.shape[0]) for i in range(vals.shape[0])]))
                ax.set_title(axtit)
                ax.set_xlabel('Channel number')
            if k < 4:
                if k == 1:
                    ## We want to normalise to the maximum value here.
                    maxV = vals[:,2*k].max()
                    maxE = vals[:,2*k+1][vals[:,2*k].argmax()]
                    vp = vals[:,2*k] / maxV
                    vpE = np.fromiter((z*np.sqrt((ex/x)**2+(maxE/maxV)**2) for z, x, ex in zip(vp, vals[:,2*k], vals[:,2*k+1])), np.float)
                    ax.errorbar(chNos, vp, yerr=vpE, label='Run '+run)
                else:
                    ax.errorbar(chNos, vals[:,2*k], yerr=vals[:,2*k+1], label='Run '+run)
            elif k == 4:
                ax.plot(chNos, vals[:,2*k], label='Run '+run)
            else:
                ## trick to put legend on empty subplot
                ax.plot(0,0, label='Run '+run)
                ax.legend(ncol=2)
    plt.tight_layout()
    fig.show()
    plt.show()


def poisson_plots(fileNames, funcName, led_positions, min_stat, limit_ped):

    fResults = []
    for i in range(len(fileNames)):
        #print('File: ', fileNames[i])
        with tb.open_file(fileNames[i], 'r') as dataF:
            fResults.append(fit_dataset(dataF, funcName, min_stat, limit_ped))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
    pm1_mean     = []
    pm1_errs     = []
    pms_relative = []
    pms_rel_errs = []
    mask = np.array([True, False, True, True, True, True, True, True, True, True, True, True])
    for j, (vals, led) in enumerate(zip(fResults, led_positions)):
        pm1_val = vals[1,2] / vals[:,2].mean()
        meanErr = np.sqrt(np.sum(vals[:,3]**2)) / len(vals[:,2])
        pm1_err = pm1_val * np.sqrt((vals[1,3]/vals[1,2])**2+(meanErr/vals[:,2].mean())**2)
        pm1_mean.append(pm1_val)
        pm1_errs.append(pm1_errs)
        maxV = vals[:,2].max()
        vp = vals[:,2] / maxV
        maxE = vals[:,3][vals[:,2].argmax()]
        vpE = np.fromiter((z*np.sqrt((ex/x)**2+(maxE/maxV)**2) for z, x, ex in zip(vp, vals[:,2], vals[:,3])), np.float)
        pms_relative.append(vp[mask])
        pms_rel_errs.append(vpE[mask])

    ## PMT positions, doesn't change run to run, use first cal run from Run III
    db = DB.DataPMT(5316)
    pmt_x = db.X.values
    pmt_y = db.Y.values
    pm1_rel  = np.fromiter((np.sqrt((pmt_x[1]-p[0])**2+(pmt_y[1]-p[1])**2) for p in led_positions), np.float)
    pms_rpos = [np.fromiter((np.sqrt((x-p[0])**2+(y-p[1])**2) for x, y in zip(pmt_x[mask], pmt_y[mask])), np.float) for p in led_positions]
    pms_rpos = np.concatenate(pms_rpos)
    axes[0].errorbar(pm1_rel, pm1_mean, yerr=pm1_err, fmt='.')
    axes[0].set_title('PMT 1 Poisson mu relative to run average')
    axes[0].set_xlabel('XY distance between LED and PMT (mm)')
    axes[0].set_ylabel('Poisson mu relative to mean')
    axes[1].errorbar(pms_rpos, np.concatenate(pms_relative), yerr=np.concatenate(pms_rel_errs), fmt='.')
    axes[1].set_title('Poisson mu relative to max.')
    axes[1].set_xlabel('XY distance between LED and PMT (mm)')
    axes[1].set_ylabel('Poisson mu relative to max value')
    plt.tight_layout()
    fig.show()
    plt.show()
        
        

def optSiPMCal(fileNames, intWidths, funcName, min_stat, limit_ped):

    fResults = []
    for i in range(len(fileNames)):
        fResults.append(sipmF(fileNames[i], funcName, min_stat, limit_ped))

    axistitles = ['Gain distribution',
                  '1pe sigma distribution',
                  'Poisson mu distribution',
                  'Chi^2 distribution']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,6))
    for j, vals in enumerate(fResults):
        for ax, val, axtit in zip(axes.flatten(), vals, axistitles):
            ax.hist(val, bins=100, range=(0, 30), log=True, label='Integral '+intWidths[j])
            if j == 0:
                ax.set_title(axtit)
    plt.legend()
    plt.tight_layout()
    fig.show()
    catcher = input('thoughts?')
    if 's' in catcher:
        fig.savefig('sipmOptPlots.png')


def sipm_comparison(fileNames, funcName, min_stat, limit_ped):

    run_nos = [f[f.find('R')+1:f.find('R')+5] for f in fileNames]
    fResults = []
    for i in range(len(fileNames)):
        fResults.append(sipmF(fileNames[i], funcName, min_stat, limit_ped))

    axistitles = ['Gain differences', '1pe sigma differences', 'Poisson mu differences', 'Chi2 differences']

    print('Making difference plots...')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,6))
    r0vals = fResults[0]
    for j in range(1, len(fResults)):
        for k, (ax, val, axtit) in enumerate(zip(axes.flatten(), fResults[j], axistitles)):
            valDiff = val - r0vals[k]
            if k == 0:
                print('Run ', run_nos[j], np.argwhere(np.abs(valDiff) > 1))
                ax.hist(valDiff[np.abs(valDiff)<=20], bins=100, log=True,
                        label='Difference R'+run_nos[j]+' - R'+run_nos[0])
            else:
                ax.hist(valDiff, bins=100, log=True,
                        label='Difference R'+run_nos[j]+' - R'+run_nos[0])
            if j == 1:
                ax.set_title(axtit)
    plt.legend()
    plt.tight_layout()
    fig.show()
    catcher = input('thoughts?')
    if 's' in catcher:
        fig.savefig('sipmRunDifferencePlots.png')

    
