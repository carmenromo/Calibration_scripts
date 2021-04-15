import argparse
import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt

from scipy.signal        import find_peaks_cwt
from functools           import partial
from argparse            import Namespace
from enum                import auto

from   invisible_cities.core    .stat_functions  import poisson_sigma
from   invisible_cities.reco    .calib_functions import seeds_and_bounds
from   invisible_cities.reco    .calib_functions import dark_scaler
from   invisible_cities.reco    .calib_functions import SensorType
from   invisible_cities.types   .ic_types        import AutoNameEnumBase
from   invisible_cities.cities  .components      import get_run_number

import invisible_cities.database.load_db    as db
import invisible_cities.reco.spe_response   as sper
import invisible_cities.core.fit_functions  as fitf
import invisible_cities.io.channel_param_io as cpio


def str2bool(v):
    """
    This function is added because the argparse add_argument('use_db_gain_seeds', type=bool)
    was not working in False case, everytime True was taken.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_spec_without_fit(bins, spec, errs, chann):
    plt.errorbar(bins, spec, xerr=0.5*np.diff(bins)[0], yerr=errs, fmt='b.')
    plt.title('Spe distribution for channel ' + str(chann))
    plt.xlabel('ADC')
    plt.ylabel('Entries')
    plt.show()

def histplot2d(pos_x, pos_y, variable, title):
    plt.scatter(pos_x, pos_y, c=variable, s=300)
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar()
    plt.show()



parser = argparse.ArgumentParser(description='Fit function on SiPM spectra.')
parser.add_argument('file_in',           type=str,      help='input spectra',                      )
parser.add_argument('func_name',         type=str,      help='function that will be used to fit',   default='dfunc')
parser.add_argument('use_db_gain_seeds', type=str2bool, help='option to take gain values from db',  default=False)
parser.add_argument('min_stat',          type=int,      help='min statistics for the peaks',        default=10)
parser.add_argument('black_box',         type=str2bool, help='In case the spectrum is from the bb', default=False)
args = parser.parse_args()


#db_file = '/Users/carmenromoluque/IC/invisible_cities/database/localdb.DEMOPPDB.sqlite3'
db_file = 'demopp'

file_name         = args.file_in
func_name         = args.func_name
use_db_gain_seeds = args.use_db_gain_seeds
min_stat          = args.min_stat
black_box         = args.black_box

h5in      = tb.open_file(file_name, 'r')
run_no    = get_run_number(h5in)
channs    = db.DataSiPM(db_file, run_no).SensorID.values
sens_type = SensorType.SIPM if 'sipm' in file_name else SensorType.PMT

## Bins are the same for dark and light, just use light for now
bins   = np.array(h5in.root.HIST.sipm_spe_bins)
## LED correlated and anticorrelated spectra:
lspecs = np.array(h5in.root.HIST.sipm_spe ).sum(axis=0)
dspecs = np.array(h5in.root.HIST.sipm_dark).sum(axis=0)

ffuncs = {'ngau'  :sper.poisson_scaled_gaussians(n_gaussians=7),
          'intgau':sper.poisson_scaled_gaussians(min_integral=100),
          'dfunc' :partial(sper.scaled_dark_pedestal, min_integral=100),
          'conv'  :partial(sper.dark_convolution, min_integral=100)}

## Loop over the spectra:
outData = []
outDict = {}
llchans = []
fnam = {'ngau':'poisson_scaled_gaussians_ngau', 'intgau':'poisson_scaled_gaussians_min', 'dfunc':'scaled_dark_pedestal', 'conv':'dark_convolution'}
out_file = tb.open_file('sipmCalParOut_R'+str(run_no)+'_F'+func_name+'_scale1.5.h5', 'w')
param_writer = cpio.channel_param_writer(out_file, sensor_type='sipm', func_name=fnam[func_name], param_names=cpio.generic_params)

knownDead    = []
specialCheck = [16028]

if black_box:
    init_sns = 192
    detector = 'Black Box'
else:
    init_sns = 0
    detector = 'DEMO++'

for ich, led, dar in zip(range(init_sns, len(lspecs)), lspecs[init_sns:], dspecs[init_sns:]):
    if channs[ich] in knownDead:
        if 'gau' in func_name:
            outData.append([channs[ich], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 0, 0])
        else:
            outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0, 0])

        for kname in cpio.generic_params:
            outDict[kname] = (0, 0)
        param_writer(channs[ich], outDict)
        print('no peaks in dark spectrum, spec ', channs[ich])
        continue

    ## Limits for safe fit
    b1 = 0
    b2 = len(dar)
    if min_stat != 0:
        try:
            valid_bins = np.argwhere(led>=min_stat)
            b1 = valid_bins[ 0][0] # This is due to the nature of np.argwhere. b1 first bin, b2 last bin
            b2 = valid_bins[-1][0]
        except IndexError:
            print(f'Error extracting valid bins for channel: {channs[ich]}')
            continue

    ## Fit limits
    outDict[cpio.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)])

    # Seed finding
    peaks_dark = find_peaks_cwt(dar, np.arange(2, 20), min_snr=2)
    if len(peaks_dark) == 0:
        ## Try to salvage in case not a masked channel
        ## Masked channels have al entries in one bin.
        if led[led>0].size == 1:
            outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0, 0])
            print('no peaks in dark spectrum, spec ', ich)
            continue
        else:
            peaks_dark = np.array([dar.argmax()])

    ## Fit the dark spectrum with a Gaussian (not really necessary for the conv option)
    gb0  = [(0, -100, 0), (1e99, 100, 10000)]
    sd0  = (dar.sum(), 0, 2)
    sel  = np.arange(peaks_dark[0]-5, peaks_dark[0]+5)
    errs = poisson_sigma(dar[sel], default=0.1)

    gauss_fit_dark = fitf.fit(fitf.gauss, bins[sel], dar[sel], sd0, sigma=errs, bounds=gb0)
    outDict[cpio.generic_params[2]] = (gauss_fit_dark.values[1], gauss_fit_dark.errors[1])
    outDict[cpio.generic_params[3]] = (gauss_fit_dark.values[2], gauss_fit_dark.errors[2])

    ## Scale just in case we lost a different amount of integrals in dark and led
    ## scale = led.sum() / dar.sum()
    scale = 1.5

    ## Take into account the scale in seed finding (could affect Poisson mu)????
    ped_vals    = np.array([gauss_fit_dark.values[0] * scale,
                            gauss_fit_dark.values[1],
                            gauss_fit_dark.values[2]])
    scaler_func = dark_scaler(dar[b1:b2][(bins[b1:b2]>=-5) & (bins[b1:b2]<=5)])

    try:
        seeds, bounds = seeds_and_bounds(sens_type, run_no, ich, scaler_func, bins[b1:b2],
                                     led[b1:b2], ped_vals, 'demopp', gauss_fit_dark.errors, func_name, use_db_gain_seeds)
        if seeds[2] == 0:
            ## Channel was bad but maybe recovered
            seeds, bounds = seeds_and_bounds(sens_type, run_no, ich, scaler_func, bins[b1:b2],
                                             led[b1:b2], ped_vals, 'demopp', gauss_fit_dark.errors, func_name, use_db_gain_seeds=False)
    except RuntimeError:
        print(f'An error occured with channel: {channs[ich]}')
        outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0, 0])
        plt.errorbar(bins, led, xerr=0.5*np.diff(bins)[0], fmt='b.')
        plt.title(f'{detector}: Spe response fit to channel {channs[ich]}')
        plt.xlabel('ADC')
        plt.ylabel('AU')
        plt.xlim(-30, 250)
        plt.show()
        continue

    ## Protect low light channels
    if seeds[1] < 0.2:
        llchans.append(channs[ich])
        ## Dodgy setting of high charge dark bins to zero
        dar[bins>gauss_fit_dark.values[1] + 3*gauss_fit_dark.values[2]] = 0

    if 'dfunc' in func_name:
        respf = ffuncs[func_name](dark_spectrum  = dar[b1:b2] * scale,
                                  pedestal_mean  = gauss_fit_dark.values[1],
                                  pedestal_sigma = gauss_fit_dark.values[2])
    elif 'conv' in func_name:
        respf = ffuncs[func_name](dark_spectrum = dar [b1:b2] * scale,
                                  bins          = bins[b1:b2])
    else:
        respf = ffuncs[func_name]


    ## The fit
    errs = poisson_sigma(led, default=0.001)
    if not 'gau' in func_name:
        #errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dar)
        errs = np.sqrt(errs**2 + np.exp(-2 * seeds[1]) * dar * ((0.1*seeds[1])**2 + 1))

    try:
        rfit = fitf.fit(respf, bins[b1:b2], led[b1:b2], seeds, sigma=errs[b1:b2], bounds=bounds)
        chi = rfit.chi2
    except RuntimeError:
        print(f'Fit doesnt converge, saving zeros for channel {channs[ich]}')
        outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0, 0])
        plot_spec_without_fit(bins, led, errs, channs[ich])
        continue
    except ValueError:
        print(f'x0 is infeasible {channs[ich]}')
        outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0, 0])
        plot_spec_without_fit(bins, led, errs, channs[ich])
        continue


    ## Attempt to catch bad fits and refit (currently only valid for dfunc and conv)
    if chi >= 7 or rfit.values[3] >= 2.5 or rfit.values[3] <= 1:
        ## The offending parameter seems to be the sigma in most cases
        nseed = rfit.values
        nseed[3] = 1.7
        nbound = [(bounds[0][0], bounds[0][1], bounds[0][2], 1),
                  (bounds[1][0], bounds[1][1], bounds[1][2], 2.5)]
        rfit = fitf.fit(respf, bins[b1:b2], led[b1:b2], nseed, sigma=errs[b1:b2], bounds=nbound)
        chi  = rfit.chi2
    if chi >= 10 or rfit.values[2] < 12 or rfit.values[2] > 19 or rfit.values[3] > 3:
        print('Fit with high errors: '+ str(channs[ich]))

    print(f'Channel fit: {rfit.values}, Chi: {chi}')
    if channs[ich] in specialCheck or chi >= 10 or rfit.values[2] < 12 or rfit.values[2] > 19 or rfit.values[3] > 3:
        print(channs[ich])
        plt.errorbar(bins, led, xerr=0.5*np.diff(bins)[0], yerr=errs, fmt='b.')
        plt.plot(bins[b1:b2], respf(bins[b1:b2], *rfit.values), 'r')
        #plt.plot(bins[b1:b2], respf(bins[b1:b2], *seeds), 'g')
        plt.title(f'{detector}: Spe response fit to channel {channs[ich]}')
        plt.xlabel('ADC')
        plt.ylabel('AU')
        plt.xlim(-30, 250)
        plt.show()

    outData.append([channs[ich], rfit.values, rfit.errors, respf.n_gaussians, chi])
    outDict[cpio.generic_params[0]] = (rfit.values[0], rfit.errors[0])
    outDict[cpio.generic_params[1]] = (rfit.values[1], rfit.errors[1])
    gIndx = 2
    if 'gau' in func_name:
        gIndx = 4
    outDict[cpio.generic_params[4]]  = (rfit.values[gIndx]  , rfit.errors[gIndx])
    outDict[cpio.generic_params[5]]  = (rfit.values[gIndx+1], rfit.errors[gIndx+1])
    outDict[cpio.generic_params[-1]] = (respf.n_gaussians, rfit.chi2)
    #outDict[cpio.generic_params[-1]] = (rfit.chi2)
    param_writer(channs[ich], outDict)
    ich += 1
    # except:
    #     print(f'An error occured with channel: {channs[ich]}')
    #     outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0, 0])
    #     continue

## Couple of plots
gainIndx = 2
if 'gau' in func_name:
    gainIndx = 4

pVals = [np.fromiter((ch[1][gainIndx]   for ch in outData), np.float),
         np.fromiter((ch[1][gainIndx+1] for ch in outData), np.float),
         np.fromiter((ch[1][1] for ch in outData), np.float),
         np.fromiter((ch[4]    for ch in outData), np.float)]
out_file.close()

pos_x  = db.DataSiPM(db_file, run_no).X       .values[init_sns:]
pos_y  = db.DataSiPM(db_file, run_no).Y       .values[init_sns:]
channs = db.DataSiPM(db_file, run_no).SensorID.values[init_sns:]

histplot2d(pos_x, pos_y, pVals[3], "Fit chi^2 map")
histplot2d(pos_x, pos_y, pVals[2], "Fit poisson mu")
histplot2d(pos_x, pos_y, pVals[0], "Fit conversion gain")


plot_names = ["Gain", "1pe sigma", "Poisson mu", "chi2"]
fig, axes  = plt.subplots(nrows=2, ncols=2, figsize=(16,6))
chiVs = pVals[3]
for ax, val, nm in zip(axes.flatten(), pVals, plot_names):
    ax.hist(val[(chiVs<10) & (chiVs!=0)], bins=100)
    ax.set_title(nm)
plt.tight_layout()
fig.show()

print('Low light chans:', llchans)
