import sys

import numpy  as np
import tables as tb
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

from invisible_cities.database import load_db as DB
from invisible_cities.core.core_functions import weighted_mean_and_std

det_db = 'new'

def compare_runs(run_no, infiles):

    #run_no  = int(sys.argv[1]) # Only for sensor positions and mapping
    #infiles = sys.argv[2:]

    sensor_x = DB.DataSiPM(det_db, run_no).X.values
    sensor_y = DB.DataSiPM(det_db, run_no).Y.values
    chi_vals = []
    mu_vals  = []
    for ifile in infiles:
        chis, mus = pde(run_no, ifile)
        chi_vals.append(chis)
        mu_vals.append(mus)

    fig, axes = plt.subplots(nrows=2, ncols=len(infiles), figsize=(20,6))
    for chis, mus, ax, fl in zip(chi_vals, mu_vals, axes[0], infiles):
        plt_info = ax.scatter(sensor_x[chis<5], sensor_y[chis<5], c=mus[chis<5])
        ax.set_title("Poisson mu map file "+fl)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.colorbar(plt_info, ax=ax)
    plt.tight_layout()
    conditions = (chi_vals[0] < 5) & (chi_vals[1] < 5) & (mu_vals[0] > 0.9) & (mu_vals[1] > 0.9)
    dif_plt = axes[1][0].scatter(sensor_x[conditions], sensor_y[conditions],
                                 c=mu_vals[0][conditions] - mu_vals[1][conditions])
    plt.colorbar(dif_plt, ax=axes[1][0])
    rat_plt = axes[1][1].scatter(sensor_x[conditions], sensor_y[conditions],
                                 c=(mu_vals[0][conditions] / mu_vals[1][conditions])/(mu_vals[0][1024]/mu_vals[1][1024]))
    plt.colorbar(rat_plt, ax=axes[1][1])
    fig.show()
    plt.show()

def pde(run_no, file_name):

    #file_name = sys.argv[1]
    #run_no    = int(sys.argv[2])

    m_channels = DB.DataSiPM(det_db, run_no).Active.values
    sensor_id  = DB.DataSiPM(det_db, run_no).SensorID.values

    ## For some checks
    chans = [19040, 19041, 19042, 19048, 19049, 19050, 19056, 19057, 19058,
             17043, 17044, 17045, 17051, 17052, 17053, 17059, 17060, 17061]

    mu_vals  = []
    chi_vals = []
    with tb.open_file(file_name, 'r') as data_file:

        bins   = np.array(data_file.root.HIST.sipm_dark_bins)
        specsL = np.array(data_file.root.HIST.sipm_spe) .sum(axis=0)
        specsD = np.array(data_file.root.HIST.sipm_dark).sum(axis=0)

        for ich, (led, dar, act) in enumerate(zip(specsL, specsD, m_channels)):

            if not act:
                print('Channel ', sensor_id[ich], ' not active')
                mu_vals.append(0)
                chi_vals.append(0)
                continue

            valid_bins = np.argwhere(led >= 10)
            b1 = valid_bins[0][0]
            b2 = np.argwhere(bins <= 1)[-1][0]

            dscale = led[b1:b2].sum() / dar[b1:b2].sum()

            pfit = leastsq(scale_chi, dscale, args=(dar[b1:b2], led[b1:b2]))
            chi2 = np.sum(scale_chi(pfit[0], dar[b1:b2], led[b1:b2])**2) / (b2 - b1 - 1)
            mu_vals.append(pfit[0][0])
            chi_vals.append(chi2)
            if sensor_id[ich] in chans:
                print('Interesting channel ', sensor_id[ich])
                print('Poisson mu = ', pfit[0][0], ' chi2 = ', chi2)
            ## if not ich%64 or chi2 > 10:
            ##     print('Check: ', pfit, chi2)
            ##     plt.errorbar(bins, led,
            ##                  xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(led), fmt='b.')
            ##     plt.plot(bins, np.exp(-pfit[0]) * dar, 'r')
            ##     plt.title('Scale fit to channel '+str(sensor_id[ich]))
            ##     plt.xlabel('ADC')
            ##     plt.ylabel('AU')
            ##     plt.show()

    ## print(sensor_x.shape, sensor_y.shape, len(chi_vals), len(mu_vals))
    chi_vals = np.array(chi_vals)
    mu_vals = np.array(mu_vals)
    ## plt.scatter(sensor_x, sensor_y, c=chi_vals)
    ## plt.title("Fit chi^2 map")
    ## plt.xlabel("X (mm)")
    ## plt.ylabel("Y (mm)")
    ## plt.colorbar()
    ## plt.show()

    ## plt.scatter(sensor_x[(chi_vals>0) & (chi_vals<10)],
    ##             sensor_y[(chi_vals>0) & (chi_vals<10)],
    ##             c=np.array(mu_vals)[(chi_vals>0) & (chi_vals<10)])
    ## plt.title("Fit Poisson mu map")
    ## plt.xlabel("X (mm)")
    ## plt.ylabel("Y (mm)")
    ## plt.colorbar()
    ## plt.show()
    return chi_vals, mu_vals


def elec_dark_comp(run_no, infiles):

    elec_file = infiles[0]
    dark_file = infiles[1]

    sensor_id  = DB.DataSiPM(det_db, run_no).SensorID.values
    sensor_x   = DB.DataSiPM(det_db, run_no).X.values
    sensor_y   = DB.DataSiPM(det_db, run_no).Y.values

    chans = [19040, 19041, 19042, 19048, 19049, 19050, 19056, 19057, 19058,
             17043, 17044, 17045, 17051, 17052, 17053, 17059, 17060, 17061]

    mu_vals  = []
    chi_vals = []
    dark_mean = []
    elec_mean = []
    with tb.open_file(elec_file) as efile, tb.open_file(dark_file) as dfile:

        bins   = np.array(dfile.root.HIST.sipm_dark_bins)
        specsD = np.array(dfile.root.HIST.sipm_spe      ).sum(axis=0)
        specsE = np.array(efile.root.HIST.sipm_dark     ).sum(axis=0)

        for ich, (dar, ele) in enumerate(zip(specsD, specsE)):

            valid_bins = np.argwhere(dar >= 10)
            b1         = valid_bins[0][0]
            b2         = np.argwhere(bins <= 0)[-1][0]

            dscale     = dar[b1:b2].sum() / ele[b1:b2].sum()

            pfit = leastsq(scale_chi, dscale, args=(ele[b1:b2], dar[b1:b2]))
            chi2 = np.sum(scale_chi(pfit[0], ele[b1:b2], dar[b1:b2])**2) / (b2 - b1 - 1)

            #if sensor_id[ich] in chans:
            ## if pfit[0] < 0:
            ##     print('Interesting channel ', sensor_id[ich])
            ##     print('Poisson mu = ', pfit[0][0], ' chi2 = ', chi2)
            ##     plt.errorbar(bins, dar,
            ##                  xerr=0.5*np.diff(bins)[0],
            ##                  yerr=np.sqrt(dar), fmt='b.')
            ##     plt.errorbar(bins, ele,
            ##                  xerr=0.5*np.diff(bins)[0],
            ##                  yerr=np.sqrt(ele), fmt='g.')
            ##     plt.plot(bins, np.exp(-pfit[0]) * ele, 'r')
            ##     plt.title('Scale fit to channel '+str(sensor_id[ich]))
            ##     plt.xlabel('ADC')
            ##     plt.ylabel('AU')
            ##     plt.show()

            mu_vals .append(pfit[0][0])
            chi_vals.append(chi2)

            dmean, _ = weighted_mean_and_std(bins, dar)
            if dmean > 70.:
                dark_mean.append(2)
                elec_mean.append(0)
                continue
            dark_mean.append(dmean)

            emean, _ = weighted_mean_and_std(bins, ele)
            elec_mean.append(emean)

        a_dark_mean = np.array(dark_mean)
        a_elec_mean = np.array(elec_mean)

        plt.scatter(sensor_x, sensor_y, c=dark_mean)
        plt.title("Mean dark spectrum")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.colorbar()
        plt.show()

        plt.scatter(sensor_x, sensor_y, c=elec_mean)
        plt.title("Mean elec spectrum")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.colorbar()
        plt.show()

        plt.scatter(sensor_x, sensor_y, c=a_dark_mean-a_elec_mean)
        plt.title("Mean dark - elec spectrum")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.colorbar()
        plt.show()

        plt.scatter(sensor_x, sensor_y, c=chi_vals)
        plt.title("Chi^2 map for scale")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.colorbar()
        plt.show()

        plt.scatter(sensor_x, sensor_y, c=mu_vals)
        plt.title("Poisson mu map, average dark counts")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.colorbar()
        plt.show()

def scale_chi(p, dark, led):

    scaled_dark = np.exp(-p[0]) * dark

    return (led - scaled_dark) / np.sqrt(led + scaled_dark)

            
if __name__ == '__main__':

    run_no     = int(sys.argv[1])
    run_type   = sys.argv[2]
    file_names = sys.argv[3:]

    if 'comp' in run_type:
        compare_runs(run_no, file_names)
    elif 'elec' in run_type:
        elec_dark_comp(run_no, file_names)
    else:
        pde(run_no, file_names[0])
