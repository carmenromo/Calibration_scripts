import argparse
import numpy             as np
import matplotlib.pyplot as plt

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


#### POSSIBILITY OF TAKING THE NAME OF THE FILES AS INPUTS
parser = argparse.ArgumentParser(description='Upload the DB. Calculation of the mean value of the parameters for different runs.')
parser.add_argument('files', metavar='Name of the files', nargs='+') #positional argument
ns     = parser.parse_args()

files    = ns.files
run_nos  = [f[f.find('R')+1:f.find('R')+5] for f in files]
fResults = [[] for i in range(len(files))]

for i, file in enumerate(files):
    ped_sigma_list  = []
    mu_list         = []
    gain_list       = []
    sigma_list      = []
    chi2_list       = []
    with open(file) as df:
        lines  = df.read().split('\n')
        for line in lines:
            if line == '':
                continue
            run_no, h, sens_no, ped_sigma, ped_sigma_err, poiss_mu, poiss_mu_err, gain, gain_err, sigma, sigma_err, chi2 = [num(x) for x in line.split(',')]
            ped_sigma_list.append(    ped_sigma)
            ped_sigma_list.append(ped_sigma_err)
            mu_list       .append(     poiss_mu)
            mu_list       .append( poiss_mu_err)
            gain_list     .append(         gain)
            gain_list     .append(     gain_err)
            sigma_list    .append(        sigma)
            sigma_list    .append(    sigma_err)
            chi2_list     .append(         chi2)

    fResults[i].append(np.array(ped_sigma_list))
    fResults[i].append(np.array(       mu_list))
    fResults[i].append(np.array(     gain_list))
    fResults[i].append(np.array(    sigma_list))
    fResults[i].append(np.array(     chi2_list))


### Representing the DIFFERENCE between the values of different runs:

axistitles = ['Pedestal sigma vs. channel number'       ,
              'Normalised Poisson mu vs. channel number',
              'Gain vs. channel number'                 ,
              '1pe sigma vs. channel number'            ,
              'Fit chi^2'                               ,
              'Legend'                                  ]
all_channel_nos = np.arange(12)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20,6))
for j, (vals1, run) in enumerate(zip(fResults, run_nos)):
    #print(vals1)
    channs = all_channel_nos
    vals   = vals1
    for k, (ax, axtit) in enumerate(zip(axes.flatten(), axistitles)):
        if j == 0:
            #ax.set_prop_cycle(cycler('color', [cm(1.*i/vals.shape[0]) for i in range(vals.shape[0])]))
            ax.set_title(axtit) # Vale, esto para cada k
            ax.set_xlabel('Channel number')
        if k < 4:
            if k == 1:
                ## We want to normalise to the maximum value here.
                maxV = vals[k][ ::2].max()
                maxE = vals[k][1::2].max()
                vp   = vals[k][ ::2]/maxV
                vpE = np.fromiter((z*np.sqrt((ex/x)**2+(maxE/maxV)**2) for z, x, ex in zip(vp, vals[k][::2], vals[k][1::2])), np.float)
                ax.errorbar(channs, vp, yerr=vpE, label='Run '+run)
            else:
                ax.errorbar(channs, vals[k][::2], yerr=vals[k][1::2], label='Run '+run)
        elif k == 4:
            ax.plot(channs, vals[k], label='Run '+run)
        else:
            ## trick to put legend on empty subplot
            ax.plot(0,0, label='Run '+run)
            ax.legend(ncol=2)
plt.tight_layout()
fig.show()
plt.show()
fig.savefig('pmtRunDifferencePlots.png')
