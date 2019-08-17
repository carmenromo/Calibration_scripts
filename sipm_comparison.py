import argparse
import numpy             as np
import matplotlib.pyplot as plt

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


#### POSSIBILITY OF TAKING RUN NUMBERS AS INPUTS
#parser = argparse.ArgumentParser(description='Comparison of parameters for different runs.')
#parser.add_argument('run_nos', metavar='Run numbers', type=int, nargs='+')
#ns     = parser.parse_args()
#
#run_nos  = ns.run_nos
#files    = []
#fResults = [[] for i in range(len(run_nos))]
#
#file_path = '/Users/carmenromoluque/Calibration/october_cal/SiPM/spectra/'
#for run_no in run_nos:
#    name_file = file_path+'sipmGain_R'+str(run_no)+'.txt'
#    files.append(name_file)


#### POSSIBILITY OF TAKING THE NAME OF THE FILES AS INPUTS
parser = argparse.ArgumentParser(description='Upload the DB. Calculation of the mean value of the parameters for different runs.')
parser.add_argument('files', metavar='Name of the files', nargs='+') #positional argument
ns     = parser.parse_args()

files    = ns.files
run_nos  = [f[f.find('R')+1:f.find('R')+5] for f in files]
fResults = [[] for i in range(len(files))]

#initial_sipm = 579
#final_sipm   = 640
for i, file in enumerate(files):
    gain_list  = []
    sigma_list = []
    mu_list    = []
    chi2_list  = []
    with open(file) as df:
        lines  = df.read().split('\n')
        for j, line in enumerate(lines):
            if line == '':
                continue
            #if j > initial_sipm and j < final_sipm:
            run_no, h, sens_no, gain, gain_err, sigma, sigma_err, poiss_mu, poiss_mu_err, chi2 = [num(x) for x in line.split(',')]
            gain_list .append(    gain)
            sigma_list.append(   sigma)
            mu_list   .append(poiss_mu)
            chi2_list .append(    chi2)

    fResults[i].append(np.array( gain_list))
    fResults[i].append(np.array(sigma_list))
    fResults[i].append(np.array(   mu_list))
    fResults[i].append(np.array( chi2_list))


### Representing the DIFFERENCE between the values of different runs:

axistitles = ['Gain differences', '1pe sigma differences', 'Poisson mu differences', 'Chi2 differences']
fig, axes  = plt.subplots(nrows=2, ncols=2, figsize=(20,6))
r0vals     = fResults[0]
for j in range(1, len(fResults)):
    for k, (ax, val, axtit) in enumerate(zip(axes.flatten(), fResults[j], axistitles)):
        valDiff = val - r0vals[k] #Substraction of arrays. valDiff is another array
        if k == 0:
            print('Run ', run_nos[j])
            ax.hist(valDiff, bins=np.linspace(-15, 15, 120), log=True, alpha=0.7, label='R'+str(run_nos[j])+' - R'+str(run_nos[0])) #histtype='step'
            #array_bad_sensors = np.where(np.abs(valDiff) >= 2.5)
            #if any(np.abs(valDiff) >= 1):
            #print(np.where(np.abs(valDiff) >= 1), val)
        elif k == 1:
            ax.hist(valDiff, bins=np.linspace(-2.0, 2.0, 120), log=True, alpha=0.7, label='Difference R'+str(run_nos[j])+' - R'+str(run_nos[0]))
        elif k == 2:
            ax.hist(valDiff, bins=np.linspace(-2.5, 2.5, 120), log=True, alpha=0.7, label='Difference R'+str(run_nos[j])+' - R'+str(run_nos[0]))
        else:
            ax.hist(valDiff, bins=np.linspace(-150, 150, 120), log=True, alpha=0.7, label='Difference R'+str(run_nos[j])+' - R'+str(run_nos[0]))
        if j == 1:
            ax.set_title(axtit)

plt.legend()
plt.tight_layout()
fig.show()
catcher = input('If you want to save the plot press s')
if 's' in catcher:
    fig.savefig('sipmRunDifferencePlots.png')


##### Representing the ABSOLUTE values:
#axistitles = ['Gain', '1pe sigma', 'Poisson mu', 'Chi2']
#fig, axes  = plt.subplots(nrows=2, ncols=2, figsize=(20,8))
#r0vals     = fResults[0]
#
#for j in range(0, len(fResults)):
#    for k, (ax, val, axtit) in enumerate(zip(axes.flatten(), fResults[j], axistitles)):
#        if k == 0:
#            #print('Run ', run_nos[j])
#            ax.hist(val, bins=100, log=True,
#                    label='R'+str(run_nos[j])+' - R'+str(run_nos[0]))
#        else:
#            ax.hist(val, bins=100, log=True,
#                    label='R'+str(run_nos[j])+' - R'+str(run_nos[0]))
#        #if j == 1:
#        ax.set_title(axtit)
#
##plt.legend()
#plt.tight_layout()
#fig.show()
#catcher = input('If you want to save the plot press s')
#if 's' in catcher:
#    fig.savefig('sipmRunPlots.png')
