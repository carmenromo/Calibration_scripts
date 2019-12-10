import numpy    as np
import tables   as tb
import argparse

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

parser = argparse.ArgumentParser(description='Calculation of the mean value of the parameters for different runs.')
parser.add_argument('files', metavar='Name of the files', nargs='+') #positional argument
ns       = parser.parse_args()

files    = ns.files
run_nos  = [f[f.find('R')+1:f.find('R')+5] for f in files]
mean_run = int(round(np.mean(np.array([int(r) for r in run_nos]))))
print(f"Mean values for runs: {run_nos}")
print(f"Created file: pmt_comp_values_R{mean_run}.txt")

sensor_number      = []
ped_sigma_list     = [[] for i in range(len(files))]
ped_sigma_err_list = [[] for i in range(len(files))]
poiss_mu_list      = [[] for i in range(len(files))]
poiss_mu_err_list  = [[] for i in range(len(files))]
gain_list          = [[] for i in range(len(files))]
gain_err_list      = [[] for i in range(len(files))]
sigma_list         = [[] for i in range(len(files))]
sigma_err_list     = [[] for i in range(len(files))]
n_gauss_chi2_list  = [[] for i in range(len(files))]

for k, file in enumerate(files):
    with open(file, 'r') as df:
        lines  = df.read().split('\n')
        for line in lines:
            if line == '':
                continue
            _, _, sens_no, p_sig, p_sig_err, poiss, poiss_err, gain, gain_err, sigma, sigma_err, ng_chi2 = [num(x) for x in line.split(',')]
            if k == 0:
                sensor_number.append(sens_no)
            ped_sigma_list    [k].append(p_sig    )
            ped_sigma_err_list[k].append(p_sig_err)
            poiss_mu_list     [k].append(poiss    )
            poiss_mu_err_list [k].append(poiss_err)
            gain_list         [k].append(gain     )
            gain_err_list     [k].append(gain_err )
            sigma_list        [k].append(sigma    )
            sigma_err_list    [k].append(sigma_err)
            n_gauss_chi2_list [k].append(ng_chi2  )



p_sig_mean     = np.mean       (np.array([j for j in     ped_sigma_list]), axis=0)
p_sig_err_mean = np.linalg.norm(np.array([j for j in ped_sigma_err_list]), axis=0)
poiss_mean     = np.mean       (np.array([j for j in      poiss_mu_list]), axis=0)
poiss_err_mean = np.linalg.norm(np.array([j for j in  poiss_mu_err_list]), axis=0)
gain_mean      = np.mean       (np.array([j for j in          gain_list]), axis=0)
gain_err_mean  = np.linalg.norm(np.array([j for j in      gain_err_list]), axis=0)
sigma_mean     = np.mean       (np.array([j for j in         sigma_list]), axis=0)
sigma_err_mean = np.linalg.norm(np.array([j for j in     sigma_err_list]), axis=0)
ng_chi2_mean   = np.mean       (np.array([j for j in  n_gauss_chi2_list]), axis=0)


with open('pmt_comp_values_R'+str(mean_run)+'.txt', 'w') as out_file:
    for n,sens in enumerate(sensor_number):
        out_file.write(str(mean_run)+', 100000, '+str(sens)+', '
                        +str(  p_sig_mean[n])+', '+str(p_sig_err_mean[n])+', '
                        +str(  poiss_mean[n])+', '+str(poiss_err_mean[n])+', '
                        +str(   gain_mean[n])+', '+str( gain_err_mean[n])+', '
                        +str(  sigma_mean[n])+', '+str(sigma_err_mean[n])+', '
                        +str(ng_chi2_mean[n])+'\n')

