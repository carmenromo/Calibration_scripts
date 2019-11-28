import numpy    as np
import tables   as tb
import argparse

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

parser   = argparse.ArgumentParser(description='Calculation of the mean value of the parameters for different runs.')
parser.add_argument('files', metavar='Name of the files', nargs='+') #positional argument
ns       = parser.parse_args()

files    = ns.files
run_nos  = [f[f.find('R')+1:f.find('R')+5] for f in files]
mean_run = int(round(np.mean(np.array([int(r) for r in run_nos]))))
print(f"Mean values for runs: {run_nos}")
print(f"Created file: sipm_comp_values_R{mean_run}.txt")

sensor_number  = []
gain_list      = [[] for i in range(len(files))]
gain_err_list  = [[] for i in range(len(files))]
sigma_list     = [[] for i in range(len(files))]
sigma_err_list = [[] for i in range(len(files))]
mu_list        = [[] for i in range(len(files))]
mu_err_list    = [[] for i in range(len(files))]
chi2_list      = [[] for i in range(len(files))]


for k, file in enumerate(files):
    with open(file, 'r') as df:
        lines  = df.read().split('\n')
        for line in lines:
            if line == '':
                continue
            _, _, sens_no, gain, gain_err, sigma, sigma_err, poiss_mu, poiss_mu_err, chi2 = [num(x) for x in line.split(',')]
            if k == 0:
                sensor_number.append(sens_no)
            gain_list     [k].append(gain        )
            gain_err_list [k].append(gain_err    )
            sigma_list    [k].append(sigma       )
            sigma_err_list[k].append(sigma_err   )
            mu_list       [k].append(poiss_mu    )
            mu_err_list   [k].append(poiss_mu_err)
            chi2_list     [k].append(chi2        )

gain_mean      = np.mean       (np.array([j for j in      gain_list]), axis=0)
gain_err_mean  = np.linalg.norm(np.array([j for j in  gain_err_list]), axis=0)
sigma_mean     = np.mean       (np.array([j for j in     sigma_list]), axis=0)
sigma_err_mean = np.linalg.norm(np.array([j for j in sigma_err_list]), axis=0)
mu_mean        = np.mean       (np.array([j for j in        mu_list]), axis=0)
mu_err_mean    = np.linalg.norm(np.array([j for j in    mu_err_list]), axis=0)
chi2_mean      = np.mean       (np.array([j for j in      chi2_list]), axis=0)


#param_names = ['MinRun', 'MaxRun', 'SensorID', 'Centroid', 'ErrorCentroid', 'Sigma', 'ErrorSigma']
with open('sipm_comp_values_R'+str(mean_run)+'.txt', 'w') as out_file:
    for n,sens in enumerate(sensor_number):
        out_file.write(str(mean_run)+',100000,'+str(sens)             +','
                        +str(gain_mean [n])+','+str(gain_err_mean [n])+','
                        +str(sigma_mean[n])+','+str(sigma_err_mean[n])+','
                        +str(mu_mean   [n])+','+str(mu_err_mean   [n])+','
                        +str(chi2_mean [n])+'\n')
