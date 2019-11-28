import argparse
import numpy             as np
import matplotlib.pyplot as plt

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

col = ['k', 'r', 'b', 'g']


#### POSSIBILITY OF TAKING THE NAME OF THE FILES AS INPUTS
parser = argparse.ArgumentParser(description='Upload the DB. Calculation of the mean value of the parameters for different runs.')
parser.add_argument('files', metavar='Name of the files', nargs='+') #positional argument
ns     = parser.parse_args()

files    = ns.files
run_nos  = [f[f.find('R')+1:f.find('R')+5] for f in files]
fResults = [[] for i in range(len(files))]

for i, file in enumerate(files):
    gain_list = []
    with open(file) as df:
        lines  = df.read().split('\n')
        for line in lines:
            if line == '':
                continue
            run_no, _, sens_no, gain, gain_err, sigma, sigma_err = [num(x) for x in line.split(',')]
            gain_list.append(gain)
    fResults[i].append(np.array(gain_list))

## Representing the DIFFERENCE between the values of different runs:
fig, ax  = plt.subplots(figsize=(8,6))
r0vals     = np.array(fResults[0])
months = ['July', 'Aug', 'Sep', 'Oct']
month0 = months[0]
for j, month in zip(range(1, len(fResults)), months[1:]):
    valDiff = np.array(fResults[j]) - r0vals #Substraction of arrays. valDiff is another array
    ax.hist(valDiff[0], bins=np.linspace(-13, 13, 120), log=True, alpha=0.7, label='R'+str(run_nos[j])+' ('+month+') - R'+str(run_nos[0])+' ('+month0+')') #histtype='step'
ax.set_title('SiPM gain differences', fontsize=14)
ax.set_xlabel('diff gain (ADC)', fontsize=12)
plt.rcParams["font.size"] = 11
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)

plt.legend()
plt.tight_layout()
fig.show()
catcher = input('If you want to save the plot press s')
if 's' in catcher:
    fig.savefig('/Users/carmenromoluque/Calibration/Collaboration_meeting/november_2019/sipmRunDifferencePlots.png')


#### Representing the ABSOLUTE values:
#fig, ax    = plt.subplots()
#r0vals     = fResults[0]
#
#print(len(fResults[1]))
#for j, c in zip(range(0, len(fResults)), col):
#    ax.hist(fResults[j], bins=100, log=True, label='R'+str(run_nos[j]), color=c, alpha=0.8)
#ax.set_title('Gain')
#ax.set_xlim([2, 23])
#plt.legend()
#plt.tight_layout()
#fig.show()
#catcher = input('If you want to save the plot press s')
#if 's' in catcher:
#    fig.savefig('/Users/carmenromoluque/Calibration/Collaboration_meeting/june_2019/sipmRunPlotsAbs.png')
