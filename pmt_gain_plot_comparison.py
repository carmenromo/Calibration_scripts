import argparse
import numpy             as np
import matplotlib.pyplot as plt

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


#### POSSIBILITY OF TAKING THE NAME OF THE FILES AS INPUTS
parser = argparse.ArgumentParser(description='Representing the evolution of the gain.')
parser.add_argument('files', metavar='Name of the files', nargs='+') #positional argument
ns     = parser.parse_args()

files    = ns.files
run_nos  = [f[f.find('R')+1:f.find('R')+5] for f in files]
fResults = [[] for i in range(len(files))]

for i, file in enumerate(files):
    gain_list     = []
    gain_err_list = []
    with open(file) as df:
        lines  = df.read().split('\n')
        for line in lines:
            if line == '':
                continue
            run_no, _, sens_no, gain, gain_err, sigma, sigma_err = [num(x) for x in line.split(',')]
            gain_list    .append(    gain)
            gain_err_list.append(gain_err)
    fResults[i].append(np.array(gain_list    ))
    fResults[i].append(np.array(gain_err_list))


### Representing the DIFFERENCE between the values of different runs:
fig, ax = plt.subplots()
all_channel_nos = np.arange(12)
months = ['July', 'Aug', 'Sept', 'Oct']
for j, (vals1, run, month) in enumerate(zip(fResults, run_nos, months)):
    channs = all_channel_nos
    vals   = vals1
    ax.errorbar(channs, vals[0], yerr=vals[1], label='R'+run+' ('+month+')')
ax.set_title('Gain vs. channel number')
ax.set_xlabel('Channel number')
ax.set_ylim([15, 35])
ax.legend()

plt.tight_layout()
fig.show()
plt.show()
catcher = input('If you want to save the plot press s')
if 's' in catcher:
    fig.savefig('/Users/carmenromoluque/Calibration/Collaboration_meeting/november_2019/gain_comp.png')
