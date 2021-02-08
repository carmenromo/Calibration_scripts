import os
import sys
import re
import argparse

import numpy  as np
import tables as tb

from invisible_cities.database.load_db import DataSiPM


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--file-in"  , type   = str         , help = "input file"          , required = True)
parser.add_argument("-o", "--file-out" , type   = str         , help = "output file"         , required = False, default = "SiPMnoisePDF_R{run_number}.csv")
parser.add_argument("-p", "--proc-mode", type   = str         , help = "processing mode"     , required = False, default = "mode")
parser.add_argument(      "--override" , action = "store_true", help = "override output file")
parser.add_argument(      "--min-run"  , type   = int         , help = "min run number"      , required = False, default = -1)
parser.add_argument(      "--max-run"  , type   = int         , help = "max run number"      , required = False, default = -1)


CLI = parser.parse_args(sys.argv[1:])

assert                     os.path.exists(CLI.file_in ), "Input file does not exist"
assert CLI.override or not os.path.exists(CLI.file_out), "Output file already exists"


db_file    = '/Users/carmenromoluque/IC/invisible_cities/database/localdb.DEMOPPDB.sqlite3'
run_number = int(re.compile("\d\d\d\d").findall(CLI.file_in)[0])
file_in    = CLI.file_in
file_out   = CLI.file_out.format(run_number = run_number)
group_name =  "HIST"
hist_name  = f"{CLI.proc_mode}"
bins_name  = f"{CLI.proc_mode}_bins"
min_run    = run_number if CLI.min_run == -1 else CLI.min_run
max_run    = "NULL"     if CLI.max_run == -1 else CLI.max_run
sipm_ids   = DataSiPM(db_file, run_number).SensorID

with tb.open_file(file_in) as file_in:
    group = getattr(file_in.root, group_name)
    hist  = getattr(group       ,  hist_name)
    bins  = getattr(group       ,  bins_name)
    nsipm = hist.shape[1]

    # Ensure same binning
    assert hist.shape[-1] == bins.shape[0]

    lines = []
    for sipm_no, entries in enumerate(hist.read().sum(axis=0)):
        progress = 100 * sipm_no // nsipm
        print(f"\rProgress: {progress:>4} %", end="", flush=True)

        sipm_id = sipm_ids[sipm_no]
        probs   = entries / entries.sum() if np.any(entries) else entries

        for energy, probability in zip(bins, probs):
            lines.append(f"{min_run},{max_run},{sipm_id},{energy},{probability}")

print("\rProgress:  100 %")

### .csv file and crompressed
with open(file_out, "w") as f_out:
    header = "MinRun,MaxRun,SensorID,BinEnergyPes,Probability"
    text   = "\n".join([header] + lines + [""])
    f_out.write(text)

print("Compressing...")
os.system(f"bzip2 -zc {file_out} > {file_out + '.bz2'}")


### .txt file
### Change output file: default = "SiPMnoisePDF_{run_number}.txt"
#with open(file_out, "w") as file_out:
#    for l in lines:
#        file_out.write(str(l)+'\n')
