import os
import sys
import argparse

from ROOT import TFile
from ROOT import TH1F

parser = argparse.ArgumentParser()
parser.add_argument("-i", '--file-in' , type=str, help="input file" , required=True)
parser.add_argument("-o", '--file-out', type=str, help="output file", required=False, default = "SiPMnoisePDF_{run_number}.csv")
parser.add_argument("--override", action="store_true", help="override output file")

CLI = parser.parse_args(sys.argv[1:])

assert                     os.path.exists(CLI.file_in ), "Input file does not exist"
assert CLI.override or not os.path.exists(CLI.file_out), "Output file already exists"


file_in    = TFile(CLI.file_in)
run_number = CLI.file_in.split("_R")[-1].split(".root")[0]
file_out   = open(CLI.file_out.format(run_number = run_number), "w")

min_run = run_number
max_run = "NULL"

h = None
buffer = "MinRun, MaxRun, SensorID, BinEnergyPes, Probability\n"
for dice in range(1, 29):
    print "DICE", dice
    for i_sipm in range(64):
        ID = dice*1000 + i_sipm
        hname = "PDF1/SiPM_{}".format(ID)

        previous_h = h
        h = file_in.Get(hname)
        try:
            h.Scale(1/h.Integral())
        except AttributeError:
            h = previous_h
            h.Reset()
        except ZeroDivisionError:
            pass
        xaxis = h.GetXaxis()
        for i in range(1, h.GetNbinsX() + 1):
            e  = xaxis.GetBinCenter(i)
            p  = h.GetBinContent(i)
            buffer +=  "{min_run}, {max_run}, {ID}, {e}, {p}\n".format(**locals())
    file_out.write(buffer)
    buffer = ""
