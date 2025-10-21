# using env: Py 3.10(VeraGridEngine 5.5.3)
# Aimed to using a pandapowernet, convert it to VeraGrid type, easier for GNN build
import networkx as nx  # For handling graph data structures
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation using DataFrames
import logging  # For logging messages
import random  # For generating random numbers
import warnings
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json

import VeraGridEngine.api as gce  # For interfacing with the GridCal API
import VeraGridEngine.Devices as dev
import VeraGridEngine.Simulations as sim
from VeraGridEngine.Compilers.circuit_to_newton_pa import translate_newton_pa_pf_results, newton_pa_pf
from VeraGridEngine.IO.file_handler import FileOpen
import VeraGridEngine.enumerations as en

import pandapower as pp
import simbench as sb
import pandapower.topology as top  # For topology analysis in Pandapower
import pandapower.plotting as plot  # For plotting in Pandapower
import pandapower.networks as nw

import src.GC_PandaPowerImporter as GC_PandaPowerImporter

sb_code = "1-HV-urban--0-sw"
grid = sb.get_simbench_net(sb_code)
#plot.simple_plot(grid)

gridVG = GC_PandaPowerImporter.PP2GC(grid)
plot.simple_plot(gridVG)
for line in gridVG.lines:
    line.active = True
options = gce.PowerFlowOptions(gce.SolverType.NR, initialize_with_existing_solution=False,control_q=False, verbose=False)
power_flowPP2GC = gce.PowerFlowDriver(gridVG, options)
power_flowPP2GC.run()
print("   ", power_flowPP2GC.results.get_bus_df().tail(4))
print("   ", power_flowPP2GC.results.get_branch_df().tail(1))
print("   power losses:", power_flowPP2GC.results.losses.sum())

#list grid features
print("grid :")
print("name:",gridVG.name)
print("Sbase:",gridVG.Sbase)
print("number of buses:",len(gridVG.buses))
print("number of lines:",len(gridVG.lines))
print("number of loads:",len(gridVG.loads))
print("number of generators:",len(gridVG.generators))
print("number of transformers:",len(gridVG.transformers2w))
print("number of shunts:",len(gridVG.shunts))

