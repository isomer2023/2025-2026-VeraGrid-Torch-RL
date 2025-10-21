# under python 3.8, pandapower 2.2.0, simbench 1.5.3, works perfectly.

import pandapower as pp
import simbench as sb

sb_code = "1-HVMV-urban-2.203-0-no_sw"
net = sb.get_simbench_net(sb_code)

print("✅ Loaded network with", len(net.bus), "buses and", len(net.line), "lines")

pp.runpp(net, algorithm='nr')
print("✅ Power flow converged!")

# Let's do some necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import pandapower as pp
import pandapower.topology as top
import pandapower.plotting as plot
import simbench as sb

# get lists of simbench codes
all_simbench_codes = sb.collect_all_simbench_codes()
all_simbench_code_with_LV_as_lower_voltage_level = sb.collect_all_simbench_codes(lv_level="LV")
complete_data_sb_codes = ["1-complete_data-mixed-all-%i-sw" % scenario for scenario in [0, 1, 2]]
complete_grid_sb_codes = ["1-EHVHVMVLV-mixed-all-%i-sw" % scenario for scenario in [0, 1, 2]]
sb_code1 = "1-MV-rural--0-sw"  # rural MV grid of scenario 0 with full switchs
net = sb.get_simbench_net(sb_code1)

sb_code2 = "1-HVMV-urban-all-0-no_sw"  # urban hv grid with one connected mv grid which has the subnet 2.202
multi_voltage_grid = sb.get_simbench_net(sb_code2)

sb_code3 = "1-HVMV-urban-2.203-0-no_sw"
net_1 = sb.get_simbench_net(sb_code3)

# plot the grid to show the open ring systems
#plot.simple_plot(net)
#plot.simple_plot(net_1)

# let's run a simple power flow calculation while assuming an outage of the first line in feeder 1
outage_line = 1
outage_line_switches = net.switch.index[(net.switch.element == outage_line) & (net.switch.et == "l")]
net.switch.loc[outage_line_switches, "closed"] = False

# resupply feeder 1 via feeder 5
feeder1_buses = net.bus.index[net.bus.subnet.str.contains("Feeder1")]
feeder5_buses = net.bus.index[net.bus.subnet.str.contains("Feeder5")]
loop_line_1_5 = net.line.index[net.line.from_bus.isin(feeder1_buses) & net.line.to_bus.isin(feeder5_buses)]
loop_switches_1_5 = net.switch.index[(net.switch.element == loop_line_1_5[0]) & (net.switch.et == "l")]
net.switch.loc[loop_switches_1_5, "closed"] = True

# run a simple power flow
pp.runpp(net)
print("✅ Power flow converged again!")
# analyze maximal loaded lines
feeder_1_5_lines = net.line.subnet.str.contains("Feeder1") | net.line.subnet.str.contains("Feeder5")
net.res_line.loading_percent.loc[feeder_1_5_lines].max()  # maximal loaded line of Feeder 1 and 5
# -> maximal line loading is less than 100%