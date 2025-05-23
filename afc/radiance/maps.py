# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Default facade mappings.
"""

# Conversion maps
shade_map_0x6 = {
    '[0, 0, 0]': [0,0,0,0,0,0], # 100 % dn
    '[1, 0, 0]': [1,0,0,0,0,0], # 80 %
    '[2, 0, 0]': [1,1,0,0,0,0], # 60 %
    '[2, 1, 0]': [1,1,1,0,0,0], # 40 %
    '[2, 2, 0]': [1,1,1,1,0,0], # 33 %
    '[2, 2, 1]': [1,1,1,1,1,0], # 13 %
    '[2, 2, 2]': [1,1,1,1,1,1]} # 0 % dn
height_shade_map_0x6 = [100, 80, 60, 40, 33, 13, 0]

shade_map_0x4 = {
    '[0, 0, 0]': [0,0,0,0,0], # 100 % dn
    '[1, 0, 0]': [1,0,0,0,0], # 80 %
    '[2, 0, 0]': [1,1,0,0,0], # 60 %
    '[2, 1, 0]': [1,1,1,0,0], # 40 %
    '[2, 2, 0]': [1,1,1,1,0], # 33 %
    '[2, 2, 2]': [1,1,1,1,1]} # 0 % dn
height_shade_map_0x4 = [100, 80, 60, 40, 33, 0]

blinds_map = {
    '[1, 1, 1]': [12], # clear
    '[1, 0, 0]': [1],  # -15
    '[2, 0, 0]': [2],  # -30
    '[3, 0, 0]': [3],  # -45
    '[4, 0, 0]': [4],  # -60
    '[5, 0, 0]': [5],  # -75
    '[1, 0, 1]': [6],  # 0
    '[0, 0, 1]': [7],  # 15
    '[0, 0, 2]': [8],  # 30
    '[0, 0, 3]': [9],  # 45
    '[0, 0, 4]': [10], # 60
    '[0, 0, 5]': [11], # 75
    '[0, 0, 0]': [0]}  # 90
#height_blinds_map = [0, 30, 60, 100, 100, 100, 0, 30, 60, 100, 100, 100, 100]
height_blinds_map = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
