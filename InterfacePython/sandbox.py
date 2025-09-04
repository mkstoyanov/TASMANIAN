#!@Tasmanian_string_python_hashbang@

# necessary import for every use of TASMANIAN
#
import Tasmanian
import numpy as np
import math

#from random import uniform

import matplotlib.pyplot as plt
import matplotlib.colors as cols
#from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# This file is included for testing, debugging and development reasons
# This file will reside in the cmake build folder, but not the install folder
# Basically, add testing code to this file and it will be automatically
# included as part of the cmake
# You can also use the file without cmake
###############################################################################

print("Add code to this file to test, debug, or develop features")

def weight(x):
    return 1

integrand = lambda x : np.exp(-np.linalg.norm(x) ** 2)
sinc1s0   = lambda x : np.sinc(x / np.pi)
sinc10s1  = lambda x : np.sinc(10 * (x - 1) / np.pi)

ref1 = 1.4321357541271255E-00
ref2 = 6.4062055930705356E-02

def compute_integral(grid, integrand):
    return np.sum(np.apply_along_axis(integrand, 1, grid.getPoints()) * grid.getQuadratureWeights())

level = 5
rule  = 'rleja'
shift = 1

nref = 10

ct = Tasmanian.createNestedExoticQuadratureFromFunction(level, rule, shift, sinc1s0, nref, 'nested-custom')

grid = Tasmanian.makeGlobalGridCustom(1, 0, level-1, "level", ct)

p = grid.getPoints()
w = grid.getQuadratureWeights()

print(math.fabs( compute_integral(grid, integrand) - ref1 ) )


# ct = Tasmanian.createNestedExoticQuadratureFromFunction(level, rule, shift, weight, nref, 'nested-custom')
#
# grid = Tasmanian.makeGlobalGridCustom(1, 0, level-1, "level", ct)
#
# print(grid.getPoints())
# print(grid.getQuadratureWeights())
#
# refgrid = Tasmanian.makeGlobalGrid(1, 0, level -1, 'level', rule)
#
# print(refgrid.getPoints())
# print(refgrid.getQuadratureWeights())
