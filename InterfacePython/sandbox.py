#!@Tasmanian_string_python_hashbang@

# necessary import for every use of TASMANIAN
#
import Tasmanian
import numpy as np
import math

#from random import uniform

import matplotlib.pyplot as plt
import matplotlib.colors as cols
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# This file is included for testing, debugging and development reasons
# This file will reside in the cmake build folder, but not the install folder
# Basically, add testing code to this file and it will be automatically
# included as part of the cmake
# You can also use the file without cmake
###############################################################################

print("Add code to this file to test, debug, or develop features")

def model(x):
    eps  = 1.E-2
    kink = 5.0
    if x < math.pi / 8.0:
        return math.exp(x)
    elif x < math.pi / 8.0 + eps:
        return math.exp(x) + kink * (x - math.pi / 8.0)
    else:
        return math.exp(x) + kink * eps


x = np.linspace(-1.0, 1.0, 10000)
y = np.copy(x)

for i in range(len(x)):
    y[i] = model(x[i])


glob  = { "x" : [], "y" : [] }
loc   = { "x" : [], "y" : [] }
mixed = { "x" : [], "y" : [] }


print("Global polynomial case")

for glevel in range(9):
    gridg = Tasmanian.makeGlobalGrid(1, 1, glevel, 'level', 'clenshaw-curtis')
    gpnts = gridg.getPoints()
    gvals = np.zeros((gpnts.shape[0], 1))

    for i in range(gpnts.shape[0]):
        gvals[i, 0] = model(gpnts[i, 0])

    gridg.loadNeededValues(gvals)

    errg = np.max(np.abs( gridg.evaluateBatch(x.reshape(-1,1)).reshape(-1) - y ))
    print(gridg.getNumPoints(), errg)

    if gridg.getNumPoints() < 66:
        glob["x"].append(gridg.getNumPoints())
        glob["y"].append(errg)


print("Local polynomial case")

gridl = Tasmanian.makeLocalPolynomialGrid(1, 1, 2, iOrder = 1, sRule = 'localp')
while gridl.getNumNeeded() > 0:
    gpnts = gridl.getNeededPoints()
    gvals = np.zeros((gpnts.shape[0], 1))

    for i in range(gpnts.shape[0]):
        gvals[i, 0] = model(gpnts[i, 0])

    gridl.loadNeededValues(gvals)

    errg = np.max(np.abs( gridl.evaluateBatch(x.reshape(-1,1)).reshape(-1) - y ))
    print(gridl.getNumPoints(), errg)

    gridl.setSurplusRefinement(1.E-3, 0, "classic")

    if gridl.getNumNeeded() > 0:
        loc["x"].append(gridl.getNumLoaded())
        loc["y"].append(errg)


print("Mixed polynomial case")

gridg = Tasmanian.makeGlobalGrid(1, 1, 2, 'level', 'clenshaw-curtis')
gpnts = gridg.getPoints()
gvals = np.zeros((gpnts.shape[0], 1))
for i in range(gpnts.shape[0]):
    gvals[i, 0] = model(gpnts[i, 0])
gridg.loadNeededValues(gvals)

gridl = Tasmanian.makeLocalPolynomialGrid(1, 1, 1, iOrder = 1, sRule = 'localp')
while gridl.getNumNeeded() > 0:
    gpnts = gridl.getNeededPoints()
    gvals = np.zeros((gpnts.shape[0], 1))

    for i in range(gpnts.shape[0]):
        gvals[i, 0] = model(gpnts[i, 0]) - gridg.evaluateBatch(gpnts[i, :].reshape(1, -1))[0, 0]

    gridl.loadNeededValues(gvals)

    errg = np.max(np.abs( gridg.evaluateBatch(x.reshape(-1,1)).reshape(-1) + gridl.evaluateBatch(x.reshape(-1,1)).reshape(-1) - y ))
    print(gridl.getNumPoints(), errg)

    gridl.setSurplusRefinement(1.E-1, 0, "classic")

    if gridl.getNumNeeded() > 0:
        mixed["x"].append(gridl.getNumLoaded() + 2)
        mixed["y"].append(errg)


plt.figure(1, figsize = (10, 8))

leg1, = plt.semilogy(glob["x"], glob["y"], 'b', label='global grid')
leg2, = plt.semilogy(loc["x"], loc["y"], 'r', label='linear grid')
leg3, = plt.semilogy(mixed["x"], mixed["y"], 'g', label='mixed grid')

plt.legend( handles = [ leg1, leg2, leg3 ], loc = 1, fontsize=18 )

plt.savefig('compare3.png')

plt.figure(2, figsize = (10, 8))

plt.plot(x, y)

plt.savefig('target.png')

plt.show()
