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

level = 4
ll = [level, level]
depth = 2 * level

sgrid = Tasmanian.makeLocalPolynomialGrid(2, 1, level, 1, 'localp')
dgrid = Tasmanian.makeLocalPolynomialGrid(2, 1, depth, 1, 'localp', ll)


sgrid.plotPoints2D()
plt.savefig('boundary-sparse-grid{0}.png'.format(level))

dgrid.plotPoints2D()
plt.savefig('boundary-dense-grid{0}.png'.format(level))

plt.show()
