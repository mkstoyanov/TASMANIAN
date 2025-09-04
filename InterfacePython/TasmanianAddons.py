##############################################################################################################################################################################
# Copyright (c) 2017, Miroslav Stoyanov
#
# This file is part of
# Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
#    or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND IMPLIED.
# THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT,
# COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE.
# THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF,
# IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
##############################################################################################################################################################################

from ctypes import c_int, c_double, c_char_p, c_void_p, POINTER, CFUNCTYPE, CDLL, RTLD_GLOBAL
import numpy as np
import sys

import TasmanianConfig
import TasmanianSG
TasmanianInputError = TasmanianConfig.TasmanianInputError

pLibCTSG = CDLL(TasmanianConfig.__path_libcaddons__, mode = RTLD_GLOBAL)

type_1Dfunc = CFUNCTYPE(c_double, c_double)
type_lpnmodel = CFUNCTYPE(None, c_int, POINTER(c_double), c_int, POINTER(c_double), c_int, POINTER(c_int))
type_scsmodel = CFUNCTYPE(None, c_int, c_int, POINTER(c_double), c_int, POINTER(c_double), c_int, POINTER(c_int))
type_icsmodel = CFUNCTYPE(None, c_int, c_int, POINTER(c_double), c_int, c_int, POINTER(c_double), c_int, POINTER(c_int))

pLibCTSG.tsgLoadNeededValues.argtypes = [c_int, type_lpnmodel, c_void_p, c_int, POINTER(c_int)]

pLibCTSG.tsgConstructSurrogateNoIGSurplus.argtypes = [type_scsmodel, c_int, c_int, c_int, c_void_p, c_double, c_char_p,
                                                      c_int, POINTER(c_int), c_char_p, POINTER(c_int)]
pLibCTSG.tsgConstructSurrogateNoIGAniso.argtypes = [type_scsmodel, c_int, c_int, c_int, c_void_p, c_char_p, c_int, POINTER(c_int), c_char_p, POINTER(c_int)]
pLibCTSG.tsgConstructSurrogateNoIGAnisoFixed.argtypes = [type_scsmodel, c_int, c_int, c_int, c_void_p, c_char_p,
                                                         POINTER(c_int), POINTER(c_int), c_char_p, POINTER(c_int)]

pLibCTSG.tsgConstructSurrogateWiIGSurplus.argtypes = [type_icsmodel, c_int, c_int, c_int, c_void_p, c_double, c_char_p,
                                                      c_int, POINTER(c_int), c_char_p, POINTER(c_int)]
pLibCTSG.tsgConstructSurrogateWiIGAniso.argtypes = [type_icsmodel, c_int, c_int, c_int, c_void_p, c_char_p, c_int, POINTER(c_int), c_char_p, POINTER(c_int)]
pLibCTSG.tsgConstructSurrogateWiIGAnisoFixed.argtypes = [type_icsmodel, c_int, c_int, c_int, c_void_p, c_char_p,
                                                         POINTER(c_int), POINTER(c_int), c_char_p, POINTER(c_int)]

pLibCTSG.tsgLoadUnstructuredDataL2.argtypes = [POINTER(c_double), c_int, POINTER(c_double), c_double, c_void_p]

pLibCTSG.tsgCreateExoticQuadratureFromGrid.argtypes = [c_void_p, c_int, c_double, c_void_p, c_char_p, c_int]
pLibCTSG.tsgCreateExoticQuadratureFromFunction.argtypes = [c_void_p, c_int, c_double, type_1Dfunc, c_int, c_char_p, c_int]
pLibCTSG.tsgCreateNestedExoticQuadratureFromFunction.argtypes = [c_void_p, c_int, c_char_p, c_double, type_1Dfunc, c_int, c_char_p]

def tsgLnpModelWrapper(oUserModel, iSizeX, pX, iSizeY, pY, iThreadID, pErrInfo):
    '''
    DO NOT CALL DIRECTLY
    This is callback from C++, see TasGrid::loadNeededValues()

    Creates an interface between a user callable object and
    the Tasmanian callback from C++.
    The callback passes in two raw-arrays with given sizes
    (that correspond to grid inputs and outputs)
    and the id of the running thread.
    The raw-array is wrapped in a numpy array structure and given
    to the user, the result is written to the output array.

    oUserModel: is callable object, e.g., function or lambda

    iSizeX: number of entries in x, equal to getNumDimensions()
    pX:     raw-array corresponding to the model inputs
    iSizeY: number of entries in y, equal to getNumOutputs()
    pY:     raw-array corresponding to the model outputs

    iThreadID: the id of the running thread
    '''
    pErrInfo[0] = 1
    aX = np.ctypeslib.as_array(pX, (iSizeX,))
    aY = np.ctypeslib.as_array(pY, (iSizeY,))
    aResult = oUserModel(aX, iThreadID)
    if aResult.shape != (iSizeY,):
        if TasmanianConfig.enableVerboseErrors:
            print("ERROR: incorrect model output dimensions, should be (iNumOutputs,)")
        return
    aY[0:iSizeY] = aResult[0:iSizeY]
    pErrInfo[0] = 0


def loadNeededValues(callableModel, grid, iNumThreads = 1):
    '''
    Wrapper to TasGrid::loadNeededValues(), non-overwrite version.

    If the grid has needed points, the callableModel will be called
    for each grid point (i.e., model input) and the resulting values
    will be loaded in the grid.

    callableModel: is callable object, e.g., function or lambda
        The object must accept two inputs and give one output:
        aY = callableModel(aX, iThreadID)

        aX: is a one dimensional numpy.ndarray with size equal
            to the number of model inputs

        iThreadID: is the ID of the thread executing the model,
            always between 0 and iNumThreads -1
            Two simultaneous calls to callableModel() will always
            have different ids.

        Return: aY must be a one dimensional numpy.ndarray with
            size equal to the number of model outputs

        Note: if iNumThreads > 1, then callableModel() must be thread-safe.

    grid: must be an instance of Tasmanian.SparseGrid()
        model values will be loaded in the grid

    iNumThreads: integer, if greater than 1 the model will be called
        in parallel from multiple threads.
        See TasGrid::loadNeededValues().

    '''
    iOverwrite = 0 # do not overwrite
    pErrorCode = (c_int * 1)()
    pLibCTSG.tsgLoadNeededValues(iOverwrite,
                                 type_lpnmodel(lambda nx, x, ny, y, tid, err : tsgLnpModelWrapper(callableModel, nx, x, ny, y, tid, err)),
                                 grid.pGrid, iNumThreads, pErrorCode)
    if pErrorCode[0] != 0:
        raise TasmanianInputError("loadNeededValues", "An error occurred during the call to Tasmanian.")

def loadNeededPoints(callableModel, grid, iNumThreads = 1):
    '''
    Alias to loadNeededValues()
    '''
    loadNeededValues(callableModel, grid, iNumThreads)

def reloadLoadedValues(callableModel, grid, iNumThreads = 1):
    '''
    Wrapper to TasGrid::loadNeededPoints(), overwrite version.

    Clears any pending refinement (i.e., needed points) and overwrites the model
    values associated with the existing loaded points.

    The inputs are identical to Tasmanian.loadNeededPoints().

    '''
    iOverwrite = 1 # do overwrite
    pErrorCode = (c_int * 1)()
    pLibCTSG.tsgLoadNeededValues(iOverwrite,
                                 type_lpnmodel(lambda nx, x, ny, y, tid, err : tsgLnpModelWrapper(callableModel, nx, x, ny, y, tid, err)),
                                 grid.pGrid, iNumThreads, pErrorCode)
    if pErrorCode[0] != 0:
        raise TasmanianInputError("reloadLoadedPoints()", "An error occurred during the call to Tasmanian.")

def reloadLoadedPoints(callableModel, grid, iNumThreads = 1):
    '''
    Alias to reloadLoadedValues()
    '''
    reloadLoadedValues(callableModel, grid, iNumThreads)


###############################################################################
################### Construct Surrogate #######################################
###############################################################################

def tsgScsModelWrapper(oUserModel, iNumSamples, iNumDims, pX, iNumOuts, pY, iThreadID, pErrInfo):
    '''
    DO NOT CALL DIRECTLY
    This is callback from C++, see TasGrid::constructSurrogate()

    Handles the case of batch models:
    oUserModel: user defined model that takes a two dimensional
        array of inputs and returns a two dimensional array
        of outputs, and a thread ID

    iNumSamples: number of samples in the batch
    iNumDims:    number of model inputs per sample
    iNumOuts:    number of model outputs per sample

    pX and pY are 2D arrays of c_doubles with size iNumSamples
        times iNumDims and iNumOuts respectively

    iThreadID: the id of the running thread

    '''
    pErrInfo[0] = 1
    aX = np.ctypeslib.as_array(pX, (iNumSamples,iNumDims))
    aY = np.ctypeslib.as_array(pY, (iNumSamples,iNumOuts))
    aResult = oUserModel(aX, iThreadID)
    if aResult.shape != (iNumSamples, iNumOuts):
        if TasmanianConfig.enableVerboseErrors:
            print("ERROR: incorrect model output dimensions, should be (iNumSamples, iNumOutputs)")
        return
    aY[0:iNumSamples, 0:iNumOuts] = aResult[0:iNumSamples, 0:iNumOuts]
    pErrInfo[0] = 0

def tsgIcsModelWrapper(oUserModel, iNumSamples, iNumDims, pX, iHasGuess, iNumOuts, pY, iThreadID, pErrInfo):
    '''
    DO NOT CALL DIRECTLY
    This is callback from C++, see TasGrid::constructSurrogate()

    See tsgScsModelWrapper(), the only difference is that
    the user model oUserModel() takes two arrays,
    one with the inputs and one with the initial guess.
    The initial guess could be empty.

    iHasGuess is a boolean that determines whether an
    initial guess has been loaded in pY.
    '''
    pErrInfo[0] = 1
    aX = np.ctypeslib.as_array(pX, (iNumSamples,iNumDims))
    aY = np.ctypeslib.as_array(pY, (iNumSamples,iNumOuts))
    if (iHasGuess == 0): # no guess
        aResult = oUserModel(aX, np.empty([0,0], np.float64), iThreadID)
    else:
        aResult = oUserModel(aX, aY, iThreadID)
    if aResult.shape != (iNumSamples, iNumOuts):
        if TasmanianConfig.enableVerboseErrors:
            print("ERROR: incorrect model output dimensions, should be (iNumSamples, iNumOutputs)")
        return
    aY[0:iNumSamples, 0:iNumOuts] = aResult[0:iNumSamples, 0:iNumOuts]
    pErrInfo[0] = 0

def constructAnisotropicSurrogate(callableModel, iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid,
                                  sDepthType, liAnisotropicWeightsOrOutput,
                                  liLevelLimits = [], bUseInitialGuess = False,
                                  sCheckpointFilename = ""):
    '''
    Construct a surrogate model to the callableModel using
    anisotropic refinement until the iMaxPoints is reached.

    See the documentation for TasGrid::constructSurrogate()
    This is wrapper around the anisotropic refinement variant.

    callableModel is a function (or lambda) that returns a two dimensional
        numpy.ndarray of outputs for the model,
        If bUseInitialGuess is true, the model has to take two inputs,
        otherwise the model uses one.
        The first input is a two dimensional numpy.ndarray of inputs
        similar to TasmanianSparseGrid.evaluateBatch().
        The second input corresponds to the initial guess, the size will
        either match the expected output or will be empty, if no
        guess can be computed.

    iMaxPoints: is a positive integer indicating the maximum number of points
        that the grid will have.
    iMaxParallel: is a positive integer indicating the number of simultaneous
        calls to the user model, i.e., the number of threads.
    iMaxSamplesPerCall: maximum number of samples that will be given to
        a single call to the user model.
    grid: must be an instance of Tasmanian.SparseGrid() with either
        global, sequence or Fourier grid.
    sDepthType: the type used for refinement, see TasGrid::constructSurrogate()

    liAnisotropicWeightsOrOutput: is either an output to use to determine
        the model anisotropy or a list/typle/ndarray of user selected
        anisotropic weights.
    liLevelLimits: same as in all other refinement calls, the refinement will
        never add points below the given level in the diven direction
        even if the budget has not been reached yet.

    sCheckpointFilename: filename to use to checkpoint the algorithm so that
        construction can proceed from a saved point in case of a crash.
    '''
    iNumDims = grid.getNumDimensions()
    pLevelLimits = None
    if (len(liLevelLimits) > 0):
        if (len(liLevelLimits) != iNumDims):
            raise TasmanianInputError("liLevelLimits", "ERROR: invalid number of level limits, must be equal to the grid dimension")
        pLevelLimits = (c_int*iNumDims)()
        for iI in range(iNumDims):
            pLevelLimits[iI] = liLevelLimits[iI]

    if (sys.version_info.major == 3):
        sDepthTypeCtypes = bytes(sDepthType, encoding='utf8')
        if (sCheckpointFilename):
            sCheckpointFilename = bytes(sCheckpointFilename, encoding='utf8')
    else:
        sDepthTypeCtypes = sDepthType

    pCPFname = None
    if (sCheckpointFilename):
        pCPFname = c_char_p(sCheckpointFilename)

    pErrorCode = (c_int * 1)()

    if (((sys.version_info.major == 3) and isinstance(liAnisotropicWeightsOrOutput, int))
            or ((sys.version_info.major == 2) and isinstance(liAnisotropicWeightsOrOutput, (int, long)))):
        # will call the algorithm to dynamically estimate the weights
        iOutput = liAnisotropicWeightsOrOutput

        if (bUseInitialGuess):
            pLibCTSG.tsgConstructSurrogateWiIGAniso(
                type_icsmodel(lambda nx, nd, x, f, ny, y, tid, err : tsgIcsModelWrapper(callableModel, nx, nd, x, f, ny, y, tid, err)),
                iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid.pGrid,
                c_char_p(sDepthTypeCtypes), iOutput, pLevelLimits, pCPFname, pErrorCode)
        else:
            pLibCTSG.tsgConstructSurrogateNoIGAniso(
                type_scsmodel(lambda nx, nd, x, ny, y, tid, err : tsgScsModelWrapper(callableModel, nx, nd, x, ny, y, tid, err)),
                iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid.pGrid,
                c_char_p(sDepthTypeCtypes), iOutput, pLevelLimits, pCPFname, pErrorCode)
    else:
        # weights are set by the user
        pAnisoWeights = None
        if (len(liAnisotropicWeightsOrOutput) > 0):
            if (sDepthType in TasmanianSG.lsTsgCurvedTypes):
                iNumWeights = 2*grid.getNumDimensions()
            else:
                iNumWeights = grid.getNumDimensions()
            if (len(liAnisotropicWeightsOrOutput) != iNumWeights):
                raise TasmanianInputError("liAnisotropicWeightsOrOutput", "ERROR: wrong number of liAnisotropicWeightsOrOutput, sType '{0:s}' needs {1:1d} weights but len(liAnisotropicWeightsOrOutput) == {2:1d}".format(sDepthType, iNumWeights, len(liAnisotropicWeightsOrOutput)))
            else:
                aAWeights = np.array([liAnisotropicWeightsOrOutput[i] for i in range(iNumWeights)], np.int32)
                pAnisoWeights = np.ctypeslib.as_ctypes(aAWeights)

        if (bUseInitialGuess):
            pLibCTSG.tsgConstructSurrogateWiIGAnisoFixed(
                type_icsmodel(lambda nx, nd, x, f, ny, y, tid, err : tsgIcsModelWrapper(callableModel, nx, nd, x, f, ny, y, tid, err)),
                iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid.pGrid,
                c_char_p(sDepthTypeCtypes), pAnisoWeights, pLevelLimits, pCPFname, pErrorCode)
        else:
            pLibCTSG.tsgConstructSurrogateNoIGAnisoFixed(
                type_scsmodel(lambda nx, nd, x, ny, y, tid, err : tsgScsModelWrapper(callableModel, nx, nd, x, ny, y, tid, err)),
                iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid.pGrid,
                c_char_p(sDepthTypeCtypes), pAnisoWeights, pLevelLimits, pCPFname, pErrorCode)

    if pErrorCode[0] != 0:
        raise TasmanianInputError("constructSurplusSurrogate", "An error occurred during the call to Tasmanian.")


def constructSurplusSurrogate(callableModel, iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid,
                              fTolerance, sRefinementType, iOutput = -1,
                              liLevelLimits = [], bUseInitialGuess = False,
                              sCheckpointFilename = ""):
    '''
    Construct a surrogate model to the callableModel using surplus refinement
    until either the iMaxPoints or the tolerance are reached.

    See Tasmanian.constructAnisotropicSurrogate() for all matchin inputs,
    except the grid has to be local polynomial and the refinement proceeds
    until the budget is exhausted or the fTolerance is reached.
    The sRefinementType is the same as in the call to local surplus refinement,
    same with the iOutput.
    '''
    iNumDims = grid.getNumDimensions()
    pLevelLimits = None
    if (len(liLevelLimits) > 0):
        if (len(liLevelLimits) != iNumDims):
            raise TasmanianInputError("liLevelLimits", "ERROR: invalid number of level limits, must be equal to the grid dimension")
        pLevelLimits = (c_int*iNumDims)()
        for iI in range(iNumDims):
            pLevelLimits[iI] = liLevelLimits[iI]

    pCPFname = bytes(sCheckpointFilename, encoding='utf8') if sCheckpointFilename else None

    pErrorCode = (c_int * 1)()

    if (bUseInitialGuess):
        pLibCTSG.tsgConstructSurrogateWiIGSurplus(
            type_icsmodel(lambda nx, nd, x, f, ny, y, tid, err : tsgIcsModelWrapper(callableModel, nx, nd, x, f, ny, y, tid, err)),
            iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid.pGrid,
            fTolerance, bytes(sRefinementType, encoding='utf8'), iOutput, pLevelLimits, pCPFname, pErrorCode)
    else:
        pLibCTSG.tsgConstructSurrogateNoIGSurplus(
            type_scsmodel(lambda nx, nd, x, ny, y, tid, err : tsgScsModelWrapper(callableModel, nx, nd, x, ny, y, tid, err)),
            iMaxPoints, iMaxParallel, iMaxSamplesPerCall, grid.pGrid,
            fTolerance, bytes(sRefinementType, encoding='utf8'), iOutput, pLevelLimits, pCPFname, pErrorCode)

    if pErrorCode[0] != 0:
        raise TasmanianInputError("constructSurplusSurrogate", "An error occurred during the call to Tasmanian.")


def loadUnstructuredDataL2(points, model_data, tolerance, grid):
    '''
    Wrapper around TasGrid::loadUnstructuredDataL2(), see the C++ reference.

    points is 2D ndarray with shape(num_data, grid.getNumDimensions())
    model_data is 2D ndarray with shape(num_data, grid.getNumOutputs())
    grid is an instance of TasmanianSparseGrid
    '''
    if len(points.shape) != 2 or points.shape[1] != grid.getNumDimensions():
        raise TasmanianInputError("points", "ERROR: points must be a 2D numpy.ndarray with points.shape[1] == grid.getNumDimensions()")
    if len(model_data.shape) != 2 or model_data.shape[1] != grid.getNumOutputs():
        raise TasmanianInputError("model_data", "ERROR: model_data must be a 2D numpy.ndarray with model_data.shape[1] == grid.getNumOutputs()")
    if points.shape[0] != model_data.shape[0]:
        raise TasmanianInputError("model_data", "ERROR: mismatch between shape[0] of points and model_data")
    if not hasattr(grid, "TasmanianSparseGridObject"):
        raise TasmanianInputError("grid", "ERROR: grid must be an instance of TasmanianSparseGrid")

    num_data = points.shape[0]
    pLibCTSG.tsgLoadUnstructuredDataL2(np.ctypeslib.as_ctypes(points.reshape((points.size,))), num_data,
                                       np.ctypeslib.as_ctypes(model_data.reshape((model_data.size,))), tolerance, grid.pGrid)

def createExoticQuadratureFromGrid(level, shift, ref_grid, description, is_symmetric = False):
    '''
    Calls TasGrid::getExoticQuadrature() from a one dimensional interpolant/surrogate of the weight function, and output a
    python CustomTabulated object.
    See the C++ reference for more information.

    level:        positive integer representing the level of the exotic quadrature grid.
    shift:        double where [weight_function(x) + shift] is nonegative for every x in the domain of the weight function.
    ref_grid:     Python TasmanianSparseGrid object that represents the one dimensional surrogate/interpolant of the weight function.
    description:  string describing the Exotic quadrature instance.
    is_symmetric: (optional) boolean that should be set to True if the weight function is symmetric.

    output:       a Python CustomTabulated object.
    '''
    if not hasattr(ref_grid, "TasmanianSparseGridObject"):
        raise TasmanianInputError("ref_grid", "ERROR: ref_grid must be an instance of TasmanianSparseGrid")
    ct = TasmanianSG.CustomTabulated()
    pLibCTSG.tsgCreateExoticQuadratureFromGrid(c_void_p(ct.pCustomTabulated), c_int(level), c_double(shift), c_void_p(ref_grid.pGrid),
                                               bytes(description, encoding='utf8'), c_int(is_symmetric))
    return ct

def createExoticQuadratureFromFunction(level, shift, weight_fn, nref, description, is_symmetric = False):
    '''
    Calls TasGrid::getExoticQuadrature() from a function lambda representing the weight function, and output a Python
    CustomTabulated object.
    See the C++ reference for more information.

    level:        positive integer representing the level of the exotic quadrature grid.
    shift:        double where [weight_function(x) + shift] is nonegative for every x in the domain of the weight function.
    weight_fn:    Python lambda function representing the weight function.
    nref:         positive integer representing the number of points used to generate the weight function surrogate/interpolant.
    description:  string describing the Exotic quadrature instance.
    is_symmetric: (optional) boolean that should be set to True if the weight function is symmetric.

    output:       a Python CustomTabulated object.
    '''
    ct = TasmanianSG.CustomTabulated()
    pLibCTSG.tsgCreateExoticQuadratureFromFunction(c_void_p(ct.pCustomTabulated), c_int(level), c_double(shift), type_1Dfunc(weight_fn),
                                                   c_int(nref), bytes(description, encoding='utf8'), c_int(is_symmetric))
    return ct

def createNestedExoticQuadratureFromFunction(level, rule, shift, weight_fn, nref, description):
    '''
    Calls TasGrid::getExoticQuadrature() from a function lambda representing the weight function, and output a Python
    CustomTabulated object.
    See the C++ reference for more information.

    level:        positive integer representing the level of the exotic quadrature grid.
    rule:         a nested rule to use as the points for the exotic quadrature.
    shift:        double where [weight_function(x) + shift] is nonegative for every x in the domain of the weight function.
    weight_fn:    Python lambda function representing the weight function.
    nref:         positive integer representing the number of points used to generate the weight function surrogate/interpolant.
    description:  string describing the Exotic quadrature instance.

    output:       a Python CustomTabulated object.
    '''
    ct = TasmanianSG.CustomTabulated()
    pLibCTSG.tsgCreateNestedExoticQuadratureFromFunction(
            c_void_p(ct.pCustomTabulated), c_int(level), bytes(rule, encoding='utf8'),  c_double(shift), type_1Dfunc(weight_fn),
            c_int(nref), bytes(description, encoding='utf8'))
    return ct
