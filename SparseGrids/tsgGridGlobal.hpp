/*
 * Copyright (c) 2017, Miroslav Stoyanov
 *
 * This file is part of
 * Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
 *    and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND IMPLIED.
 * THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT,
 * COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE.
 * THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF,
 * IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
 */

#ifndef __TASMANIAN_SPARSE_GRID_GLOBAL_HPP
#define __TASMANIAN_SPARSE_GRID_GLOBAL_HPP

#include "tsgGridSequence.hpp"
#include "tsgCacheLagrange.hpp"

namespace TasGrid{

#ifndef __TASMANIAN_DOXYGEN_SKIP
class GridGlobal : public BaseCanonicalGrid{
public:
    GridGlobal(ThreadEngine *num_threads) : BaseCanonicalGrid(num_threads), alpha(0.0), beta(0.0){}
    ~GridGlobal(){}

    bool isGlobal() const{ return true; }

    void write(std::ostream &os, bool iomode) const{ if (iomode == mode_ascii) write<mode_ascii>(os); else write<mode_binary>(os); }
    void read(std::istream &is, bool iomode){ if (iomode == mode_ascii) read<mode_ascii>(is); else read<mode_binary>(is); }

    template<bool iomode> void write(std::ostream &os) const;
    template<bool iomode> void read(std::istream &is);

    void makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, TypeOneDRule crule, const std::vector<int> &anisotropic_weights, double calpha, double cbeta, const char* custom_filename, const std::vector<int> &level_limits);
    void copyGrid(const GridGlobal *global, int ibegin, int iend);

    void setTensors(MultiIndexSet &&tset, int cnum_outputs, TypeOneDRule crule, double calpha, double cbeta);

    void updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);

    TypeOneDRule getRule() const{ return rule; }
    const char* getCustomRuleDescription() const{ return (custom.getNumLevels() > 0) ? custom.getDescription() : "";  }

    double getAlpha() const{ return alpha; }
    double getBeta() const{ return beta; }

    void getLoadedPoints(double *x) const;
    void getNeededPoints(double *x) const;
    void getPoints(double *x) const; // returns the loaded points unless no points are loaded, then returns the needed points

    void getQuadratureWeights(double weights[]) const;
    void getInterpolationWeights(const double x[], double weights[]) const;

    void loadNeededPoints(const double *vals);
    const double* getLoadedValues() const;

    void evaluate(const double x[], double y[]) const;
    void integrate(double q[], double *conformal_correction) const;

    void evaluateBatch(const double x[], int num_x, double y[]) const;

    #ifdef Tasmanian_ENABLE_BLAS
    void evaluateBlas(const double x[], int num_x, double y[]) const;
    #endif

    #ifdef Tasmanian_ENABLE_CUDA
    void loadNeededPointsCuda(CudaEngine *engine, const double *vals);
    void evaluateCudaMixed(CudaEngine *engine, const double x[], int num_x, double y[]) const;
    void evaluateCuda(CudaEngine *engine, const double x[], int num_x, double y[]) const;
    void evaluateBatchGPU(CudaEngine*, const double[], int, double[]) const;
    void evaluateBatchGPU(CudaEngine*, const float[], int, float[]) const;
    template<typename T> void evaluateBatchGPUtempl(CudaEngine*, T const[], int, T *) const;
    void evaluateHierarchicalFunctionsGPU(const double[], int, double *) const;
    void evaluateHierarchicalFunctionsGPU(const float[], int, float *) const;
    template<typename T> void evaluateHierarchicalFunctionsGPUtempl(T const[], int, T *) const;
    #endif

    void estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const;

    void setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits);
    void setSurplusRefinement(double tolerance, int output, const std::vector<int> &level_limits);
    void clearRefinement();
    void mergeRefinement();

    void beginConstruction();
    void writeConstructionData(std::ostream &os, bool) const;
    void readConstructionData(std::istream &is, bool);
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, const std::vector<int> &weights, const std::vector<int> &level_limits);
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, int output, const std::vector<int> &level_limits);
    std::vector<double> getCandidateConstructionPoints(std::function<double(const int *)> getTensorWeight, const std::vector<int> &level_limits);
    void loadConstructedPoint(const double x[], const std::vector<double> &y);
    void loadConstructedPoint(const double x[], int numx, const double y[]);
    void finishConstruction();

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const;
    void setHierarchicalCoefficients(const double c[], TypeAcceleration acc);
    void integrateHierarchicalFunctions(double integrals[]) const;

    void clearAccelerationData();

    std::vector<int> getPolynomialSpace(bool interpolation) const;

protected:
    void reset(bool includeCustom);

    std::vector<double> computeSurpluses(int output, bool normalize) const; // only for sequence rules, select the output to compute the surpluses

    static double legendre(int n, double x);

    // assumes that if rule == rule_customtabulated, then custom is already loaded
    MultiIndexSet selectTensors(size_t dims, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights,
                                TypeOneDRule rule, std::vector<int> const &level_limits) const;

    void recomputeTensorRefs(const MultiIndexSet &work);
    void proposeUpdatedTensors();
    void acceptUpdatedTensors();
    MultiIndexSet getPolynomialSpaceSet(bool interpolation) const;

    void mapIndexesToNodes(const std::vector<int> &indexes, double *x) const;
    void loadConstructedTensors();
    std::vector<int> getMultiIndex(const double x[]);

    #ifdef Tasmanian_ENABLE_CUDA
    std::unique_ptr<CudaGlobalData<double>>& getCudaCache(double) const{ return cuda_cache; }
    std::unique_ptr<CudaGlobalData<float>>& getCudaCache(float) const{ return cuda_cachef; }
    template<typename T> void loadCudaValues() const;
    void clearCudaValues() const;
    template<typename T> void loadCudaNodes() const;
    void clearCudaNodes() const;
    #endif

private:
    TypeOneDRule rule;
    double alpha, beta;

    OneDimensionalWrapper wrapper;

    MultiIndexSet tensors;
    MultiIndexSet active_tensors;
    std::vector<int> active_w;

    std::vector<std::vector<int>> tensor_refs;

    std::vector<int> max_levels; // for evaluation purposes, counts the maximum level in each direction (only counts tensors)

    MultiIndexSet updated_tensors;
    MultiIndexSet updated_active_tensors;
    std::vector<int> updated_active_w;

    CustomTabulated custom;

    std::unique_ptr<DynamicConstructorDataGlobal> dynamic_values;

    #ifdef Tasmanian_ENABLE_CUDA
    mutable std::unique_ptr<CudaGlobalData<double>> cuda_cache;
    mutable std::unique_ptr<CudaGlobalData<float>> cuda_cachef;
    #endif
};
#endif // __TASMANIAN_DOXYGEN_SKIP

}

#endif
