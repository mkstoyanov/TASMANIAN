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

#ifndef __TASMANIAN_SPARSE_GRID_WAVELET_HPP
#define __TASMANIAN_SPARSE_GRID_WAVELET_HPP

#include "tsgRuleWavelet.hpp"

namespace TasGrid{

#ifndef __TASMANIAN_DOXYGEN_SKIP
class GridWavelet : public BaseCanonicalGrid{
public:
    GridWavelet(ThreadEngine * num_threads) : BaseCanonicalGrid(num_threads), rule1D(1, 10), order(1){}
    ~GridWavelet(){}

    bool isWavelet() const{ return true; }

    void write(std::ostream &os, bool iomode) const{ if (iomode == mode_ascii) write<mode_ascii>(os); else write<mode_binary>(os); }
    void read(std::istream &is, bool iomode){ if (iomode == mode_ascii) read<mode_ascii>(is); else read<mode_binary>(is); }

    template<bool iomode> void write(std::ostream &os) const;
    template<bool iomode> void read(std::istream &is);

    void makeGrid(int cnum_dimensions, int cnum_outputs, int depth, int corder, const std::vector<int> &level_limits);
    void copyGrid(const GridWavelet *wav, int ibegin, int iend);
    void setNodes(MultiIndexSet &nodes, int cnum_outputs, int corder); // for FDS purposes

    TypeOneDRule getRule() const{ return rule_wavelet; }
    int getOrder() const{ return order; }

    void getLoadedPoints(double *x) const;
    void getNeededPoints(double *x) const;
    void getPoints(double *x) const; // returns the loaded points unless no points are loaded, then returns the needed points

    void getQuadratureWeights(double weights[]) const;
    void getInterpolationWeights(const double x[], double weights[]) const;

    void loadNeededPoints(const double *vals);

    void evaluate(const double x[], double y[]) const;
    void integrate(double q[], double *conformal_correction) const;

    void evaluateBatch(const double x[], int num_x, double y[]) const;

    #ifdef Tasmanian_ENABLE_BLAS
    void evaluateBlas(const double x[], int num_x, double y[]) const;
    #endif

    #ifdef Tasmanian_ENABLE_CUDA
    void loadNeededPointsCuda(CudaEngine *engine, const double *vals);
    void evaluateCudaMixed(CudaEngine*, const double*, int, double[]) const;
    void evaluateCuda(CudaEngine*, const double*, int, double[]) const;
    void evaluateBatchGPU(CudaEngine*, const double*, int, double[]) const;
    void evaluateBatchGPU(CudaEngine*, const float*, int, float[]) const;
    template<typename T> void evaluateBatchGPUtempl(CudaEngine*, const T*, int, T[]) const;
    void evaluateHierarchicalFunctionsGPU(const double gpu_x[], int cpu_num_x, double *gpu_y) const;
    void evaluateHierarchicalFunctionsGPU(const float gpu_x[], int cpu_num_x, float *gpu_y) const;
    #endif

    void setSurplusRefinement(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits);
    void clearRefinement();
    void mergeRefinement();

    void beginConstruction();
    void writeConstructionData(std::ostream&, bool) const;
    void readConstructionData(std::istream&, bool);
    std::vector<double> getCandidateConstructionPoints(double tolerance, TypeRefinement criteria, int output, std::vector<int> const &level_limits);
    void loadConstructedPoint(const double[], const std::vector<double> &);
    void loadConstructedPoint(const double[], int, const double[]);
    void finishConstruction();

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const;

    void setHierarchicalCoefficients(const double c[], TypeAcceleration acc);
    void integrateHierarchicalFunctions(double integrals[]) const;

    const double* getSurpluses() const;

    void clearAccelerationData();

protected:
    void reset();

    double evalBasis(const int p[], const double x[]) const;
    void buildInterpolationMatrix();
    void recomputeCoefficients();
    void solveTransposed(double w[]) const;
    double evalIntegral(const int p[]) const;

    std::vector<double> getNormalization() const;
    std::vector<int> getMultiIndex(const double x[]);

    Data2D<int> buildUpdateMap(double tolerance, TypeRefinement criteria, int output) const;
    MultiIndexSet getRefinementCanidates(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits) const;

    bool addParent(const int point[], int direction, Data2D<int> &destination) const;
    void addChild(const int point[], int direction, Data2D<int> &destination) const;
    void addChildLimited(const int point[], int direction, const std::vector<int> &level_limits, Data2D<int> &destination) const;

    #ifdef Tasmanian_ENABLE_CUDA
    std::unique_ptr<CudaWaveletData<double>>& getCudaCache(double) const{ return cuda_cache; }
    std::unique_ptr<CudaWaveletData<float>>& getCudaCache(float) const{ return cuda_cachef; }
    template<typename T> void loadCudaCoefficients() const;
    void clearCudaCoefficients();
    template<typename T> void loadCudaBasis() const;
    void clearCudaBasis();
    #endif

private:
    RuleWavelet rule1D;

    int order;

    Data2D<double> coefficients; // a.k.a., surpluses

    TasSparse::SparseMatrix inter_matrix;

    std::unique_ptr<SimpleConstructData> dynamic_values;

    #ifdef Tasmanian_ENABLE_CUDA
    mutable std::unique_ptr<CudaWaveletData<double>> cuda_cache;
    mutable std::unique_ptr<CudaWaveletData<float>> cuda_cachef;
    #endif
};
#endif // __TASMANIAN_DOXYGEN_SKIP

}

#endif
