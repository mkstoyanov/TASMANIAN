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
    GridWavelet(AccelerationContext const *acc) : BaseCanonicalGrid(acc), rule1D(1, 10), order(1){}
    friend struct GridReaderVersion5<GridWavelet>;
    GridWavelet(AccelerationContext const *acc, const GridWavelet *wav, int ibegin, int iend);
    GridWavelet(AccelerationContext const *acc, int cnum_dimensions, int cnum_outputs, int depth, int corder, const std::vector<int> &level_limits);
    GridWavelet(AccelerationContext const *acc, MultiIndexSet &&pset, int cnum_outputs, int corder, Data2D<double> &&vals);
    ~GridWavelet() = default;

    GridWavelet(GridWavelet &&) = default;
    GridWavelet &operator = (GridWavelet &&) = default;

    bool isWavelet() const override{ return true; }

    void write(std::ostream &os, bool iomode) const override{ if (iomode == mode_ascii) write<mode_ascii>(os); else write<mode_binary>(os); }

    template<bool iomode> void write(std::ostream &os) const;

    TypeOneDRule getRule() const override{ return rule_wavelet; }
    int getOrder() const{ return order; }

    void getLoadedPoints(double *x) const override;
    void getNeededPoints(double *x) const override;
    void getPoints(double *x) const override; // returns the loaded points unless no points are loaded, then returns the needed points

    void getQuadratureWeights(double weights[]) const override;
    void getInterpolationWeights(const double x[], double weights[]) const override;
    void getDifferentiationWeights(const double x[], double weights[]) const override;

    void loadNeededValues(const double *vals) override;

    void evaluate(const double x[], double y[]) const override;
    void integrate(double q[], double *conformal_correction) const override;
    void differentiate(const double x[], double jacobian[]) const override;

    void evaluateBatch(const double x[], int num_x, double y[]) const override;

    void evaluateGpuMixed(const double*, int, double[]) const;
    void evaluateBatchGPU(const double*, int, double[]) const override;
    void evaluateBatchGPU(const float*, int, float[]) const override;
    template<typename T> void evaluateBatchGPUtempl(const T*, int, T[]) const;
    void evaluateHierarchicalFunctionsGPU(const double gpu_x[], int cpu_num_x, double *gpu_y) const override;
    void evaluateHierarchicalFunctionsGPU(const float gpu_x[], int cpu_num_x, float *gpu_y) const override;

    void setSurplusRefinement(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits);
    void clearRefinement() override;
    void mergeRefinement() override;

    void beginConstruction() override;
    void writeConstructionData(std::ostream&, bool) const override;
    void readConstructionData(std::istream&, bool) override;
    std::vector<double> getCandidateConstructionPoints(double tolerance, TypeRefinement criteria, int output, std::vector<int> const &level_limits);
    void loadConstructedPoint(const double[], const std::vector<double> &) override;
    void loadConstructedPoint(const double[], int, const double[]) override;
    void finishConstruction() override;

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const override;
    std::vector<double> getSupport() const override;

    void setHierarchicalCoefficients(const double c[]) override;
    void integrateHierarchicalFunctions(double integrals[]) const override;

    const double* getSurpluses() const;

    void updateAccelerationData(AccelerationContext::ChangeType change) const override;

protected:
    double evalBasis(const int p[], const double x[]) const;
    void buildInterpolationMatrix() const;
    void recomputeCoefficients();
    void solveTransposed(double w[]) const;
    double evalIntegral(const int p[]) const;
    void evalDiffBasis(const int p[], const double x[], double jacobian[]) const;

    std::vector<double> getNormalization() const;
    std::vector<int> getMultiIndex(const double x[]);

    Data2D<int> buildUpdateMap(double tolerance, TypeRefinement criteria, int output) const;
    MultiIndexSet getRefinementCanidates(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits) const;

    bool addParent(const int point[], int direction, Data2D<int> &destination) const;
    void addChild(const int point[], int direction, Data2D<int> &destination) const;
    void addChildLimited(const int point[], int direction, const std::vector<int> &level_limits, Data2D<int> &destination) const;

    void clearGpuCoefficients() const;
    void clearGpuBasis() const;

private:
    RuleWavelet rule1D;

    int order;

    Data2D<double> coefficients; // a.k.a., surpluses

    mutable TasSparse::WaveletBasisMatrix inter_matrix;

    std::optional<CudaWaveletData<double>>& getGpuCacheOverload(double) const{ return gpu_cache; }
    std::optional<CudaWaveletData<float>>& getGpuCacheOverload(float) const{ return gpu_cachef; }
    template<typename T> std::optional<CudaWaveletData<T>>& getGpuCache() const{
        return getGpuCacheOverload(static_cast<T>(0.0));
    }
    template<typename T> void loadGpuCoefficients() const;
    template<typename T> void loadGpuBasis() const;
    mutable std::optional<CudaWaveletData<double>> gpu_cache;
    mutable std::optional<CudaWaveletData<float>> gpu_cachef;

    std::unique_ptr<SimpleConstructData> dynamic_values;
};

// Old version reader
template<> struct GridReaderVersion5<GridWavelet>{
    template<typename iomode> static void read(std::istream &is, GridWavelet *grid) {
        grid->num_dimensions = IO::readNumber<iomode, int>(is);
        grid->num_outputs = IO::readNumber<iomode, int>(is);
        grid->order = IO::readNumber<iomode, int>(is);
        grid->rule1D.updateOrder(grid->order);

        if (IO::readFlag<iomode>(is)) grid->points = MultiIndexSet(is, iomode());
        if (std::is_same<iomode, IO::mode_ascii_type>::value){ // backwards compatible: surpluses and needed, or needed and surpluses
            if (IO::readFlag<iomode>(is))
                grid->coefficients = IO::readData2D<iomode, double>(is, grid->num_outputs, grid->points.getNumIndexes());
            if (IO::readFlag<iomode>(is)) grid->needed = MultiIndexSet(is, iomode());
        }else{
            if (IO::readFlag<iomode>(is)) grid->needed = MultiIndexSet(is, iomode());
            if (IO::readFlag<iomode>(is))
                grid->coefficients = IO::readData2D<iomode, double>(is, grid->num_outputs, grid->points.getNumIndexes());
        }

        if (grid->num_outputs > 0) grid->values = StorageSet(is, iomode());
        grid->buildInterpolationMatrix();
    }
};
#endif // __TASMANIAN_DOXYGEN_SKIP

}

#endif
