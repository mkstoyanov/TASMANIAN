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
    GridGlobal(AccelerationContext const *acc) : BaseCanonicalGrid(acc), rule(rule_none), alpha(0.0), beta(0.0){}
    friend struct GridReaderVersion5<GridGlobal>;
    GridGlobal(AccelerationContext const *acc, const GridGlobal *global, int ibegin, int iend);
    GridGlobal(AccelerationContext const *acc, int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, TypeOneDRule crule, const std::vector<int> &anisotropic_weights, double calpha, double cbeta, const char* custom_filename, const std::vector<int> &level_limits) : BaseCanonicalGrid(acc){
        makeGrid(cnum_dimensions, cnum_outputs, depth, type, crule, anisotropic_weights, calpha, cbeta, custom_filename, level_limits);
    }
    GridGlobal(AccelerationContext const *acc, int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type,
               CustomTabulated &&crule, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits)
        : BaseCanonicalGrid(acc), custom(std::move(crule))
    {
        setTensors(selectTensors((size_t) cnum_dimensions, depth, type, anisotropic_weights, rule_customtabulated, level_limits),
                   cnum_outputs, rule_customtabulated, 0.0, 0.0);
    }

    GridGlobal(GridGlobal &&) = default;
    GridGlobal &operator = (GridGlobal &&) = default;

    bool isGlobal() const override{ return true; }

    void write(std::ostream &os, bool iomode) const override{ if (iomode == mode_ascii) write<mode_ascii>(os); else write<mode_binary>(os); }

    template<bool iomode> void write(std::ostream &os) const;

    void makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, TypeOneDRule crule, const std::vector<int> &anisotropic_weights, double calpha, double cbeta, const char* custom_filename, const std::vector<int> &level_limits);

    void setTensors(MultiIndexSet &&tset, int cnum_outputs, TypeOneDRule crule, double calpha, double cbeta);

    void updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);

    TypeOneDRule getRule() const override{ return rule; }
    const char* getCustomRuleDescription() const{ return (custom.getNumLevels() > 0) ? custom.getDescription() : "";  }

    double getAlpha() const{ return alpha; }
    double getBeta() const{ return beta; }

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

    void evaluateBatchGPU(const double[], int, double[]) const override;
    void evaluateBatchGPU(const float[], int, float[]) const override;
    template<typename T> void evaluateBatchGPUtempl(T const[], int, T *) const;
    void evaluateHierarchicalFunctionsGPU(const double[], int, double *) const override;
    void evaluateHierarchicalFunctionsGPU(const float[], int, float *) const override;
    template<typename T> void evaluateHierarchicalFunctionsGPUtempl(T const[], int, T *) const;

    void estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const;

    void setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits);
    void setSurplusRefinement(double tolerance, int output, const std::vector<int> &level_limits);
    void clearRefinement() override;
    void mergeRefinement() override;

    void beginConstruction() override;
    void writeConstructionData(std::ostream &os, bool) const override;
    void readConstructionData(std::istream &is, bool) override;
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, const std::vector<int> &weights, const std::vector<int> &level_limits);
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, int output, const std::vector<int> &level_limits);
    std::vector<double> getCandidateConstructionPoints(std::function<double(const int *)> getTensorWeight, const std::vector<int> &level_limits);
    void loadConstructedPoint(const double x[], const std::vector<double> &y) override;
    void loadConstructedPoint(const double x[], int numx, const double y[]) override;
    void finishConstruction() override;

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const override;
    void setHierarchicalCoefficients(const double c[]) override;
    void integrateHierarchicalFunctions(double integrals[]) const override;

    void updateAccelerationData(AccelerationContext::ChangeType change) const override;

    std::vector<int> getPolynomialSpace(bool interpolation) const;

protected:
    std::vector<double> computeSurpluses(int output, bool normalize) const; // only for sequence rules, select the output to compute the surpluses

    static double legendre(int n, double x);

    // assumes that if rule == rule_customtabulated, then custom is already loaded
    MultiIndexSet selectTensors(size_t dims, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights,
                                TypeOneDRule rule, std::vector<int> const &level_limits) const;

    void recomputeTensorRefs(const MultiIndexSet &work);
    void proposeUpdatedTensors();
    void acceptUpdatedTensors();
    MultiIndexSet getPolynomialSpaceSet(bool interpolation) const;

    void loadConstructedTensors();
    std::vector<int> getMultiIndex(const double x[]);

    void clearGpuValues() const;
    void clearGpuNodes() const;

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

    template<typename T> void loadGpuNodes() const;
    template<typename T> void loadGpuValues() const;
    inline std::optional<CudaGlobalData<double>>& getGpuCacheOverload(double) const{ return gpu_cache; }
    inline std::optional<CudaGlobalData<float>>& getGpuCacheOverload(float) const{ return gpu_cachef; }
    template<typename T> inline std::optional<CudaGlobalData<T>>& getGpuCache() const{
        return getGpuCacheOverload(static_cast<T>(0.0));
    }
    mutable std::optional<CudaGlobalData<double>> gpu_cache;
    mutable std::optional<CudaGlobalData<float>> gpu_cachef;

    std::unique_ptr<DynamicConstructorDataGlobal> dynamic_values;
};

// Old version reader
template<> struct GridReaderVersion5<GridGlobal>{
    template<typename iomode> static void read(std::istream &is, GridGlobal *grid) {
        grid->num_dimensions = IO::readNumber<iomode, int>(is);
        grid->num_outputs = IO::readNumber<iomode, int>(is);
        grid->alpha = IO::readNumber<iomode, double>(is);
        grid->beta = IO::readNumber<iomode, double>(is);
        grid->rule = IO::readRule<iomode>(is);
        if (grid->rule == rule_customtabulated) grid->custom = CustomTabulated(is, iomode());
        grid->tensors = MultiIndexSet(is, iomode());
        grid->active_tensors = MultiIndexSet(is, iomode());
        grid->active_w = IO::readVector<iomode, int>(is, grid->active_tensors.getNumIndexes());

        if (IO::readFlag<iomode>(is)) grid->points = MultiIndexSet(is, iomode());
        if (IO::readFlag<iomode>(is)) grid->needed = MultiIndexSet(is, iomode());

        grid->max_levels = IO::readVector<iomode, int>(is, grid->num_dimensions);

        if (grid->num_outputs > 0) grid->values = StorageSet(is, iomode());

        int oned_max_level;
        if (IO::readFlag<iomode>(is)){
            grid->updated_tensors = MultiIndexSet(is, iomode());
            oned_max_level = grid->updated_tensors.getMaxIndex();

            grid->updated_active_tensors = MultiIndexSet(is, iomode());

            grid->updated_active_w = IO::readVector<iomode, int>(is, grid->updated_active_tensors.getNumIndexes());
        }else{
            oned_max_level = *std::max_element(grid->max_levels.begin(), grid->max_levels.end());
        }

        grid->wrapper = OneDimensionalWrapper(grid->custom, oned_max_level, grid->rule, grid->alpha, grid->beta);

        grid->recomputeTensorRefs((grid->points.empty()) ? grid->needed : grid->points);
    }
};

#endif // __TASMANIAN_DOXYGEN_SKIP

}

#endif
