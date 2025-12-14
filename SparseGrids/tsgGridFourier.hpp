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

#ifndef __TASMANIAN_SPARSE_GRID_FOURIER_HPP
#define __TASMANIAN_SPARSE_GRID_FOURIER_HPP

#include "tsgGridCore.hpp"

namespace TasGrid{

#ifndef __TASMANIAN_DOXYGEN_SKIP
class GridFourier : public BaseCanonicalGrid {
public:
    GridFourier(AccelerationContext const *acc) : BaseCanonicalGrid(acc){};
    friend struct GridReaderVersion5<GridFourier>;
    GridFourier(AccelerationContext const *acc, const GridFourier *fourier, int ibegin, int iend);
    GridFourier(AccelerationContext const *acc, int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits) : BaseCanonicalGrid(acc) {
        makeGrid(cnum_dimensions, cnum_outputs, depth, type, anisotropic_weights, level_limits);
    }
    ~GridFourier() = default;

    GridFourier(GridFourier &&) = default;
    GridFourier &operator = (GridFourier &&) = default;

    bool isFourier() const override{ return true; }

    void write(std::ostream &os, bool iomode) const override{ if (iomode == mode_ascii) write<mode_ascii>(os); else write<mode_binary>(os); }

    template<bool iomode> void write(std::ostream &os) const;

    void makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);
    void updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);

    void setTensors(MultiIndexSet &&tset, int cnum_outputs);

    TypeOneDRule getRule() const override{ return rule_fourier; }

    void loadNeededValues(const double *vals) override;

    void getLoadedPoints(double *x) const override;
    void getNeededPoints(double *x) const override;
    void getPoints(double *x) const override; // returns the loaded points unless no points are loaded, then returns the needed points

    void getInterpolationWeights(const double x[], double weights[]) const override;
    void getQuadratureWeights(double weights[]) const override;
    void getDifferentiationWeights(const double x[], double weights[]) const override;

    void evaluate(const double x[], double y[]) const override;
    void integrate(double q[], double *conformal_correction) const override;
    void differentiate(const double x[], double jacobian[]) const override;

    void evaluateBatch(const double x[], int num_x, double y[]) const override;

    void evaluateBatchGPU(const double gpu_x[], int cpu_num_x, double gpu_y[]) const override;
    void evaluateBatchGPU(const float gpu_x[], int cpu_num_x, float gpu_y[]) const override;
    template<typename T> void evaluateBatchGPUtempl(const T gpu_x[], int cpu_num_x, T gpu_y[]) const;
    void evaluateHierarchicalFunctionsGPU(const double gpu_x[], int num_x, double gpu_y[]) const override;
    void evaluateHierarchicalFunctionsGPU(const float gpu_x[], int num_x, float gpu_y[]) const override;
    template<typename T>
    void evaluateHierarchicalFunctionsInternalGPU(const T gpu_x[], int num_x, GpuVector<T> &wreal, GpuVector<T> &wimag) const;

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const override;
    void evaluateHierarchicalFunctionsInternal(const double x[], int num_x, Data2D<double> &wreal, Data2D<double> &wimag) const;
    void setHierarchicalCoefficients(const double c[]) override;

    void integrateHierarchicalFunctions(double integrals[]) const override;

    void updateAccelerationData(AccelerationContext::ChangeType change) const override;

    void estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const;
    void setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits);
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

    const double* getFourierCoefs() const;

protected:
    void calculateFourierCoefficients();

    MultiIndexSet selectTensors(size_t dims, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights,
                                std::vector<int> const &level_limits) const;
    void proposeUpdatedTensors();
    void acceptUpdatedTensors();

    std::vector<std::vector<int>> generateIndexingMap() const;

    void loadConstructedTensors();
    std::vector<int> getMultiIndex(const double x[]);

    template<typename T, bool interwoven>
    void computeBasis(const MultiIndexSet &work, const T x[], T wreal[], T wimag[]) const{
        int num_points = work.getNumIndexes();

        std::vector<std::vector<std::complex<T>>> cache(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            cache[j].resize(max_power[j] +1);
            cache[j][0] = std::complex<T>(1.0, 0.0);

            T theta = -2.0 * Maths::pi * x[j];
            std::complex<T> step(std::cos(theta), std::sin(theta));
            std::complex<T> pw(1.0, 0.0);
            for(int i=1; i<max_power[j]; i += 2){
                pw *= step;
                cache[j][i] = pw;
                cache[j][i+1] = std::conj<T>(pw);
            }
        }

        for(int i=0; i<num_points; i++){
            const int *p = work.getIndex(i);

            std::complex<T> v(1.0, 0.0);
            for(int j=0; j<num_dimensions; j++){
                v *= cache[j][p[j]];
            }

            if (interwoven){
                wreal[2*i] = v.real();
                wreal[2*i+1] = v.imag();
            }else{
                wreal[i] = v.real();
                wimag[i] = v.imag();
            }
        }
    }

    void clearGpuNodes() const;
    void clearGpuCoefficients() const;

private:
    OneDimensionalWrapper wrapper;

    MultiIndexSet tensors;
    MultiIndexSet active_tensors;
    std::vector<int> active_w;

    MultiIndexSet updated_tensors;
    MultiIndexSet updated_active_tensors;
    std::vector<int> updated_active_w;

    std::vector<int> max_levels;

    Data2D<double> fourier_coefs;

    std::vector<int> max_power;

    template<typename T> void loadGpuNodes() const;
    template<typename T> void loadGpuCoefficients() const;
    inline std::optional<CudaFourierData<double>>& getGpuCacheOverload(double) const{ return gpu_cache; }
    inline std::optional<CudaFourierData<float>>& getGpuCacheOverload(float) const{ return gpu_cachef; }
    template<typename T> inline std::optional<CudaFourierData<T>>& getGpuCache() const{
        return getGpuCacheOverload(static_cast<T>(0.0));
    }
    mutable std::optional<CudaFourierData<double>> gpu_cache;
    mutable std::optional<CudaFourierData<float>> gpu_cachef;

    std::unique_ptr<DynamicConstructorDataGlobal> dynamic_values;
};

// Old version reader
template<> struct GridReaderVersion5<GridFourier>{
    template<typename iomode> static void read(std::istream &is, GridFourier *grid) {
        grid->num_dimensions = IO::readNumber<iomode, int>(is);
        grid->num_outputs = IO::readNumber<iomode, int>(is);

        grid->tensors = MultiIndexSet(is, iomode());
        grid->active_tensors = MultiIndexSet(is, iomode());
        grid->active_w = IO::readVector<iomode, int>(is, grid->active_tensors.getNumIndexes());

        if (IO::readFlag<iomode>(is)) grid->points = MultiIndexSet(is, iomode());
        if (IO::readFlag<iomode>(is)) grid->needed = MultiIndexSet(is, iomode());

        grid->max_levels = IO::readVector<iomode, int>(is, grid->num_dimensions);

        if (grid->num_outputs > 0){
            grid->values = StorageSet(is, iomode());
            if (IO::readFlag<iomode>(is))
                grid->fourier_coefs = IO::readData2D<iomode, double>(is, grid->num_outputs, 2 * grid->points.getNumIndexes());
        }

        int oned_max_level;
        if (IO::readFlag<iomode>(is)){
            grid->updated_tensors = MultiIndexSet(is, iomode());
            oned_max_level = grid->updated_tensors.getMaxIndex();

            grid->updated_active_tensors = MultiIndexSet(is, iomode());

            grid->updated_active_w = IO::readVector<iomode, int>(is, grid->updated_active_tensors.getNumIndexes());
        }else{
            oned_max_level = *std::max_element(grid->max_levels.begin(), grid->max_levels.end());
        }

        grid->wrapper = OneDimensionalWrapper(oned_max_level, rule_fourier, 0.0, 0.0);

        grid->max_power = MultiIndexManipulations::getMaxIndexes(((grid->points.empty()) ? grid->needed : grid->points));
    }
};
#endif // __TASMANIAN_DOXYGEN_SKIP

}

#endif
