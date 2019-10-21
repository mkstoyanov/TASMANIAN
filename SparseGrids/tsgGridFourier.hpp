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
    GridFourier(ThreadEngine * num_threads) : BaseCanonicalGrid(num_threads), max_levels(0){}
    ~GridFourier(){}

    bool isFourier() const{ return true; }

    void write(std::ostream &os, bool iomode) const{ if (iomode == mode_ascii) write<mode_ascii>(os); else write<mode_binary>(os); }
    void read(std::istream &is, bool iomode){ if (iomode == mode_ascii) read<mode_ascii>(is); else read<mode_binary>(is); }

    template<bool iomode> void write(std::ostream &os) const;
    template<bool iomode> void read(std::istream &is);

    void makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);
    void copyGrid(const GridFourier *fourier, int ibegin, int iend);
    void updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);

    void setTensors(MultiIndexSet &&tset, int cnum_outputs);

    TypeOneDRule getRule() const{ return rule_fourier; }

    void loadNeededPoints(const double *vals);

    void getLoadedPoints(double *x) const;
    void getNeededPoints(double *x) const;
    void getPoints(double *x) const; // returns the loaded points unless no points are loaded, then returns the needed points

    void getInterpolationWeights(const double x[], double weights[]) const;

    void getQuadratureWeights(double weights[]) const;

    void evaluate(const double x[], double y[]) const;
    void evaluateBatch(const double x[], int num_x, double y[]) const;

    #ifdef Tasmanian_ENABLE_BLAS
    void evaluateBlas(const double x[], int num_x, double y[]) const;
    #endif

    #ifdef Tasmanian_ENABLE_CUDA
    void loadNeededPointsCuda(CudaEngine *engine, const double *vals);
    void evaluateCudaMixed(CudaEngine *engine, const double x[], int num_x, double y[]) const;
    void evaluateCuda(CudaEngine *engine, const double x[], int num_x, double y[]) const;
    void evaluateBatchGPU(CudaEngine *engine, const double gpu_x[], int cpu_num_x, double gpu_y[]) const;
    void evaluateBatchGPU(CudaEngine *engine, const float gpu_x[], int cpu_num_x, float gpu_y[]) const;
    template<typename T> void evaluateBatchGPUtempl(CudaEngine *engine, const T gpu_x[], int cpu_num_x, T gpu_y[]) const;
    void evaluateHierarchicalFunctionsGPU(const double gpu_x[], int num_x, double gpu_y[]) const;
    void evaluateHierarchicalFunctionsGPU(const float gpu_x[], int num_x, float gpu_y[]) const;
    template<typename T>
    void evaluateHierarchicalFunctionsInternalGPU(const T gpu_x[], int num_x, CudaVector<T> &wreal, CudaVector<T> &wimag) const;
    #endif

    void integrate(double q[], double *conformal_correction) const;

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const;
    void evaluateHierarchicalFunctionsInternal(const double x[], int num_x, Data2D<double> &wreal, Data2D<double> &wimag) const;
    void setHierarchicalCoefficients(const double c[], TypeAcceleration acc);

    void integrateHierarchicalFunctions(double integrals[]) const;

    void clearAccelerationData();

    void estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const;
    void setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits);
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

    const double* getFourierCoefs() const;

protected:
    void reset();
    void calculateFourierCoefficients();

    MultiIndexSet selectTensors(size_t dims, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights,
                                std::vector<int> const &level_limits) const;
    void proposeUpdatedTensors();
    void acceptUpdatedTensors();

    std::vector<std::vector<int>> generateIndexingMap() const;

    void mapIndexesToNodes(const std::vector<int> &indexes, double *x) const;
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

    #ifdef Tasmanian_ENABLE_CUDA
    std::unique_ptr<CudaFourierData<double>>& getCudaCache(double) const{ return cuda_cache; }
    std::unique_ptr<CudaFourierData<float>>& getCudaCache(float) const{ return cuda_cachef; }
    template<typename T> void loadCudaNodes() const;
    void clearCudaNodes();
    template<typename T> void loadCudaCoefficients() const;
    void clearCudaCoefficients();
    #endif

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

    std::unique_ptr<DynamicConstructorDataGlobal> dynamic_values;

    #ifdef Tasmanian_ENABLE_CUDA
    mutable std::unique_ptr<CudaFourierData<double>> cuda_cache;
    mutable std::unique_ptr<CudaFourierData<float>> cuda_cachef;
    #endif
};
#endif // __TASMANIAN_DOXYGEN_SKIP

}

#endif
