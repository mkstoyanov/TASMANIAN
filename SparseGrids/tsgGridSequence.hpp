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

#ifndef __TASMANIAN_SPARSE_GRID_GLOBAL_NESTED_HPP
#define __TASMANIAN_SPARSE_GRID_GLOBAL_NESTED_HPP

#include "tsgGridCore.hpp"

namespace TasGrid{

#ifndef __TASMANIAN_DOXYGEN_SKIP
class GridSequence : public BaseCanonicalGrid{
public:
    GridSequence(ThreadEngine * num_threads) : BaseCanonicalGrid(num_threads) {}
    ~GridSequence(){}

    bool isSequence() const{ return true; }

    void write(std::ostream &os, bool iomode) const{ if (iomode == mode_ascii) write<mode_ascii>(os); else write<mode_binary>(os); }
    void read(std::istream &is, bool iomode){ if (iomode == mode_ascii) read<mode_ascii>(is); else read<mode_binary>(is); }

    template<bool iomode> void write(std::ostream &os) const;
    template<bool iomode> void read(std::istream &is);

    void makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, TypeOneDRule crule, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);
    void copyGrid(const GridSequence *seq, int ibegin, int iend);
    void setPoints(MultiIndexSet &pset, int cnum_outputs, TypeOneDRule crule);

    void updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);
    void updateGrid(MultiIndexSet &update);

    TypeOneDRule getRule() const{ return rule; }

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
    void evaluateBatchGPU(CudaEngine* engine, const double gpu_x[], int cpu_num_x, double gpy_y[]) const;
    void evaluateHierarchicalFunctionsGPU(const double x[], int num_x, double y[]) const;
    void evaluateCudaMixed(CudaEngine*, const float*, int, float[]) const;
    void evaluateBatchGPU(CudaEngine* engine, const float gpu_x[], int cpu_num_x, float gpy_y[]) const;
    template<typename T> void evaluateBatchGPUtempl(CudaEngine* engine, const T gpu_x[], int cpu_num_x, T gpy_y[]) const;
    void evaluateHierarchicalFunctionsGPU(const float gpu_x[], int num_x, float gpu_y[]) const;
    #endif

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const;

    void estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const;
    void setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits);
    void setSurplusRefinement(double tolerance, int output, const std::vector<int> &level_limits);
    void clearRefinement();
    void mergeRefinement();

    void beginConstruction();
    void writeConstructionData(std::ostream &ofs, bool) const;
    void readConstructionData(std::istream &ifs, bool);
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, const std::vector<int> &weights, const std::vector<int> &level_limits);
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, int output, const std::vector<int> &level_limits);
    std::vector<double> getCandidateConstructionPoints(std::function<double(const int *)> getTensorWeight, const std::vector<int> &level_limits);
    void loadConstructedPoint(const double x[], const std::vector<double> &y);
    void loadConstructedPoint(const double x[], int numx, const double y[]);
    void finishConstruction();

    void setHierarchicalCoefficients(const double c[], TypeAcceleration acc);
    void integrateHierarchicalFunctions(double integrals[]) const;

    std::vector<int> getPolynomialSpace(bool interpolation) const;

    const double* getSurpluses() const;

    void clearAccelerationData();

protected:
    void reset();

    void evalHierarchicalFunctions(const double x[], double fvalues[]) const;

    //! \brief Cache the nodes and polynomial coefficients, cache is determined by the largest index in \b points and \b needed, or \b num_external (pass zero if not using dy-construction).
    void prepareSequence(int num_external);
    std::vector<double> cacheBasisIntegrals() const;

    template<typename T>
    std::vector<std::vector<T>> cacheBasisValues(const T x[]) const{
        std::vector<std::vector<T>> cache(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            cache[j].resize(max_levels[j] + 1);
            T b = 1.0;
            T this_x = x[j];
            cache[j][0] = b;
            for(int i=0; i<max_levels[j]; i++){
                b *= (this_x - nodes[i]);
                cache[j][i+1] = b;
            }
            for(int i=1; i<=max_levels[j]; i++){
                cache[j][i] /= coeff[i];
            }
        }
        return cache;
    }

    std::vector<int> getMultiIndex(const double x[]);
    void expandGrid(const std::vector<int> &point, const std::vector<double> &values, const std::vector<double> &surplus);
    void loadConstructedPoints();
    void recomputeSurpluses();
    void applyTransformationTransposed(double weights[]) const;

    double evalBasis(const int f[], const int p[]) const; // evaluate function corresponding to f at p

    #ifdef Tasmanian_ENABLE_CUDA
    // specialize below for the float case
    std::unique_ptr<CudaSequenceData<double>>& getCudaCache(double) const{ return cuda_cache; }
    std::unique_ptr<CudaSequenceData<float>>& getCudaCache(float) const{ return cuda_cachef; }
    template<typename T>
    void loadCudaNodes() const{
        auto& ccache = getCudaCache(static_cast<T>(0.0));
        if (!ccache) ccache = std::make_unique<CudaSequenceData<T>>();
        if (!ccache->num_nodes.empty()) return;

        ccache->nodes.load(nodes);
        ccache->coeff.load(coeff);

        std::vector<int> num_nodes(num_dimensions);
        std::transform(max_levels.begin(), max_levels.end(), num_nodes.begin(), [](int i)->int{ return i+1; });
        ccache->num_nodes.load(num_nodes);

        const MultiIndexSet *work = (points.empty()) ? &needed : &points;
        int num_points = work->getNumIndexes();
        Data2D<int> transpoints(work->getNumIndexes(), num_dimensions);
        for(int i=0; i<num_points; i++){
            for(int j=0; j<num_dimensions; j++){
                transpoints.getStrip(j)[i] = work->getIndex(i)[j];
            }
        }
        ccache->points.load(transpoints.getVector());
    }
    void clearCudaNodes();
    template<typename T> void loadCudaSurpluses() const{
        auto& ccache = getCudaCache(static_cast<T>(0.0));
        if (!ccache) ccache = std::make_unique<CudaSequenceData<T>>();
        if (ccache->surpluses.empty()) ccache->surpluses.load(surpluses.getVector());
    }
    void clearCudaSurpluses();
    #endif

private:
    TypeOneDRule rule;

    Data2D<double> surpluses;
    std::vector<double> nodes;
    std::vector<double> coeff;

    std::vector<int> max_levels;

    std::unique_ptr<SimpleConstructData> dynamic_values;

    #ifdef Tasmanian_ENABLE_CUDA
    mutable std::unique_ptr<CudaSequenceData<double>> cuda_cache;
    mutable std::unique_ptr<CudaSequenceData<float>> cuda_cachef;
    #endif
};
#endif // __TASMANIAN_DOXYGEN_SKIP

}

#endif
