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

#ifndef __TASMANIAN_SPARSE_GRID_WAVELET_CPP
#define __TASMANIAN_SPARSE_GRID_WAVELET_CPP

#include "tsgGridWavelet.hpp"
#include "tsgHiddenExternals.hpp"

namespace TasGrid{

GridWavelet::GridWavelet() : rule1D(1, 10), order(1){}
GridWavelet::~GridWavelet(){}

void GridWavelet::reset(){
    clearAccelerationData();
    points = MultiIndexSet();
    needed = MultiIndexSet();
    values = StorageSet();
    inter_matrix = TasSparse::SparseMatrix();
    coefficients.clear();
    dynamic_values.reset();
}

template<bool iomode> void GridWavelet::write(std::ostream &os) const{
    if (iomode == mode_ascii){ os << std::scientific; os.precision(17); }
    IO::writeNumbers<iomode, IO::pad_line>(os, num_dimensions, num_outputs, order);
    IO::writeFlag<iomode, IO::pad_auto>(!points.empty(), os);
    if (!points.empty()) points.write<iomode>(os);
    if (iomode == mode_ascii){ // backwards compatible: surpluses and needed, or needed and surpluses
        IO::writeFlag<iomode, IO::pad_auto>((coefficients.getNumStrips() != 0), os);
        if (coefficients.getNumStrips() != 0) IO::writeVector<iomode, IO::pad_line>(coefficients.getVector(), os);
        IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
        if (!needed.empty()) needed.write<iomode>(os);
    }else{
        IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
        if (!needed.empty()) needed.write<iomode>(os);
        IO::writeFlag<iomode, IO::pad_auto>((coefficients.getNumStrips() != 0), os);
        if (coefficients.getNumStrips() != 0) IO::writeVector<iomode, IO::pad_line>(coefficients.getVector(), os);
    }

    if (num_outputs > 0) values.write<iomode>(os);
}
template<bool iomode> void GridWavelet::read(std::istream &is){
    reset();
    num_dimensions = IO::readNumber<iomode, int>(is);
    num_outputs = IO::readNumber<iomode, int>(is);
    order = IO::readNumber<iomode, int>(is);
    rule1D.updateOrder(order);

    if (IO::readFlag<iomode>(is)) points.read<iomode>(is);
    if (iomode == mode_ascii){ // backwards compatible: surpluses and needed, or needed and surpluses
        if (IO::readFlag<iomode>(is))
            coefficients = IO::readData2D<iomode, double>(is, num_outputs, points.getNumIndexes());
        if (IO::readFlag<iomode>(is)) needed.read<iomode>(is);
    }else{
        if (IO::readFlag<iomode>(is)) needed.read<iomode>(is);
        if (IO::readFlag<iomode>(is))
            coefficients = IO::readData2D<iomode, double>(is, num_outputs, points.getNumIndexes());
    }

    if (num_outputs > 0) values.read<iomode>(is);
    buildInterpolationMatrix();
}

template void GridWavelet::write<mode_ascii>(std::ostream &) const;
template void GridWavelet::write<mode_binary>(std::ostream &) const;
template void GridWavelet::read<mode_ascii>(std::istream &);
template void GridWavelet::read<mode_binary>(std::istream &);

void GridWavelet::makeGrid(int cnum_dimensions, int cnum_outputs, int depth, int corder, const std::vector<int> &level_limits){
    reset();
    num_dimensions = cnum_dimensions;
    num_outputs = cnum_outputs;
    order = corder;

    rule1D.updateOrder(order);

    MultiIndexSet tensors = MultiIndexManipulations::selectTensors((size_t) num_dimensions, depth, type_level, [](int i) -> int{ return i; }, std::vector<int>(), level_limits);

    if (order == 1){
        needed = MultiIndexManipulations::generateNestedPoints(num_threads, tensors, [](int l)->int{ return (1 << (l + 1)) + 1; });
    }else{
        needed = MultiIndexManipulations::generateNestedPoints(num_threads, tensors, [](int l)->int{ return (1 << (l + 2)) + 1; });
    }

    if (num_outputs == 0){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        values.resize(num_outputs, needed.getNumIndexes());
    }

    buildInterpolationMatrix();
}
void GridWavelet::copyGrid(const GridWavelet *wav, int ibegin, int iend){
    num_dimensions = wav->num_dimensions;
    num_outputs    = iend - ibegin;
    points = wav->points;
    needed = wav->needed;

    rule1D = wav->rule1D;
    order  = wav->order;

    coefficients = (num_outputs == wav->num_outputs) ? wav->coefficients : wav->coefficients.splitData(ibegin, iend);
    values       = (num_outputs == wav->num_outputs) ? wav->values : wav->values.splitValues(ibegin, iend);

    inter_matrix = wav->inter_matrix;
}

void GridWavelet::setNodes(MultiIndexSet &nodes, int cnum_outputs, int corder){
    reset();
    num_dimensions = (int) nodes.getNumDimensions();
    num_outputs = cnum_outputs;
    order = corder;

    rule1D.updateOrder(order);

    if (num_outputs == 0){
        points = std::move(nodes);
    }else{
        needed = std::move(nodes);
        values.resize(num_outputs, needed.getNumIndexes());
    }

    buildInterpolationMatrix();
}

void GridWavelet::getLoadedPoints(double *x) const{
    int num_points = points.getNumIndexes();
    #pragma omp parallel for schedule(static)
    for(int i=0; i<num_points; i++){
        const int *p = points.getIndex(i);
        for(int j=0; j<num_dimensions; j++){
            x[i*num_dimensions + j] = rule1D.getNode(p[j]);
        }
    }
}
void GridWavelet::getNeededPoints(double *x) const{
    int num_points = needed.getNumIndexes();
    #pragma omp parallel for schedule(static)
    for(int i=0; i<num_points; i++){
        const int *p = needed.getIndex(i);
        for(int j=0; j<num_dimensions; j++){
            x[i*num_dimensions + j] = rule1D.getNode(p[j]);
        }
    }
}
void GridWavelet::getPoints(double *x) const{
    if (points.empty()){ getNeededPoints(x); }else{ getLoadedPoints(x); }
}

void GridWavelet::getQuadratureWeights(double *weights) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();
    #pragma omp parallel for
    for(int i=0; i<num_points; i++){
        weights[i] = evalIntegral(work.getIndex(i));
    }
    solveTransposed(weights);
}
void GridWavelet::getInterpolationWeights(const double x[], double *weights) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();
    #pragma omp parallel for
    for(int i=0; i<num_points; i++){
        weights[i] = evalBasis(work.getIndex(i), x);
    }
    solveTransposed(weights);
}
void GridWavelet::loadNeededPoints(const double *vals){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaCoefficients();
    #endif
    if (points.empty()){
        #ifdef Tasmanian_ENABLE_CUDA
        clearCudaBasis();
        #endif
        values.setValues(vals);
        points = std::move(needed);
        needed = MultiIndexSet();
    }else if (needed.empty()){
        values.setValues(vals);
    }else{
        #ifdef Tasmanian_ENABLE_CUDA
        clearCudaBasis();
        #endif
        values.addValues(points, needed, vals);
        points.addMultiIndexSet(needed);
        needed = MultiIndexSet();
        buildInterpolationMatrix();
    }
    recomputeCoefficients();
}
void GridWavelet::mergeRefinement(){
    if (needed.empty()) return; // nothing to do
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaCoefficients();
    clearCudaBasis();
    #endif
    int num_all_points = getNumLoaded() + getNumNeeded();
    size_t size_vals = ((size_t) num_all_points) * ((size_t) num_outputs);
    values.setValues(std::vector<double>(size_vals, 0.0));
    if (points.empty()){
        points = std::move(needed);
    }else{
        points.addMultiIndexSet(needed);
        buildInterpolationMatrix();
    }
    needed = MultiIndexSet();
    coefficients.resize(num_outputs, num_all_points);
    coefficients.fill(0.0);
}
void GridWavelet::evaluate(const double x[], double y[]) const{
    std::fill(y, y + num_outputs, 0.0);

    int num_points = points.getNumIndexes();

    for(int i=0; i<num_points; i++){
        double basis_value = evalBasis(points.getIndex(i), x);
        const double *s = coefficients.getStrip(i);
        for(int k=0; k<num_outputs; k++)
            y[k] += basis_value * s[k];
    }
}
void GridWavelet::evaluateBatch(const double x[], int num_x, double y[]) const{
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_outputs, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++)
        evaluate(xwrap.getStrip(i), ywrap.getStrip(i));
}

#ifdef Tasmanian_ENABLE_BLAS
void GridWavelet::evaluateBlas(const double x[], int num_x, double y[]) const{
    int num_points = points.getNumIndexes();
    Data2D<double> weights(num_points, num_x);
    evaluateHierarchicalFunctions(x, num_x, weights.getStrip(0));
    TasBLAS::denseMultiply(num_outputs, num_x, num_points, 1.0, coefficients.getStrip(0), weights.getStrip(0), 0.0, y);
}
#endif

#ifdef Tasmanian_ENABLE_CUDA
void GridWavelet::loadNeededPointsCuda(CudaEngine *, const double *vals){ loadNeededPoints(vals); }
void GridWavelet::evaluateCudaMixed(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    loadCudaCoefficients<double>();

    Data2D<double> weights(points.getNumIndexes(), num_x);
    evaluateHierarchicalFunctions(x, num_x, weights.getStrip(0));

    engine->denseMultiply(num_outputs, num_x, points.getNumIndexes(), 1.0, cuda_cache->coefficients, weights.getVector(), y);
}
void GridWavelet::evaluateCuda(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    if ((order != 1) || (num_x == 1)){
        // GPU evaluations are available only for order 1
        // cannot use GPU to accelerate the evaluation of a single vector
        evaluateCudaMixed(engine, x, num_x, y);
        return;
    }
    CudaVector<double> gpu_x(num_dimensions, num_x, x), gpu_result(num_x, num_outputs);
    evaluateBatchGPU(engine, gpu_x.data(), num_x, gpu_result.data());
    gpu_result.unload(y);
}
template<typename T> void GridWavelet::evaluateBatchGPUtempl(CudaEngine *engine, const T *gpu_x, int cpu_num_x, T gpu_y[]) const{
    if (order != 1) throw std::runtime_error("ERROR: GPU evaluations are available only for wavelet grids with order 1");
    loadCudaCoefficients<T>();
    int num_points = points.getNumIndexes();

    CudaVector<T> gpu_basis(cpu_num_x, num_points);
    evaluateHierarchicalFunctionsGPU(gpu_x, cpu_num_x, gpu_basis.data());
    engine->denseMultiply(num_outputs, cpu_num_x, num_points, 1.0, getCudaCache(static_cast<T>(0.0))->coefficients, gpu_basis, 0.0, gpu_y);
}
void GridWavelet::evaluateBatchGPU(CudaEngine *engine, const double *gpu_x, int cpu_num_x, double gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
void GridWavelet::evaluateBatchGPU(CudaEngine *engine, const float *gpu_x, int cpu_num_x, float gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
void GridWavelet::evaluateHierarchicalFunctionsGPU(const double gpu_x[], int cpu_num_x, double *gpu_y) const{
    loadCudaBasis<double>();
    TasCUDA::devalpwpoly(order, rule_wavelet, num_dimensions, cpu_num_x, getNumPoints(), gpu_x, cuda_cache->nodes.data(), cuda_cache->support.data(), gpu_y);
}
void GridWavelet::evaluateHierarchicalFunctionsGPU(const float gpu_x[], int cpu_num_x, float *gpu_y) const{
    loadCudaBasis<float>();
    TasCUDA::devalpwpoly(order, rule_wavelet, num_dimensions, cpu_num_x, getNumPoints(), gpu_x, cuda_cachef->nodes.data(), cuda_cachef->support.data(), gpu_y);
}
template<typename T> void GridWavelet::loadCudaBasis() const{
    auto &ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaWaveletData<T>>();
    if (!ccache->nodes.empty()) return;

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    Data2D<double> cpu_scale(num_dimensions, work.getNumIndexes());
    Data2D<double> cpu_shift(num_dimensions, work.getNumIndexes());

    for(int i=0; i<work.getNumIndexes(); i++){
        int const *p = work.getIndex(i);
        double *scale = cpu_scale.getStrip(i);
        double *shift = cpu_shift.getStrip(i);
        for(int j=0; j<num_dimensions; j++)
            rule1D.getShiftScale(p[j], scale[j], shift[j]);
    }
    ccache->nodes.load(cpu_scale.getVector());
    ccache->support.load(cpu_shift.getVector());
}
void GridWavelet::clearCudaBasis(){
    if (cuda_cache) cuda_cache->clearNodes();
    if (cuda_cachef) cuda_cachef->clearNodes();
}
template<typename T> void GridWavelet::loadCudaCoefficients() const{
    auto &ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaWaveletData<T>>();
    if (ccache->coefficients.empty()) ccache->coefficients.load(coefficients.getVector());
}
void GridWavelet::clearCudaCoefficients(){
    if (cuda_cache) cuda_cache->coefficients.clear();
    if (cuda_cachef) cuda_cachef->coefficients.clear();
}
#endif

void GridWavelet::integrate(double q[], double *conformal_correction) const{
    int num_points = points.getNumIndexes();

    if (conformal_correction == 0){
        std::fill_n(q, num_outputs, 0.0);
        for(int i=0; i<num_points; i++){
            double basis_integrals = evalIntegral(points.getIndex(i));
            const double *coeff = coefficients.getStrip(i);
            for(int j=0; j<num_outputs; j++)
                q[j] += basis_integrals * coeff[j];
        }

    }else{
        std::fill(q, q + num_outputs, 0.0);
        std::vector<double> w(num_points);
        getQuadratureWeights(w.data());
        for(int i=0; i<num_points; i++){
            w[i] *= conformal_correction[i];
            const double *vals = values.getValues(i);
            for(int k=0; k<num_outputs; k++){
                q[k] += w[i] * vals[k];
            }
        }
    }
}

double GridWavelet::evalBasis(const int p[], const double x[]) const{
    // Evaluates the wavelet basis given at point p at the coordinates given by x.
    double v = 1.0;
    for(int i = 0; i < num_dimensions; i++){
        v *= rule1D.eval(p[i], x[i]);
        if (v == 0.0){ break; }; // MIRO: reduce the expensive wavelet evaluations
    }
    return v;
}
double GridWavelet::evalIntegral(const int p[]) const{
    // For a given node p, evaluate the integral of the associated wavelet.
    double v = 1.0;
    for(int i = 0; i < num_dimensions; i++){
        v *= rule1D.getWeight(p[i]);
        if (v == 0.0){ break; }; // MIRO: reduce the expensive wavelet evaluations
    }
    return v;
}

void GridWavelet::buildInterpolationMatrix(){
    // updated code, using better parallelism
    // Wavelets don't have a nice rule of support to use monkeys and graphs (or I cannot find the support rule)
    MultiIndexSet &work = (points.empty()) ? needed : points;
    inter_matrix = TasSparse::SparseMatrix();

    int num_points = work.getNumIndexes();

    int num_chunk = 32;
    int num_blocks = num_points / num_chunk + ((num_points % num_chunk == 0) ? 0 : 1);

    std::vector<std::vector<int>> indx(num_blocks);
    std::vector<std::vector<double>> vals(num_blocks);
    std::vector<int> pntr(num_points);

    #pragma omp parallel for
    for(int b=0; b<num_blocks; b++){
        int block_end = (b < num_blocks - 1) ? (b+1) * num_chunk : num_points;
        for(int i=b * num_chunk; i < block_end; i++){
            const int *p = work.getIndex(i);
            std::vector<double> xi(num_dimensions);
            for(int j = 0; j<num_dimensions; j++) // get the node
                xi[j] = rule1D.getNode(p[j]);

            // loop over the basis functions to see if supported
            int numpntr = 0;
            for(int wi=0; wi<num_points; wi++){
                const int *w = work.getIndex(wi);

                double v = 1.0;
                for(int j=0; j<num_dimensions; j++){
                    v *= rule1D.eval(w[j], xi[j]);
                    if (v == 0.0) break; // evaluating the wavelets is expensive, stop if any one of them is zero
                }

                if(v != 0.0){ // testing != is safe since it can only happen in multiplication by 0.0
                    indx[b].push_back(wi);
                    vals[b].push_back(v);
                    numpntr++;
                }
            }
            pntr[i] = numpntr;
        }
    }

    inter_matrix.load(pntr, indx, vals);
}

void GridWavelet::recomputeCoefficients(){
    // Recalculates the coefficients to interpolate the values in points.
    //  Make sure buildInterpolationMatrix has been called since the list was updated.

    int num_points = points.getNumIndexes();
    coefficients.resize(num_outputs, num_points);

    if (inter_matrix.getNumRows() != num_points) buildInterpolationMatrix();

    std::vector<double> b(num_points), x(num_points);

    for(int output = 0; output < num_outputs; output++){
        // Copy relevant portion
        std::fill(x.begin(), x.end(), 0.0);
        std::fill(b.begin(), b.end(), 0.0);

        // Populate RHS
        for(int i = 0; i < num_points; i++){
            b[i] = values.getValues(i)[output];
        }

        // Solve system
        inter_matrix.solve(b.data(), x.data());

        // Populate surplus
        for(int i = 0; i < num_points; i++){
            coefficients.getStrip(i)[output] = x[i];
        }
    }
}

void GridWavelet::solveTransposed(double w[]) const{
    // Solves the system A^T * w = y. Used to calculate interpolation and integration
    // weights. RHS values should be passed in through w. At exit, w will contain the
    // required weights.
    int num_points = inter_matrix.getNumRows();

    std::vector<double> y(num_points);

    std::copy(w, w + num_points, y.data());

    inter_matrix.solve(y.data(), w, true);
}

std::vector<double> GridWavelet::getNormalization() const{
    std::vector<double> norm(num_outputs);
    std::fill(norm.begin(), norm.end(), 0.0);
    for(int i=0; i<points.getNumIndexes(); i++){
        const double *v = values.getValues(i);
        for(int j=0; j<num_outputs; j++){
            if (norm[j] < std::abs(v[j])) norm[j] = std::abs(v[j]);
        }
    }
    return norm;
}
Data2D<int> GridWavelet::buildUpdateMap(double tolerance, TypeRefinement criteria, int output) const{
    int num_points = points.getNumIndexes();
    Data2D<int> pmap(num_dimensions, num_points);
    if (tolerance == 0.0){
        pmap.fill(1); // if tolerance is 0, refine everything
        return pmap;
    }else{
        pmap.fill(0);
    }

    std::vector<double> norm = getNormalization();

    if ((criteria == refine_classic) || (criteria == refine_parents_first)){
        // classic refinement
        #pragma omp parallel for
        for(int i=0; i<num_points; i++){
            bool small = true;
            const double *s = coefficients.getStrip(i);
            if (output == -1){
                for(size_t k=0; k<((size_t) num_outputs); k++){
                    if (small && ((std::abs(s[k]) / norm[k]) > tolerance)) small = false;
                }
            }else{
                small = !((std::abs(s[output]) / norm[output]) > tolerance);
            }
            if (!small){
                int *p = pmap.getStrip(i);
                std::fill(p, p + num_dimensions, 1);
            }
        }
    }else{
        HierarchyManipulations::SplitDirections split(points);

        for(int s=0; s<split.getNumJobs(); s++){
            int d = split.getJobDirection(s);
            int nump = split.getJobNumPoints(s);
            const int *pnts = split.getJobPoints(s);

            int active_outputs = (output == -1) ? num_outputs : 1;

            Data2D<double> vals;
            vals.resize(active_outputs, nump);
            Data2D<int> indexes;
            indexes.resize(num_dimensions, nump);

            for(int i=0; i<nump; i++){
                const double* v = values.getValues(pnts[i]);
                double *vls = vals.getStrip(i);
                if (output == -1){
                    std::copy(v, v + num_outputs, vls);
                }else{
                    vls[0] = v[output];
                }
                const int *p = points.getIndex(pnts[i]);
                std::copy(p, p + num_dimensions, indexes.getStrip(i));
            }

            MultiIndexSet pointset(num_dimensions, std::move(indexes.getVector()));

            GridWavelet direction_grid;
            direction_grid.setNodes(pointset, active_outputs, order);
            direction_grid.loadNeededPoints(vals.getStrip(0));

            for(int i=0; i<nump; i++){
                bool small = true;
                const double *coeff = direction_grid.coefficients.getStrip(i);
                const double *soeff = coefficients.getStrip(pnts[i]);
                if (output == -1){
                    for(int k=0; k<num_outputs; k++){
                        if (small && ((std::abs(soeff[k]) / norm[k]) > tolerance) && ((std::abs(coeff[k]) / norm[k]) > tolerance)) small = false;
                    }
                }else{
                    if (((std::abs(soeff[output]) / norm[output]) > tolerance) && ((std::abs(coeff[0]) / norm[output]) > tolerance)) small = false;
                }
                pmap.getStrip(pnts[i])[d] = (small) ? 0 : 1;
            }
        }
    }
    return pmap;
}

bool GridWavelet::addParent(const int point[], int direction, Data2D<int> &destination) const{
    std::vector<int> dad(num_dimensions);
    std::copy_n(point, num_dimensions, dad.data());
    bool added = false;
    dad[direction] = rule1D.getParent(point[direction]);
    if (dad[direction] == -2){
        for(int c=0; c<rule1D.getNumPoints(0); c++){
            dad[direction] = c;
            if (points.missing(dad)){
                destination.appendStrip(dad);
                added = true;
            }
        }
    }else if (dad[direction] >= 0){
        if (points.missing(dad)){
            destination.appendStrip(dad);
            added = true;
        }
    }
    return added;
}
void GridWavelet::addChild(const int point[], int direction, Data2D<int> &destination) const{
    std::vector<int> kid(num_dimensions);
    std::copy_n(point, num_dimensions, kid.data());
    int L, R; rule1D.getChildren(point[direction], L, R);
    kid[direction] = L;
    if ((kid[direction] != -1) && points.missing(kid)){
        destination.appendStrip(kid);
    }
    kid[direction] = R;
    if ((kid[direction] != -1) && points.missing(kid)){
        destination.appendStrip(kid);
}
}
void GridWavelet::addChildLimited(const int point[], int direction, const std::vector<int> &level_limits, Data2D<int> &destination) const{
    std::vector<int> kid(num_dimensions);
    std::copy_n(point, num_dimensions, kid.data());
    int L, R; rule1D.getChildren(point[direction], L, R);
    kid[direction] = L;
    if ((kid[direction] != -1)
        && ((level_limits[direction] == -1) || (rule1D.getLevel(L) <= level_limits[direction]))
        && points.missing(kid)){
        destination.appendStrip(kid);
    }
    kid[direction] = R;
    if ((kid[direction] != -1)
        && ((level_limits[direction] == -1) || (rule1D.getLevel(R) <= level_limits[direction]))
        && points.missing(kid)){
        destination.appendStrip(kid);
    }
}

void GridWavelet::clearRefinement(){ needed = MultiIndexSet(); }
const double* GridWavelet::getSurpluses() const{
    return coefficients.getStrip(0);
}

void GridWavelet::beginConstruction(){
    dynamic_values = std::unique_ptr<SimpleConstructData>(new SimpleConstructData);
    if (points.empty()){
        dynamic_values->initial_points = std::move(needed);
        needed = MultiIndexSet();
    }
}
void GridWavelet::writeConstructionData(std::ostream &os, bool iomode) const{
    if (iomode == mode_ascii) dynamic_values->write<mode_ascii>(os); else dynamic_values->write<mode_binary>(os);
}
void GridWavelet::readConstructionData(std::istream &is, bool iomode){
    if (iomode == mode_ascii) dynamic_values = readSimpleConstructionData<mode_ascii>(num_dimensions, num_outputs, is);
    else dynamic_values = readSimpleConstructionData<mode_binary>(num_dimensions, num_outputs, is);
}

namespace WaveManipulations{
inline void touchAllImmediateRelatives(std::vector<int> &point, MultiIndexSet const &mset, RuleWavelet const &rule, std::function<void(int i)> apply){
    for(auto &v : point){
        int save = v; // replace one by one each index of p with either parent or kid

        // check the parents
        v = rule.getParent(save);
        if (v == -2){ // include everything on level 0
            for(v = 0; v < rule.getNumPoints(0); v++){
                int parent_index = mset.getSlot(point);
                if (parent_index > -1)
                    apply(parent_index);
            }
        }else if (v > -1){
            int parent_index = mset.getSlot(point);
            if (parent_index > -1)
                apply(parent_index);
        }

        int kid1, kid2;
        rule.getChildren(save, kid1, kid2);
        std::vector<int> kids = {kid1, kid2};
        for(auto k : kids){
            if (k > -1){
                v = k;
                int parent_index = mset.getSlot(point);
                if (parent_index > -1)
                    apply(parent_index);
            }
        }

        v = save; // restore the original index for the next iteration
    }
}

std::vector<int> computeLevels(MultiIndexSet const &mset, RuleWavelet const &rule){
    size_t num_dimensions = mset.getNumDimensions();
    int num_points = mset.getNumIndexes();
    std::vector<int> level((size_t) num_points);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<num_points; i++){
        const int *p = mset.getIndex(i);
        int current_level = rule.getLevel(p[0]);
        for(size_t j=1; j<num_dimensions; j++){
            current_level += rule.getLevel(p[j]);
        }
        level[i] = current_level;
    }
    return level;
}

inline MultiIndexSet getLargestConnected(MultiIndexSet const &current, MultiIndexSet const &candidates, RuleWavelet const &rule){
    if (candidates.empty()) return MultiIndexSet();
    auto num_dimensions = candidates.getNumDimensions();

    // always consider the points without parents
    MultiIndexSet level_zero = MultiIndexManipulations::generateFullTensorSet(std::vector<int>(num_dimensions, rule.getNumPoints(0)));

    MultiIndexSet result; // keep track of the cumulative result
    MultiIndexSet total = current; // forms a working copy of the entire merged graph

    // do not consider the points already included in total, complexity is level_zero.getNumIndexes()
    if (!total.empty()) level_zero = level_zero.diffSets(total);

    if (level_zero.getNumIndexes() > 0){ // level zero nodes are missing from current
        Data2D<int> roots(num_dimensions, 0);
        for(int i=0; i<level_zero.getNumIndexes(); i++){
            std::vector<int> p(level_zero.getIndex(i), level_zero.getIndex(i) + num_dimensions);
            if (!candidates.missing(p))
                roots.appendStrip(p);
        }

        result = MultiIndexSet(roots);
        if (total.empty()) total = result;
        else total.addMultiIndexSet(result);
    }

    if (total.empty()) return MultiIndexSet(); // current was empty and no roots could be added

    Data2D<int> update;
    do{
        update = Data2D<int>(num_dimensions, 0);

        for(int i=0; i<total.getNumIndexes(); i++){
            std::vector<int> relative(total.getIndex(i), total.getIndex(i) + num_dimensions);
            for(auto &r : relative){
                int k = r; // save the value

                std::vector<int> possible_relatives;
                possible_relatives.reserve((size_t) rule.getNumPoints(0) + 2);

                int kid1, kid2;
                rule.getChildren(k, kid1, kid2);
                possible_relatives.push_back(kid1);
                possible_relatives.push_back(kid2);

                int parent = rule.getParent(k);
                if (parent == -2){
                    for(int p=0; p<rule.getNumPoints(0); p++)
                        possible_relatives.push_back(p);
                }else{
                    possible_relatives.push_back(parent);
                }

                for(auto p : possible_relatives){
                    if (p > -1){
                        r = p;
                        if (!candidates.missing(relative) && total.missing(relative))
                            update.appendStrip(relative);
                    }
                }
                r = k;
            }
        }

        if (update.getNumStrips() > 0){
            MultiIndexSet update_set(update);
            result.addMultiIndexSet(update_set);
            total.addMultiIndexSet(update_set);
        }
    }while(update.getNumStrips() > 0);

    return result;
}
}

std::vector<double> GridWavelet::getCandidateConstructionPoints(double tolerance, TypeRefinement criteria, int output, std::vector<int> const &level_limits){

    MultiIndexSet refine_candidates = getRefinementCanidates(tolerance, criteria, output, level_limits);
    MultiIndexSet new_points = (dynamic_values->initial_points.empty()) ? std::move(refine_candidates) : refine_candidates.diffSets(dynamic_values->initial_points);

    // compute the weights for the new_points points
    std::vector<double> norm = getNormalization();

    auto getDominantSurplus = [&](int i)-> double{
        double dominant = 0.0;
        const double *s = coefficients.getStrip(i);
        if (output == -1){
            for(int k=0; k<num_outputs; k++) dominant = std::max(dominant, std::abs(s[k]) / norm[k]);
        }else{
            dominant = std::abs(s[output]) / norm[output];
        }
        return dominant;
    };

    std::vector<double> refine_weights(new_points.getNumIndexes());

    #pragma omp parallel for
    for(int i=0; i<new_points.getNumIndexes(); i++){
        double weight = 0.0;
        std::vector<int> p(new_points.getIndex(i), new_points.getIndex(i) + num_dimensions); // get the point

        WaveManipulations::touchAllImmediateRelatives(p, points, rule1D,
                                                           [&](int relative)->void{ weight = std::max(weight, getDominantSurplus(relative)); });
        refine_weights[i] = weight; // those will be inverted
    }

    // if using stable refinement, ensure the weight of the parents is never less than the children
    if (!new_points.empty() && ((criteria == refine_parents_first) || (criteria == refine_fds))){
        auto rlevels = WaveManipulations::computeLevels(new_points, rule1D);
        auto split = HierarchyManipulations::splitByLevels((size_t) num_dimensions, new_points.getVector(), rlevels);
        for(auto is = split.rbegin(); is != split.rend(); is++){
            for(int i=0; i<is->getNumStrips(); i++){
                std::vector<int> parent(is->getStrip(i), is->getStrip(i) + num_dimensions);
                double correction = refine_weights[new_points.getSlot(parent)]; // will never be missing
                for(auto &p : parent){
                    int r = p;
                    p = rule1D.getParent(r);
                    if (p > -1){
                        int ip = new_points.getSlot(parent); // if parent is among the refined
                        if (ip != -1) refine_weights[ip] += correction;
                    }else if (p == -2){
                        for(p = 0; p<rule1D.getNumPoints(0); p++){
                            int ip = new_points.getSlot(parent); // if parent is among the refined
                            if (ip != -1) refine_weights[ip] += correction;
                        }
                    }
                    p = r;
                }
            }
        }
    }else if (!new_points.empty() && (criteria == refine_stable)){
        // stable refinement, ensure that if level[i] < level[j] then weight[i] > weight[j]
        auto rlevels = WaveManipulations::computeLevels(new_points, rule1D);
        auto split = HierarchyManipulations::splitByLevels((size_t) num_dimensions, new_points.getVector(), rlevels);
        double max_weight = 0.0;
        for(auto is = split.rbegin(); is != split.rend(); is++){ // loop backwards in levels
            double correction = max_weight;
            for(int i=0; i<is->getNumStrips(); i++){
                int idx = new_points.getSlot(std::vector<int>(is->getStrip(i), is->getStrip(i) + num_dimensions));
                refine_weights[idx] += correction;
                max_weight = std::max(max_weight, refine_weights[idx]);
            }
        }
    }

    // compute the weights for the initial points
    std::vector<int> initial_levels =  WaveManipulations::computeLevels(dynamic_values->initial_points, rule1D);

    std::forward_list<NodeData> weighted_points;
    for(int i=0; i<dynamic_values->initial_points.getNumIndexes(); i++){
        std::vector<int> p(dynamic_values->initial_points.getIndex(i), dynamic_values->initial_points.getIndex(i) + num_dimensions); // write the point to vector
        weighted_points.push_front({p, {-1.0 / ((double) initial_levels[i])}});
    }
    for(int i=0; i<new_points.getNumIndexes(); i++){
        std::vector<int> p(new_points.getIndex(i), new_points.getIndex(i) + num_dimensions); // write the point to vector
        weighted_points.push_front({p, {1.0 / refine_weights[i]}});
    }

    // sort and return the sorted list
    weighted_points.sort([&](const NodeData &a, const NodeData &b)->bool{ return (a.value[0] < b.value[0]); });

    std::vector<double> x(dynamic_values->initial_points.getVector().size() + new_points.getVector().size());
    auto ix = x.begin();
    for(auto t = weighted_points.begin(); t != weighted_points.end(); t++)
        ix = std::transform(t->point.begin(), t->point.end(), ix, [&](int i)->double{ return rule1D.getNode(i); });
    return x;
}

std::vector<int> GridWavelet::getMultiIndex(const double x[]){
    std::vector<int> p(num_dimensions); // convert x to p, maybe expensive
    for(int j=0; j<num_dimensions; j++){
        int i = 0;
        while(std::abs(rule1D.getNode(i) - x[j]) > Maths::num_tol) i++;
        p[j] = i;
    }
    return p;
}
void GridWavelet::loadConstructedPoint(const double x[], const std::vector<double> &y){ loadConstructedPoint(x, 1, y.data()); }
void GridWavelet::loadConstructedPoint(const double x[], int numx, const double y[]){
    Utils::Wrapper2D<const double> wrapx(num_dimensions, x);
    std::vector<std::vector<int>> pnts(numx);
    #pragma omp parallel for
    for(int i=0; i<numx; i++)
        pnts[i] = getMultiIndex(wrapx.getStrip(i));

    if (!dynamic_values->initial_points.empty()){
        Data2D<int> combined_pnts(num_dimensions, numx);
        for(int i=0; i<numx; i++)
            std::copy_n(pnts[i].begin(), num_dimensions, combined_pnts.getIStrip(i));
        dynamic_values->initial_points = dynamic_values->initial_points.diffSets(MultiIndexSet(combined_pnts));
    }

    Utils::Wrapper2D<const double> wrapy(num_outputs, y);
    for(int i=0; i<numx; i++)
        dynamic_values->data.push_front({std::move(pnts[i]), std::vector<double>(wrapy.getStrip(i), wrapy.getStrip(i) + num_outputs)});

    Data2D<int> candidates(num_dimensions, (int) std::distance(dynamic_values->data.begin(), dynamic_values->data.end()));
    for(struct{ int i; std::forward_list<NodeData>::iterator d; } p = {0, dynamic_values->data.begin()};
        p.d != dynamic_values->data.end(); p.i++, p.d++){
        std::copy_n(p.d->point.begin(), num_dimensions, candidates.getIStrip(p.i));
    }
    auto new_points = WaveManipulations::getLargestConnected(points, MultiIndexSet(candidates), rule1D);
    if (new_points.empty()) return;

    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaBasis(); // the points will change, clear the cache
    clearCudaCoefficients();
    #endif

    auto vals = dynamic_values->extractValues(new_points);
    if (points.empty()){
        points = std::move(new_points);
        values.setValues(std::move(vals));
    }else{
        values.addValues(points, new_points, vals.data());
        points.addMultiIndexSet(new_points);
    }
    buildInterpolationMatrix();
    recomputeCoefficients(); // costly, but the only option under the circumstances
}
void GridWavelet::finishConstruction(){ dynamic_values.reset(); }

void GridWavelet::evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_points, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++){
        double const *this_x = xwrap.getStrip(i);
        double *this_y = ywrap.getStrip(i);
        for(int j=0; j<num_points; j++){
            const int* p = work.getIndex(j);
            double v = 1.0;
            for(int k=0; k<num_dimensions; k++){
                v *= rule1D.eval(p[k], this_x[k]);
                if (v == 0.0){ break; }; // MIRO: evaluating the wavelets is expensive, stop if any one of them is zero
            }
            this_y[j] = v;
        }
    }
}

void GridWavelet::setHierarchicalCoefficients(const double c[], TypeAcceleration){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaCoefficients();
    #endif
    int num_points = getNumPoints();
    size_t size_coeff = ((size_t) num_points) * ((size_t) num_outputs);
    if (!points.empty()){
        clearRefinement();
    }else{
        points = std::move(needed);
        needed = MultiIndexSet();
    }
    coefficients.resize(num_outputs, num_points);
    std::copy_n(c, size_coeff, coefficients.getStrip(0));

    values.resize(num_outputs, num_points);
    values.getVector().resize(size_coeff);

    std::vector<double> x(((size_t) num_points) * ((size_t) num_dimensions));
    getPoints(x.data());
    evaluateBatch(x.data(), points.getNumIndexes(), values.getValues(0));
}

void GridWavelet::integrateHierarchicalFunctions(double integrals[]) const{
    const MultiIndexSet& work = (points.empty()) ? needed : points;
    int const num_points = work.getNumIndexes();
    for(int i=0; i<num_points; i++)
        integrals[i] = evalIntegral(work.getIndex(i));
}

MultiIndexSet GridWavelet::getRefinementCanidates(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits) const{
    Data2D<int> pmap = buildUpdateMap(tolerance, criteria, output);

    bool useParents = (criteria == refine_fds) || (criteria == refine_parents_first);

    Data2D<int> refined(num_dimensions, 0);

    int num_points = points.getNumIndexes();

    if (level_limits.empty()){
        for(int i=0; i<num_points; i++){
            int *p = pmap.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (p[j] == 1){ // if this dimension needs to be refined
                    if (!(useParents && addParent(points.getIndex(i), j, refined))){
                        addChild(points.getIndex(i), j, refined);
                    }
                }
            }
        }
    }else{
        for(int i=0; i<num_points; i++){
            int *p = pmap.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (p[j] == 1){ // if this dimension needs to be refined
                    if (!(useParents && addParent(points.getIndex(i), j, refined))){
                        addChildLimited(points.getIndex(i), j, level_limits, refined);
                    }
                }
            }
        }
    }

    MultiIndexSet result = (refined.getNumStrips() > 0) ? MultiIndexSet(refined) : MultiIndexSet();

    if (criteria == refine_stable){ // complete needed to a lower set
        size_t num_added = 1; // set to 1 to start the loop
        while(num_added > 0){
            Data2D<int> addons(num_dimensions, 0);
            int num_needed = result.getNumIndexes();

            for(int i=0; i<num_needed; i++){
                std::vector<int> parent(result.getIndex(i), result.getIndex(i) + num_dimensions);
                for(auto &p : parent){
                    int r = p;
                    p = rule1D.getParent(r);
                    if ((p > -1) && result.missing(parent) && points.missing(parent))
                        addons.appendStrip(parent);
                    if (p == -2){ // all parents on level 0
                        for(int j=0; j<rule1D.getNumPoints(0); j++){
                            p = j;
                            if (result.missing(parent) && points.missing(parent))
                                addons.appendStrip(parent);
                        }
                    }
                    p = r;
                }
            }

            num_added = addons.getNumStrips();
            if (num_added > 0)
                result.addMultiIndexSet(MultiIndexSet(addons));
        }
    }

    return result;
}

void GridWavelet::setSurplusRefinement(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits){
    clearRefinement();
    needed = getRefinementCanidates(tolerance, criteria, output, level_limits);
}

void GridWavelet::clearAccelerationData(){
    #ifdef Tasmanian_ENABLE_CUDA
    cuda_cache.reset();
    cuda_cachef.reset();
    #endif
}

}

#endif
