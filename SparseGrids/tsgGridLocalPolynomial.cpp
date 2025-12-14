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

#ifndef __TASMANIAN_SPARSE_GRID_LPOLY_CPP
#define __TASMANIAN_SPARSE_GRID_LPOLY_CPP

#include "tsgGridLocalPolynomial.hpp"
#include "tsgTPLWrappers.hpp"

namespace TasGrid{

template<bool iomode> void GridLocalPolynomial::write(std::ostream &os) const{
    if (iomode == mode_ascii){ os << std::scientific; os.precision(17); }
    IO::writeNumbers<iomode, IO::pad_line>(os, num_dimensions, num_outputs, order, top_level);
    IO::writeRule<iomode>(RuleLocal::getRule(effective_rule), os);
    IO::writeFlag<iomode, IO::pad_auto>(!points.empty(), os);
    if (!points.empty()) points.write<iomode>(os);
    if (iomode == mode_ascii){ // backwards compatible: surpluses and needed, or needed and surpluses
        IO::writeFlag<iomode, IO::pad_auto>((surpluses.getNumStrips() != 0), os);
        if (!surpluses.empty()) surpluses.writeVector<iomode, IO::pad_line>(os);
        IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
        if (!needed.empty()) needed.write<iomode>(os);
    }else{
        IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
        if (!needed.empty()) needed.write<iomode>(os);
        IO::writeFlag<iomode, IO::pad_auto>((surpluses.getNumStrips() != 0), os);
        if (!surpluses.empty()) surpluses.writeVector<iomode, IO::pad_line>(os);
    }
    IO::writeFlag<iomode, IO::pad_auto>((parents.getNumStrips() != 0), os);
    if (!parents.empty()) parents.writeVector<iomode, IO::pad_line>(os);

    IO::writeNumbers<iomode, IO::pad_rspace>(os, static_cast<int>(roots.size()));
    if (roots.size() > 0){ // the tree is empty, can happend when using dynamic construction
        IO::writeVector<iomode, IO::pad_line>(roots, os);
        IO::writeVector<iomode, IO::pad_line>(pntr, os);
        IO::writeVector<iomode, IO::pad_line>(indx, os);
    }

    if (num_outputs > 0) values.write<iomode>(os);
}

template void GridLocalPolynomial::write<mode_ascii>(std::ostream &) const;
template void GridLocalPolynomial::write<mode_binary>(std::ostream &) const;

GridLocalPolynomial::GridLocalPolynomial(AccelerationContext const *acc, int cnum_dimensions, int cnum_outputs, int depth, int corder, TypeOneDRule crule, const std::vector<int> &level_limits)
    : BaseCanonicalGrid(acc, cnum_dimensions, cnum_outputs, MultiIndexSet(), MultiIndexSet(), StorageSet()),
      order(corder),
      effective_rule(RuleLocal::getEffectiveRule(order, (crule == rule_semilocalp and order < 2) ? rule_localp : crule))
    {

    MultiIndexSet tensors = MultiIndexManipulations::selectTensors((size_t) num_dimensions, depth, type_level, [&](int i) -> int{ return i; }, std::vector<int>(), level_limits);

    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            needed = MultiIndexManipulations::generateNestedPoints(tensors,
                [&](int l) -> int{ return RuleLocal::getNumPoints<RuleLocal::erule::pwc>(l); });
            break;
        case RuleLocal::erule::localp:
            needed = MultiIndexManipulations::generateNestedPoints(tensors,
                [&](int l) -> int{ return RuleLocal::getNumPoints<RuleLocal::erule::localp>(l); });
            break;
        case RuleLocal::erule::semilocalp:
            needed = MultiIndexManipulations::generateNestedPoints(tensors,
                [&](int l) -> int{ return RuleLocal::getNumPoints<RuleLocal::erule::semilocalp>(l); });
            break;
        case RuleLocal::erule::localp0:
            needed = MultiIndexManipulations::generateNestedPoints(tensors,
                [&](int l) -> int{ return RuleLocal::getNumPoints<RuleLocal::erule::localp0>(l); });
            break;
        default: // case RuleLocal::erule::localpb:
            needed = MultiIndexManipulations::generateNestedPoints(tensors,
                [&](int l) -> int{ return RuleLocal::getNumPoints<RuleLocal::erule::localpb>(l); });
            break;
    };

    buildTree();

    if (num_outputs == 0){
        points = std::move(needed);
        needed = MultiIndexSet();
        parents = HierarchyManipulations::computeDAGup(points, effective_rule);
    }else{
        values.resize(num_outputs, needed.getNumIndexes());
    }
}

GridLocalPolynomial::GridLocalPolynomial(AccelerationContext const *acc, GridLocalPolynomial const *pwpoly, int ibegin, int iend) :
    BaseCanonicalGrid(acc, *pwpoly, ibegin, iend),
    order(pwpoly->order),
    top_level(pwpoly->top_level),
    surpluses((num_outputs == pwpoly->num_outputs) ? pwpoly->surpluses : pwpoly->surpluses.splitData(ibegin, iend)),
    parents(pwpoly->parents),
    roots(pwpoly->roots),
    pntr(pwpoly->pntr),
    indx(pwpoly->indx),
    effective_rule(pwpoly->effective_rule){

    if (pwpoly->dynamic_values){
        dynamic_values = std::make_unique<SimpleConstructData>(*pwpoly->dynamic_values);
        if (num_outputs != pwpoly->num_outputs) dynamic_values->restrictData(ibegin, iend);
    }
}

GridLocalPolynomial::GridLocalPolynomial(AccelerationContext const *acc, int cnum_dimensions, int cnum_outputs, int corder, TypeOneDRule crule,
                                         std::vector<int> &&pnts, std::vector<double> &&vals, std::vector<double> &&surps)
    : BaseCanonicalGrid(acc, cnum_dimensions, cnum_outputs, MultiIndexSet(cnum_dimensions, std::move(pnts)), MultiIndexSet(),
                        StorageSet(cnum_outputs, static_cast<int>(vals.size() / cnum_outputs), std::move(vals))),
    order(corder),
    surpluses(Data2D<double>(cnum_outputs, points.getNumIndexes(), std::move(surps))),
    effective_rule(RuleLocal::getEffectiveRule(order, crule)){

    buildTree();
}

struct localp_loaded{};
struct localp_needed{};

template<RuleLocal::erule eff_rule, typename points_mode>
void GridLocalPolynomial::getPoints(double *x) const {
    int num_points = (std::is_same<points_mode, localp_loaded>::value) ? points.getNumIndexes() : needed.getNumIndexes();
    Utils::Wrapper2D<double> split(num_dimensions, x);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<num_points; i++) {
        int const *p = (std::is_same<points_mode, localp_loaded>::value) ? points.getIndex(i) : needed.getIndex(i);
        double *s = split.getStrip(i);
        for(int j=0; j<num_dimensions; j++)
            s[j] = RuleLocal::getNode<eff_rule>(p[j]);
    }
}
void GridLocalPolynomial::getLoadedPoints(double *x) const{
    using pmode = localp_loaded;
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            getPoints<RuleLocal::erule::pwc, pmode>(x);
            break;
        case RuleLocal::erule::localp:
            getPoints<RuleLocal::erule::localp, pmode>(x);
            break;
        case RuleLocal::erule::semilocalp:
            getPoints<RuleLocal::erule::semilocalp, pmode>(x);
            break;
        case RuleLocal::erule::localp0:
            getPoints<RuleLocal::erule::localp0, pmode>(x);
            break;
        default: // case RuleLocal::erule::localpb:
            getPoints<RuleLocal::erule::localpb, pmode>(x);
            break;
    };
}
void GridLocalPolynomial::getNeededPoints(double *x) const{
    using pmode = localp_needed;
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            getPoints<RuleLocal::erule::pwc, pmode>(x);
            break;
        case RuleLocal::erule::localp:
            getPoints<RuleLocal::erule::localp, pmode>(x);
            break;
        case RuleLocal::erule::semilocalp:
            getPoints<RuleLocal::erule::semilocalp, pmode>(x);
            break;
        case RuleLocal::erule::localp0:
            getPoints<RuleLocal::erule::localp0, pmode>(x);
            break;
        default: // case RuleLocal::erule::localpb:
            getPoints<RuleLocal::erule::localpb, pmode>(x);
            break;
    };
}
void GridLocalPolynomial::getPoints(double *x) const{
    if (points.empty()){ getNeededPoints(x); }else{ getLoadedPoints(x); }
}

void GridLocalPolynomial::evaluate(const double x[], double y[]) const{
    std::fill_n(y, num_outputs, 0.0);
    std::vector<int> sindx; // dummy variables, never references in mode 0 below
    std::vector<double> svals;
    walkTree<0>(points, x, sindx, svals, y);
}
void GridLocalPolynomial::evaluateBatchOpenMP(const double x[], int num_x, double y[]) const{
    if (num_x == 1){ evaluate(x, y); return; }
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_outputs, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++)
        evaluate(xwrap.getStrip(i), ywrap.getStrip(i));
}
void GridLocalPolynomial::evaluateBatch(const double x[], int num_x, double y[]) const{
    switch(acceleration->mode){
        case accel_gpu_magma:
        case accel_gpu_cuda: {
            acceleration->setDevice();
            if ((order == -1) || (order > 2) || (num_x == 1)){
                // GPU evaluations are available only for order 0, 1, and 2. Cubic will come later, but higher order will not be supported.
                // cannot use GPU to accelerate the evaluation of a single vector
                evaluateGpuMixed(x, num_x, y);
                return;
            }
            GpuVector<double> gpu_x(acceleration, num_dimensions, num_x, x), gpu_result(acceleration, num_x, num_outputs);
            evaluateBatchGPU(gpu_x.data(), num_x, gpu_result.data());
            gpu_result.unload(acceleration, y);
            break;
        }
        case accel_gpu_cublas: {
            acceleration->setDevice();
            evaluateGpuMixed(x, num_x, y);
            break;
        }
        case accel_cpu_blas: {
            if (acceleration->algorithm_select == AccelerationContext::algorithm_sparse or
                (acceleration->algorithm_select == AccelerationContext::algorithm_autoselect and num_outputs <= 1024)){
                evaluateBatchOpenMP(x, num_x, y);
                return;
            }

            std::vector<int> sindx, spntr;
            std::vector<double> svals;
            buildSpareBasisMatrix(x, num_x, 32, spntr, sindx, svals); // build sparse matrix corresponding to x

            int num_points = points.getNumIndexes();
            double nnz = (double) spntr[num_x];
            double total_size = ((double) num_x) * ((double) num_points);

            if ((acceleration->algorithm_select == AccelerationContext::algorithm_dense)
                or ((acceleration->algorithm_select == AccelerationContext::algorithm_autoselect) and (nnz / total_size > 0.1))){
                // potentially wastes a lot of memory
                Data2D<double> A(num_points, num_x, 0.0);
                for(int i=0; i<num_x; i++){
                    double *row = A.getStrip(i);
                    for(int j=spntr[i]; j<spntr[i+1]; j++) row[sindx[j]] = svals[j];
                }
                TasBLAS::denseMultiply(num_outputs, num_x, num_points, 1.0, surpluses.getStrip(0), A.getStrip(0), 0.0, y);
            }else{
                Utils::Wrapper2D<double> ywrap(num_outputs, y);
                #pragma omp parallel for
                for(int i=0; i<num_x; i++){
                    double *this_y = ywrap.getStrip(i);
                    std::fill(this_y, this_y + num_outputs, 0.0);
                    for(int j=spntr[i]; j<spntr[i+1]; j++){
                        double v = svals[j];
                        const double *s = surpluses.getStrip(sindx[j]);
                        for(int k=0; k<num_outputs; k++) this_y[k] += v * s[k];
                    }
                }
            }
            break;
        }
        default: {
            evaluateBatchOpenMP(x, num_x, y);
            break;
        }
    }
}

void GridLocalPolynomial::loadNeededValuesGPU(const double *vals){
    updateValues(vals);

    std::vector<int> levels = HierarchyManipulations::computeLevels(points, effective_rule);

    std::vector<Data2D<int>> lpnts = HierarchyManipulations::splitByLevels(points, levels);
    std::vector<Data2D<double>> lvals = HierarchyManipulations::splitByLevels(values, levels);

    Data2D<double> allx(num_dimensions, points.getNumIndexes());
    getPoints(allx.data());

    std::vector<Data2D<double>> lx = HierarchyManipulations::splitByLevels(allx, levels);

    MultiIndexSet cumulative_poitns((size_t) num_dimensions, lpnts[0].release());

    StorageSet cumulative_surpluses = StorageSet(num_outputs, cumulative_poitns.getNumIndexes(), lvals[0].release());

    for(size_t l = 1; l < lpnts.size(); l++){ // loop over the levels
        // note that level_points.getNumIndexes() == lx[l].getNumStrips() == lvals[l].getNumStrips()
        MultiIndexSet level_points(num_dimensions, lpnts[l].release());

        GridLocalPolynomial upper_grid(acceleration, num_dimensions, num_outputs, order, getRule(),
                                       std::vector<int>(cumulative_poitns.begin(), cumulative_poitns.end()), // copy cumulative_poitns
                                       std::vector<double>(Utils::size_mult(num_outputs, cumulative_poitns.getNumIndexes())),  // dummy values, will not be read or used
                                       std::vector<double>(cumulative_surpluses.begin(), cumulative_surpluses.end())); // copy the cumulative_surpluses

        Data2D<double> upper_evaluate(num_outputs, level_points.getNumIndexes());
        int batch_size = 20000; // needs tuning
        if (acceleration->algorithm_select == AccelerationContext::algorithm_dense){
            // dense uses lots of memory, try to keep it contained to about 4GB
            batch_size = 536870912 / upper_grid.getNumPoints() - 2 * (num_outputs + num_dimensions);
            if (batch_size < 100) batch_size = 100; // use at least 100 points
        }

        for(int i=0; i<level_points.getNumIndexes(); i += batch_size)
            upper_grid.evaluateBatch(lx[l].getStrip(i), std::min(batch_size, level_points.getNumIndexes() - i), upper_evaluate.getStrip(i));

        double *level_surps = lvals[l].getStrip(0); // maybe use BLAS here
        const double *uv = upper_evaluate.getStrip(0);
        for(size_t i=0, s = upper_evaluate.getTotalEntries(); i < s; i++) level_surps[i] -= uv[i];

        cumulative_surpluses.addValues(cumulative_poitns, level_points, level_surps);
        cumulative_poitns += level_points;
    }

    surpluses = Data2D<double>(num_outputs, points.getNumIndexes(), cumulative_surpluses.release());
}
void GridLocalPolynomial::evaluateGpuMixed(const double x[], int num_x, double y[]) const{
    loadGpuSurpluses<double>();

    std::vector<int> sindx, spntr;
    std::vector<double> svals;

    if (num_x > 1){
        buildSpareBasisMatrix(x, num_x, 32, spntr, sindx, svals);
    }else{
        walkTree<2>(points, x, sindx, svals, nullptr);
        #ifdef Tasmanian_ENABLE_DPCPP
        // CUDA and HIP have methods for sparse-vector times dense matrix or vector
        // therefore, CUDA/HIP do not use spntr when num_x is equal to 1, but DPC++ needs to set it too
        spntr = {0, static_cast<int>(sindx.size())};
        #endif
    }
    TasGpu::sparseMultiplyMixed(acceleration, num_outputs, num_x, points.getNumIndexes(), 1.0, gpu_cache->surpluses, spntr, sindx, svals, y);
}
template<typename T> void GridLocalPolynomial::evaluateBatchGPUtempl(const T gpu_x[], int cpu_num_x, T gpu_y[]) const{
    if ((order == -1) || (order > 2)) throw std::runtime_error("ERROR: GPU evaluations are availabe only for local polynomial grid with order 0, 1, and 2");
    loadGpuSurpluses<T>();
    int num_points = points.getNumIndexes();

    if (acceleration->algorithm_select == AccelerationContext::algorithm_dense){
        GpuVector<T> gpu_basis(acceleration, cpu_num_x, num_points);
        evaluateHierarchicalFunctionsGPU(gpu_x, cpu_num_x, gpu_basis.data());

        TasGpu::denseMultiply(acceleration, num_outputs, cpu_num_x, num_points, 1.0, getGpuCache<T>()->surpluses, gpu_basis, 0.0, gpu_y);
    }else{
        GpuVector<int> gpu_spntr, gpu_sindx;
        GpuVector<T> gpu_svals;
        buildSparseBasisMatrixGPU(gpu_x, cpu_num_x, gpu_spntr, gpu_sindx, gpu_svals);

        TasGpu::sparseMultiply(acceleration, num_outputs, cpu_num_x, num_points, 1.0, getGpuCache<T>()->surpluses, gpu_spntr, gpu_sindx, gpu_svals, gpu_y);
    }
}
void GridLocalPolynomial::evaluateBatchGPU(const double gpu_x[], int cpu_num_x, double gpu_y[]) const{
    evaluateBatchGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
void GridLocalPolynomial::evaluateBatchGPU(const float gpu_x[], int cpu_num_x, float gpu_y[]) const{
    evaluateBatchGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
void GridLocalPolynomial::evaluateHierarchicalFunctionsGPU(const double gpu_x[], int cpu_num_x, double *gpu_y) const{
    loadGpuBasis<double>();
    TasGpu::devalpwpoly(acceleration, order, RuleLocal::getRule(effective_rule), num_dimensions, cpu_num_x, getNumPoints(), gpu_x, gpu_cache->nodes.data(), gpu_cache->support.data(), gpu_y);
}
void GridLocalPolynomial::buildSparseBasisMatrixGPU(const double gpu_x[], int cpu_num_x, GpuVector<int> &gpu_spntr, GpuVector<int> &gpu_sindx, GpuVector<double> &gpu_svals) const{
    loadGpuBasis<double>();
    loadGpuHierarchy<double>();
    TasGpu::devalpwpoly_sparse(acceleration, order, RuleLocal::getRule(effective_rule), num_dimensions, cpu_num_x, gpu_x,
                               gpu_cache->nodes, gpu_cache->support,
                               gpu_cache->hpntr, gpu_cache->hindx, gpu_cache->hroots, gpu_spntr, gpu_sindx, gpu_svals);
}
void GridLocalPolynomial::evaluateHierarchicalFunctionsGPU(const float gpu_x[], int cpu_num_x, float *gpu_y) const{
    loadGpuBasis<float>();
    TasGpu::devalpwpoly(acceleration, order, RuleLocal::getRule(effective_rule), num_dimensions, cpu_num_x, getNumPoints(), gpu_x, gpu_cachef->nodes.data(), gpu_cachef->support.data(), gpu_y);
}
void GridLocalPolynomial::buildSparseBasisMatrixGPU(const float gpu_x[], int cpu_num_x, GpuVector<int> &gpu_spntr, GpuVector<int> &gpu_sindx, GpuVector<float> &gpu_svals) const{
    loadGpuBasis<float>();
    loadGpuHierarchy<float>();
    TasGpu::devalpwpoly_sparse(acceleration, order, RuleLocal::getRule(effective_rule), num_dimensions, cpu_num_x, gpu_x,
                               gpu_cachef->nodes, gpu_cachef->support,
                               gpu_cachef->hpntr, gpu_cachef->hindx, gpu_cachef->hroots, gpu_spntr, gpu_sindx, gpu_svals);
}
template<typename T> void GridLocalPolynomial::loadGpuBasis() const{
    auto& ccache = getGpuCache<T>();
    if (!ccache) ccache = CudaLocalPolynomialData<T>{};
    if (!ccache->nodes.empty()) return;

    Data2D<double> cpu_nodes(num_dimensions, getNumPoints());
    getPoints(cpu_nodes.getStrip(0));
    ccache->nodes.load(acceleration, cpu_nodes.totalSize(), cpu_nodes.data());

    Data2D<T> cpu_support = [&](void)->Data2D<T>{
            const MultiIndexSet &work = (points.empty()) ? needed : points;
            switch(effective_rule) {
                case RuleLocal::erule::pwc:
                    return encodeSupportForGPU<0, rule_localp, T>(work);
                case RuleLocal::erule::localp:
                    return (order == 1) ? encodeSupportForGPU<1, rule_localp, T>(work)
                                        : encodeSupportForGPU<2, rule_localp, T>(work);
                case RuleLocal::erule::semilocalp:
                    return (order == 1) ? encodeSupportForGPU<1, rule_semilocalp, T>(work)
                                        : encodeSupportForGPU<2, rule_semilocalp, T>(work);
                case RuleLocal::erule::localp0:
                    return (order == 1) ? encodeSupportForGPU<1, rule_localp0, T>(work)
                                        : encodeSupportForGPU<2, rule_localp0, T>(work);
                default: // case RuleLocal::erule::localpb:
                    return (order == 1) ? encodeSupportForGPU<1, rule_localpb, T>(work)
                                        : encodeSupportForGPU<2, rule_localpb, T>(work);
            };
        }();
    ccache->support.load(acceleration, cpu_support.totalSize(), cpu_support.data());
}
void GridLocalPolynomial::clearGpuBasisHierarchy(){
    if (gpu_cache) gpu_cache->clearBasisHierarchy();
    if (gpu_cachef) gpu_cachef->clearBasisHierarchy();
}
template<typename T> void GridLocalPolynomial::loadGpuHierarchy() const{
    auto& ccache = getGpuCache<T>();
    if (!ccache) ccache = CudaLocalPolynomialData<T>{};
    if (!ccache->hpntr.empty()) return;

    ccache->hpntr.load(acceleration, pntr);
    ccache->hindx.load(acceleration, indx);
    ccache->hroots.load(acceleration, roots);
}
template<typename T> void GridLocalPolynomial::loadGpuSurpluses() const{
    auto& ccache = getGpuCache<T>();
    if (!ccache) ccache = CudaLocalPolynomialData<T>{};
    if (ccache->surpluses.size() != 0) return;
    ccache->surpluses.load(acceleration, surpluses.totalSize(), surpluses.data());
}
void GridLocalPolynomial::clearGpuSurpluses(){
    if (gpu_cache) gpu_cache->surpluses.clear();
    if (gpu_cachef) gpu_cachef->surpluses.clear();
}

void GridLocalPolynomial::updateValues(double const *vals){
    clearGpuSurpluses();
    if (needed.empty()){
        values.setValues(vals);
    }else{
        clearGpuBasisHierarchy();
        if (points.empty()){ // initial grid, just relabel needed as points (loaded)
            values.setValues(vals);
            points = std::move(needed);
            needed = MultiIndexSet();
        }else{ // merge needed and points
            values.addValues(points, needed, vals);
            points += needed;
            needed = MultiIndexSet();
            buildTree();
        }
    }
}
void GridLocalPolynomial::loadNeededValues(const double *vals){
    #ifdef Tasmanian_ENABLE_GPU
    if (acceleration->on_gpu()){
        acceleration->setDevice();
        loadNeededValuesGPU(vals);
        return;
    }
    #endif
    updateValues(vals);
    recomputeSurpluses();
}
void GridLocalPolynomial::mergeRefinement(){
    if (needed.empty()) return; // nothing to do
    clearGpuSurpluses();
    int num_all_points = getNumLoaded() + getNumNeeded();
    values.setValues(std::vector<double>(Utils::size_mult(num_all_points, num_outputs), 0.0));
    if (points.empty()){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        points += needed;
        needed = MultiIndexSet();
        buildTree();
    }
    surpluses = Data2D<double>(num_outputs, num_all_points);
}

void GridLocalPolynomial::beginConstruction(){
    dynamic_values = std::make_unique<SimpleConstructData>();
    if (points.empty()){
        dynamic_values->initial_points = std::move(needed);
        needed = MultiIndexSet();
        roots.clear();
        pntr.clear();
        indx.clear();
    }
}
void GridLocalPolynomial::writeConstructionData(std::ostream &os, bool iomode) const{
    if (iomode == mode_ascii) dynamic_values->write<mode_ascii>(os); else dynamic_values->write<mode_binary>(os);
}
void GridLocalPolynomial::readConstructionData(std::istream &is, bool iomode){
    if (iomode == mode_ascii)
        dynamic_values = std::make_unique<SimpleConstructData>(is, num_dimensions, num_outputs, IO::mode_ascii_type());
    else
        dynamic_values = std::make_unique<SimpleConstructData>(is, num_dimensions, num_outputs, IO::mode_binary_type());
}
template<RuleLocal::erule effrule>
std::vector<double> GridLocalPolynomial::getCandidateConstructionPoints(double tolerance, TypeRefinement criteria, int output,
                                                                        std::vector<int> const &level_limits, double const *scale_correction){
    // combine the initial points with negative weights and the refinement candidates with surplus weights (no need to normalize, the sort uses relative values)
    MultiIndexSet refine_candidates = getRefinementCanidates<effrule>(tolerance, criteria, output, level_limits, scale_correction);
    MultiIndexSet new_points = (dynamic_values->initial_points.empty()) ? std::move(refine_candidates) : refine_candidates - dynamic_values->initial_points;

    // compute the weights for the new_points points
    std::vector<double> norm = getNormalization();

    int active_outputs = (output == -1) ? num_outputs : 1;
    Utils::Wrapper2D<double const> scale(active_outputs, scale_correction);
    std::vector<double> default_scale;
    if (scale_correction == nullptr){ // if no scale provided, assume default 1.0
        default_scale = std::vector<double>(Utils::size_mult(active_outputs, points.getNumIndexes()), 1.0);
        scale = Utils::Wrapper2D<double const>(active_outputs, default_scale.data());
    }

    auto getDominantSurplus = [&](int i)-> double{
        double dominant = 0.0;
        const double *s = surpluses.getStrip(i);
        const double *c = scale.getStrip(i);
        if (output == -1){
            for(int k=0; k<num_outputs; k++) dominant = std::max(dominant, c[k] * std::abs(s[k]) / norm[k]);
        }else{
            dominant = c[0] * std::abs(s[output]) / norm[output];
        }
        return dominant;
    };

    std::vector<double> refine_weights(new_points.getNumIndexes());

    #pragma omp parallel for
    for(int i=0; i<new_points.getNumIndexes(); i++){
        double weight = 0.0;
        std::vector<int> p = new_points.copyIndex(i);

        HierarchyManipulations::touchAllImmediateRelatives<effrule>(p, points,
                [&](int relative)->void{ weight = std::max(weight, getDominantSurplus(relative)); });
        refine_weights[i] = weight; // those will be inverted
    }

    // if using stable refinement, ensure the weight of the parents is never less than the children
    if (!new_points.empty() && ((criteria == refine_parents_first) || (criteria == refine_fds))){
        auto rlevels = HierarchyManipulations::computeLevels<effrule>(new_points);
        auto split = HierarchyManipulations::splitByLevels(new_points, rlevels);
        for(auto is = split.rbegin(); is != split.rend(); is++){
            for(int i=0; i<is->getNumStrips(); i++){
                std::vector<int> parent(is->getStrip(i), is->getStrip(i) + num_dimensions);
                double correction = refine_weights[new_points.getSlot(parent)]; // will never be missing
                for(auto &p : parent){
                    int r = p;
                    p = RuleLocal::getParent<effrule>(r);
                    int ip = (p == -1) ? -1 : new_points.getSlot(parent); // if parent is among the refined
                    if (ip != -1) refine_weights[ip] += correction;
                    p = RuleLocal::getStepParent<effrule>(r);
                    ip = (p == -1) ? -1 : new_points.getSlot(parent); // if parent is among the refined
                    if (ip != -1) refine_weights[ip] += correction;
                    p = r;
                }
            }
        }
    }else if (!new_points.empty() && (criteria == refine_stable)){
        // stable refinement, ensure that if level[i] < level[j] then weight[i] > weight[j]
        auto rlevels = HierarchyManipulations::computeLevels<effrule>(new_points);
        auto split = HierarchyManipulations::splitByLevels(new_points, rlevels);
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
    std::vector<int> initial_levels = HierarchyManipulations::computeLevels<effrule>(dynamic_values->initial_points);

    std::forward_list<NodeData> weighted_points;
    for(int i=0; i<dynamic_values->initial_points.getNumIndexes(); i++)
        weighted_points.push_front({dynamic_values->initial_points.copyIndex(i), {-1.0 / ((double) initial_levels[i])}});
    for(int i=0; i<new_points.getNumIndexes(); i++)
        weighted_points.push_front({new_points.copyIndex(i), {1.0 / refine_weights[i]}});

    // sort and return the sorted list
    weighted_points.sort([&](const NodeData &a, const NodeData &b)->bool{ return (a.value[0] < b.value[0]); });

    return listToLocalNodes(weighted_points, num_dimensions, [&](int i)->double{ return RuleLocal::getNode<effrule>(i); });
}
std::vector<double> GridLocalPolynomial::getCandidateConstructionPoints(double tolerance, TypeRefinement criteria, int output,
                                                                        std::vector<int> const &level_limits, double const *scale_correction) {
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            return getCandidateConstructionPoints<RuleLocal::erule::pwc>(tolerance, criteria, output, level_limits, scale_correction);
        case RuleLocal::erule::localp:
            return getCandidateConstructionPoints<RuleLocal::erule::localp>(tolerance, criteria, output, level_limits, scale_correction);
        case RuleLocal::erule::semilocalp:
            return getCandidateConstructionPoints<RuleLocal::erule::semilocalp>(tolerance, criteria, output, level_limits, scale_correction);
        case RuleLocal::erule::localp0:
            return getCandidateConstructionPoints<RuleLocal::erule::localp0>(tolerance, criteria, output, level_limits, scale_correction);
        default: // case RuleLocal::erule::localpb:
            return getCandidateConstructionPoints<RuleLocal::erule::localpb>(tolerance, criteria, output, level_limits, scale_correction);
    };
}

template<RuleLocal::erule effrule>
std::vector<int> GridLocalPolynomial::getMultiIndex(const double x[]){
    std::vector<int> p(num_dimensions); // convert x to p, maybe expensive
    for(int j=0; j<num_dimensions; j++){
        int i = 0;
        while(std::abs(RuleLocal::getNode<effrule>(i) - x[j]) > Maths::num_tol) i++;
        p[j] = i;
    }
    return p;
}

template<RuleLocal::erule effrule>
void GridLocalPolynomial::loadConstructedPoint(const double x[], const std::vector<double> &y){
    auto p = getMultiIndex<effrule>(x);

    dynamic_values->initial_points.removeIndex(p);

    bool isConnected = false;
    HierarchyManipulations::touchAllImmediateRelatives<effrule>(p, points, [&](int)->void{ isConnected = true; });
    int lvl = RuleLocal::getLevel<effrule>(p[0]);
    for(int j=1; j<num_dimensions; j++) lvl += RuleLocal::getLevel<effrule>(p[j]);

    if (isConnected || (lvl == 0)){
        expandGrid<effrule>(p, y);
        loadConstructedPoints<effrule>();
    }else{
        dynamic_values->data.push_front({p, y});
    }
}
void GridLocalPolynomial::loadConstructedPoint(const double x[], const std::vector<double> &y){
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            return loadConstructedPoint<RuleLocal::erule::pwc>(x, y);
        case RuleLocal::erule::localp:
            return loadConstructedPoint<RuleLocal::erule::localp>(x, y);
        case RuleLocal::erule::semilocalp:
            return loadConstructedPoint<RuleLocal::erule::semilocalp>(x, y);
        case RuleLocal::erule::localp0:
            return loadConstructedPoint<RuleLocal::erule::localp0>(x, y);
        default: // case RuleLocal::erule::localpb:
            return loadConstructedPoint<RuleLocal::erule::localpb>(x, y);
    };
}
template<RuleLocal::erule effrule>
void GridLocalPolynomial::expandGrid(const std::vector<int> &point, const std::vector<double> &value){
    if (points.empty()){ // only one point
        points = MultiIndexSet((size_t) num_dimensions, std::vector<int>(point));
        values = StorageSet(num_outputs, 1, std::vector<double>(value));
        surpluses = Data2D<double>(num_outputs, 1, std::vector<double>(value)); // one value is its own surplus
    }else{ // merge with existing points
        // compute the surplus for the point
        std::vector<double> xnode(num_dimensions);
        for(int j=0; j<num_dimensions; j++)
            xnode[j] = RuleLocal::getNode<effrule>(point[j]);

        std::vector<double> approximation(num_outputs), surp(num_outputs);
        evaluate(xnode.data(), approximation.data());
        std::transform(approximation.begin(), approximation.end(), value.begin(), surp.begin(), [&](double e, double v)->double{ return v - e; });

        std::vector<int> graph = getSubGraph<effrule>(point); // get the descendant nodes that must be updated later

        values.addValues(points, MultiIndexSet(num_dimensions, std::vector<int>(point)), value.data()); // added the value

        points.addSortedIndexes(point); // add the point
        int newindex = points.getSlot(point);
        surpluses.appendStrip(newindex, surp); // find the index of the new point

        for(auto &g : graph) if (g >= newindex) g++; // all points belowe the newindex have been shifted down by one spot

        std::vector<int> levels(points.getNumIndexes(), 0); // compute the levels, but only for the new indexes
        for(auto &g : graph){
            int const *pnt = points.getIndex(g);
            int l = RuleLocal::getLevel<effrule>(pnt[0]);
            for(int j=1; j<num_dimensions; j++) l += RuleLocal::getLevel<effrule>(pnt[j]);
            levels[g] = l;

            std::copy_n(values.getValues(g), num_outputs, surpluses.getStrip(g)); // reset the surpluses to the values (will be updated)
        }

        // compute the current DAG and update the surplused for the descendants
        updateSurpluses<effrule>(points, top_level + 1, levels, HierarchyManipulations::computeDAGup<effrule>(points));
    }
    buildTree(); // the tree is needed for evaluate(), must be rebuild every time the points set is updated
}
template<RuleLocal::erule effrule>
void GridLocalPolynomial::loadConstructedPoint(const double x[], int numx, const double y[]){
    Utils::Wrapper2D<const double> wrapx(num_dimensions, x);
    std::vector<std::vector<int>> pnts(numx);
    #pragma omp parallel for
    for(int i=0; i<numx; i++)
        pnts[i] = getMultiIndex<effrule>(wrapx.getStrip(i));

    if (!dynamic_values->initial_points.empty()){
        Data2D<int> combined_pnts(num_dimensions, numx);
        for(int i=0; i<numx; i++)
            std::copy_n(pnts[i].begin(), num_dimensions, combined_pnts.getIStrip(i));
        dynamic_values->initial_points = dynamic_values->initial_points - combined_pnts;
    }

    Utils::Wrapper2D<const double> wrapy(num_outputs, y);
    for(int i=0; i<numx; i++)
        dynamic_values->data.push_front({std::move(pnts[i]), std::vector<double>(wrapy.getStrip(i), wrapy.getStrip(i) + num_outputs)});

    loadConstructedPoints<effrule>();
}
void GridLocalPolynomial::loadConstructedPoint(const double x[], int numx, const double y[]){
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            return loadConstructedPoint<RuleLocal::erule::pwc>(x, numx, y);
        case RuleLocal::erule::localp:
            return loadConstructedPoint<RuleLocal::erule::localp>(x, numx, y);
        case RuleLocal::erule::semilocalp:
            return loadConstructedPoint<RuleLocal::erule::semilocalp>(x, numx, y);
        case RuleLocal::erule::localp0:
            return loadConstructedPoint<RuleLocal::erule::localp0>(x, numx, y);
        default: // case RuleLocal::erule::localpb:
            return loadConstructedPoint<RuleLocal::erule::localpb>(x, numx, y);
    };
}
template<RuleLocal::erule effrule>
void GridLocalPolynomial::loadConstructedPoints(){
    Data2D<int> candidates(num_dimensions, (int) std::distance(dynamic_values->data.begin(), dynamic_values->data.end()));
    for(struct{ int i; std::forward_list<NodeData>::iterator d; } p = {0, dynamic_values->data.begin()};
        p.d != dynamic_values->data.end(); p.i++, p.d++){
        std::copy_n(p.d->point.begin(), num_dimensions, candidates.getIStrip(p.i));
    }
    auto new_points = HierarchyManipulations::getLargestConnected<effrule>(points, MultiIndexSet(candidates));
    if (new_points.empty()) return;

    clearGpuBasisHierarchy(); // the points will change, clear the cache
    clearGpuSurpluses();

    auto vals = dynamic_values->extractValues(new_points);
    if (points.empty()){
        points = std::move(new_points);
        values.setValues(std::move(vals));
    }else{
        values.addValues(points, new_points, vals.data());
        points += new_points;
    }
    buildTree();
    recomputeSurpluses<effrule>(); // costly, but the only option under the circumstances
}
void GridLocalPolynomial::finishConstruction(){ dynamic_values.reset(); }

template<RuleLocal::erule effrule>
std::vector<int> GridLocalPolynomial::getSubGraph(std::vector<int> const &point) const{
    std::vector<int> graph, p = point;
    std::vector<bool> used(points.getNumIndexes(), false);
    int max_1d_kids = RuleLocal::getMaxNumKids<effrule>();
    int max_kids = max_1d_kids * num_dimensions;

    std::vector<int> monkey_count(1, 0), monkey_tail;

    while(monkey_count[0] < max_kids){
        if (monkey_count.back() < max_kids){
            int dim = monkey_count.back() / max_1d_kids;
            monkey_tail.push_back(p[dim]);
            p[dim] = RuleLocal::getKid<effrule>(monkey_tail.back(), monkey_count.back() % max_1d_kids);
            int slot = points.getSlot(p);
            if ((slot == -1) || used[slot]){ // this kid is missing
                p[dim] = monkey_tail.back();
                monkey_tail.pop_back();
                monkey_count.back()++;
            }else{ // found kid, go deeper in the graph
                graph.push_back(slot);
                used[slot] = true;
                monkey_count.push_back(0);
            }
        }else{
            monkey_count.pop_back();
            int dim = monkey_count.back() / max_1d_kids;
            p[dim] = monkey_tail.back();
            monkey_tail.pop_back();
            monkey_count.back()++;
        }
    }

    return graph;
}

void GridLocalPolynomial::getInterpolationWeights(const double x[], double weights[]) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;

    std::vector<int> active_points;
    std::vector<double> hbasis_values;
    std::fill_n(weights, work.getNumIndexes(), 0.0);

    // construct a sparse vector and apply transpose surplus transformation
    walkTree<1>(work, x, active_points, hbasis_values, nullptr);
    auto ibasis = hbasis_values.begin();
    for(auto i : active_points) weights[i] = *ibasis++;
    applyTransformationTransposed<0>(weights, work, active_points);
}

void GridLocalPolynomial::getDifferentiationWeights(const double x[], double weights[]) const {
    // Based on GridLocalPolynomial::getInterpolationWeights().
    const MultiIndexSet &work = (points.empty()) ? needed : points;

    std::vector<int> active_points;
    std::vector<double> diff_hbasis_values;
    std::fill_n(weights, work.getNumIndexes(), 0.0);

    walkTree<4>(work, x, active_points, diff_hbasis_values, nullptr);
    auto ibasis = diff_hbasis_values.begin();
    for(auto i : active_points)
        for (int d=0; d<num_dimensions; d++)
            weights[i * num_dimensions + d] = *ibasis++;
    applyTransformationTransposed<1>(weights, work, active_points);
}

void GridLocalPolynomial::evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_points, y);
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            #pragma omp parallel for
            for(int i=0; i<num_x; i++){
                double const *this_x = xwrap.getStrip(i);
                double *this_y = ywrap.getStrip(i);
                bool dummy;
                for(int j=0; j<num_points; j++)
                    this_y[j] = evalBasisSupported<RuleLocal::erule::pwc>(work.getIndex(j), this_x, dummy);
            }
            break;
        case RuleLocal::erule::localp:
            #pragma omp parallel for
            for(int i=0; i<num_x; i++){
                double const *this_x = xwrap.getStrip(i);
                double *this_y = ywrap.getStrip(i);
                bool dummy;
                for(int j=0; j<num_points; j++)
                    this_y[j] = evalBasisSupported<RuleLocal::erule::localp>(work.getIndex(j), this_x, dummy);
            }
            break;
        case RuleLocal::erule::semilocalp:
            #pragma omp parallel for
            for(int i=0; i<num_x; i++){
                double const *this_x = xwrap.getStrip(i);
                double *this_y = ywrap.getStrip(i);
                bool dummy;
                for(int j=0; j<num_points; j++)
                    this_y[j] = evalBasisSupported<RuleLocal::erule::semilocalp>(work.getIndex(j), this_x, dummy);
            }
            break;
        case RuleLocal::erule::localp0:
            #pragma omp parallel for
            for(int i=0; i<num_x; i++){
                double const *this_x = xwrap.getStrip(i);
                double *this_y = ywrap.getStrip(i);
                bool dummy;
                for(int j=0; j<num_points; j++)
                    this_y[j] = evalBasisSupported<RuleLocal::erule::localp0>(work.getIndex(j), this_x, dummy);
            }
            break;
        default: // case RuleLocal::erule::localpb:
            #pragma omp parallel for
            for(int i=0; i<num_x; i++){
                double const *this_x = xwrap.getStrip(i);
                double *this_y = ywrap.getStrip(i);
                bool dummy;
                for(int j=0; j<num_points; j++)
                    this_y[j] = evalBasisSupported<RuleLocal::erule::localpb>(work.getIndex(j), this_x, dummy);
            }
            break;
    };
}

template<RuleLocal::erule effrule>
void GridLocalPolynomial::recomputeSurpluses() {
    surpluses = Data2D<double>(num_outputs, points.getNumIndexes(), std::vector<double>(values.begin(), values.end()));

    // There are two available algorithms here:
    // - global sparse Kronecker (kron), implemented here in recomputeSurpluses()
    // - sparse matrix in matrix-free form (mat), implemented in updateSurpluses()
    //
    // (kron) has the additional restriction that it will only work when the hierarchy is complete
    // i.e., all point (multi-indexes) have all of their parents
    // this is guaranteed for a full-tensor grid, non-adaptive grid, or a grid adapted with the stable refinement strategy
    // other refinement strategies or using dynamic refinement (even with a stable refinement strategy)
    // may yield a hierarchy-complete grid, but this is not mathematically guaranteed
    //
    // computing the dagUp allows us to check (at runtime) if the grid is_complete, and exclude (kron) if incomplete
    //    the completeness check can be done on the fly but it does incur cost
    //
    // approximate computational cost, let n be the number of points and d be the number of dimensions
    // (kron) d n n^(1/d) due to standard Kronecker reasons, except it is hard to find the "effective" n due to sparsity
    // (mat) d^2 n log(n) since there are d log(n) ancestors and basis functions are product of d one dimensional functions
    //    naturally, there are constants and those also depend on the system, e.g., number of threads, thread scheduling, cache ...
    //    (mat) has a more even load per thread and parallelizes better
    //
    // for d = 1, the algorithms are the same, except (kron) explicitly forms the matrix and the matrix free (mat) version is much better
    // for d = 2, the (mat) algorithm is still faster
    // for d = 3, and sufficiently large size the (mat) algorithm still wins
    //            the breaking point depends on n, the order and system
    // for d = 4 and above, the (kron) algorithm is much faster
    //           it is possible that for sufficiently large n (mat) will win again
    //           but tested on 12 cores CPUs (Intel and AMD) up to n = 3.5 million, the (kron) method is over 2x faster
    // higher dimensions will favor (kron) even more

    if (num_dimensions <= 2 or (num_dimensions == 3 and points.getNumIndexes() > 2000000)) {
        Data2D<int> dagUp = HierarchyManipulations::computeDAGup<effrule>(points);
        std::vector<int> level = HierarchyManipulations::computeLevels<effrule>(points);
        updateSurpluses<effrule>(points, top_level, level, dagUp);
        return;
    }

    bool is_complete = true;
    Data2D<int> dagUp = HierarchyManipulations::computeDAGup<effrule>(points, is_complete);

    if (not is_complete) {
        // incomplete hierarchy, must use the slow algorithm
        std::vector<int> level = HierarchyManipulations::computeLevels<effrule>(points);
        updateSurpluses<effrule>(points, top_level, level, dagUp);
        return;
    }

    int num_nodes = 1 + *std::max_element(points.begin(), points.end());

    std::vector<int> vpntr, vindx;
    std::vector<double> vvals;
    RuleLocal::van_matrix<effrule>(order, num_nodes, vpntr, vindx, vvals);

    std::vector<std::vector<int>> map;
    std::vector<std::vector<int>> lines1d;
    MultiIndexManipulations::resortIndexes(points, map, lines1d);

    for(int d=num_dimensions-1; d>=0; d--) {
        #pragma omp parallel for schedule(dynamic)
        for(int job = 0; job < static_cast<int>(lines1d[d].size() - 1); job++) {
            for(int i=lines1d[d][job]+1; i<lines1d[d][job+1]; i++) {
                double *row_strip = surpluses.getStrip(map[d][i]);

                int row = points.getIndex(map[d][i])[d];
                int im  = vpntr[row];
                int ijx = lines1d[d][job];
                int ix  = points.getIndex(map[d][ijx])[d];

                while(vindx[im] < row or ix < row) {
                    if (vindx[im] < ix) {
                        ++im; // move the index of the matrix pattern (missing entry)
                    } else if (ix < vindx[im]) {
                        // entry not connected, move to the next one
                        ++ijx;
                        ix = points.getIndex(map[d][ijx])[d];
                    } else {
                        double const *col_strip = surpluses.getStrip(map[d][ijx]);
                        double const v = vvals[im];
                        for(int k=0; k<num_outputs; k++)
                            row_strip[k] -= v * col_strip[k];

                        ++im;
                        ++ijx;
                        ix = points.getIndex(map[d][ijx])[d];
                    }
                }
            }
        }
    }
}

void GridLocalPolynomial::recomputeSurpluses() {
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            recomputeSurpluses<RuleLocal::erule::pwc>();
            break;
        case RuleLocal::erule::localp:
            recomputeSurpluses<RuleLocal::erule::localp>();
            break;
        case RuleLocal::erule::semilocalp:
            recomputeSurpluses<RuleLocal::erule::semilocalp>();
            break;
        case RuleLocal::erule::localp0:
            recomputeSurpluses<RuleLocal::erule::localp0>();
            break;
        default: // case RuleLocal::erule::localpb:
            recomputeSurpluses<RuleLocal::erule::localpb>();
            break;
    };
}

template<RuleLocal::erule effrule>
void GridLocalPolynomial::updateSurpluses(MultiIndexSet const &work, int max_level, std::vector<int> const &level, Data2D<int> const &dagUp){
    int num_points = work.getNumIndexes();
    int max_parents = num_dimensions * RuleLocal::getMaxNumParents<effrule>();

    std::vector<std::vector<int>> indexses_for_levels((size_t) max_level+1);
    for(int i=0; i<num_points; i++)
        if (level[i] > 0) indexses_for_levels[level[i]].push_back(i);

    for(int l=1; l<=max_level; l++){
        int level_size = (int) indexses_for_levels[l].size();
        #pragma omp parallel
        {
            std::vector<double> x(num_dimensions);

            std::vector<int> monkey_count(max_level + 1);
            std::vector<int> monkey_tail(max_level + 1);
            std::vector<bool> used(num_points, false);

            #pragma omp for schedule(dynamic)
            for(int s=0; s<level_size; s++){
                int i = indexses_for_levels[l][s];

                int const *pnt = work.getIndex(i);
                for(int j=0; j<num_dimensions; j++)
                    x[j] = RuleLocal::getNode<effrule>(pnt[j]);

                std::fill(used.begin(), used.end(), false);

                double *surpi = surpluses.getStrip(i);

                int current = 0;

                monkey_count[0] = 0;
                monkey_tail[0] = i;

                while(monkey_count[0] < max_parents){
                    if (monkey_count[current] < max_parents){
                        int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                        if ((branch == -1) || (used[branch])){
                            monkey_count[current]++;
                        }else{
                            const double *branch_surp = surpluses.getStrip(branch);
                            double basis_value = evalBasisRaw<effrule>(work.getIndex(branch), x.data());
                            for(int k=0; k<num_outputs; k++)
                                surpi[k] -= basis_value * branch_surp[k];
                            used[branch] = true;

                            monkey_count[++current] = 0;
                            monkey_tail[current] = branch;
                        }
                    }else{
                        monkey_count[--current]++;
                    }
                }
            } // for-loop
        } // omp parallel
    } // level loop
}

template<int mode, RuleLocal::erule effrule>
void GridLocalPolynomial::applyTransformationTransposed(double weights[], const MultiIndexSet &work, const std::vector<int> &active_points) const {
    Data2D<int> lparents = (parents.getNumStrips() != work.getNumIndexes()) ? // if the current dag loaded in parents does not reflect the indexes in work
                            HierarchyManipulations::computeDAGup<effrule>(work) :
                            Data2D<int>();

    const Data2D<int> &dagUp = (parents.getNumStrips() != work.getNumIndexes()) ? lparents : parents;

    std::vector<int> level(active_points.size());
    int active_top_level = 0;
    for(size_t i=0; i<active_points.size(); i++){
        const int *p = work.getIndex(active_points[i]);
        int current_level = RuleLocal::getLevel<effrule>(p[0]);
        for(int j=1; j<num_dimensions; j++){
            current_level += RuleLocal::getLevel<effrule>(p[j]);
        }
        if (active_top_level < current_level) active_top_level = current_level;
        level[i] = current_level;
    }

    std::vector<int> monkey_count(top_level+1);
    std::vector<int> monkey_tail(top_level+1);
    std::vector<bool> used(work.getNumIndexes());
    int max_parents = RuleLocal::getMaxNumParents<effrule>() * num_dimensions;

    std::vector<double> node(num_dimensions);

    for(int l=active_top_level; l>0; l--){
        for(size_t i=0; i<active_points.size(); i++){
            if (level[i] == l){
                int const *pnt = work.getIndex(active_points[i]);
                for(int j=0; j<num_dimensions; j++)
                    node[j] = RuleLocal::getNode<effrule>(pnt[j]);

                std::fill(used.begin(), used.end(), false);

                monkey_count[0] = 0;
                monkey_tail[0] = active_points[i];
                int current = 0;

                while(monkey_count[0] < max_parents){
                    if (monkey_count[current] < max_parents){
                        int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                        if ((branch == -1) || used[branch]){
                            monkey_count[current]++;
                        }else{
                            const int *func = work.getIndex(branch);
                            double basis_value = RuleLocal::evalRaw<effrule>(order, func[0], node[0]);
                            for(int j=1; j<num_dimensions; j++) basis_value *= RuleLocal::evalRaw<effrule>(order, func[j], node[j]);

                            if (mode == 0) {
                                weights[branch] -= weights[active_points[i]] * basis_value;
                            } else {
                                for (int d=0; d<num_dimensions; d++)
                                    weights[branch * num_dimensions + d] -=
                                            weights[active_points[i] * num_dimensions + d] * basis_value;
                            }
                            used[branch] = true;

                            monkey_count[++current] = 0;
                            monkey_tail[current] = branch;
                        }
                    }else{
                        monkey_count[--current]++;
                    }
                }
            }
        }
    }
}

template<int mode>
void GridLocalPolynomial::applyTransformationTransposed(double weights[], const MultiIndexSet &work, const std::vector<int> &active_points) const {
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            applyTransformationTransposed<mode, RuleLocal::erule::pwc>(weights, work, active_points);
            break;
        case RuleLocal::erule::localp:
            applyTransformationTransposed<mode, RuleLocal::erule::localp>(weights, work, active_points);
            break;
        case RuleLocal::erule::semilocalp:
            applyTransformationTransposed<mode, RuleLocal::erule::semilocalp>(weights, work, active_points);
            break;
        case RuleLocal::erule::localp0:
            applyTransformationTransposed<mode, RuleLocal::erule::localp0>(weights, work, active_points);
            break;
        default: // case RuleLocal::erule::localpb:
            applyTransformationTransposed<mode, RuleLocal::erule::localpb>(weights, work, active_points);
            break;
    };
}

void GridLocalPolynomial::buildSpareBasisMatrix(const double x[], int num_x, int num_chunk, std::vector<int> &spntr, std::vector<int> &sindx, std::vector<double> &svals) const{
    std::vector<std::vector<int>> tindx;
    std::vector<std::vector<double>> tvals;
    std::vector<int> numnz;
    buildSparseMatrixBlockForm(x, num_x, num_chunk, numnz, tindx, tvals);

    spntr = std::vector<int>((size_t) num_x + 1);

    spntr[0] = 0;
    for(size_t i=1; i<spntr.size(); i++)
        spntr[i] += numnz[i-1] + spntr[i-1];

    sindx = std::vector<int>((size_t) spntr.back());
    svals = std::vector<double>((size_t) spntr.back());

    auto ii = sindx.begin();
    for(auto &idx : tindx) ii = std::copy(idx.begin(), idx.end(), ii);
    auto iv = svals.begin();
    for(auto &vls : tvals) iv = std::copy(vls.begin(), vls.end(), iv);
}
void GridLocalPolynomial::buildSpareBasisMatrixStatic(const double x[], int num_x, int num_chunk, int *spntr, int *sindx, double *svals) const{
    std::vector<std::vector<int>> tindx;
    std::vector<std::vector<double>> tvals;
    std::vector<int> numnz;
    buildSparseMatrixBlockForm(x, num_x, num_chunk, numnz, tindx, tvals);

    int nz = 0;
    for(int i=0; i<num_x; i++){
        spntr[i] = nz;
        nz += numnz[i];
    }
    spntr[num_x] = nz;
    size_t c = 0;
    for(auto &idx : tindx){
        std::copy(idx.begin(), idx.end(), &(sindx[c]));
        c += idx.size();
    }
    c = 0;
    for(auto &vls : tvals){
        std::copy(vls.begin(), vls.end(), &(svals[c]));
        c += vls.size();
    }
}
int GridLocalPolynomial::getSpareBasisMatrixNZ(const double x[], int num_x) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;

    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    std::vector<int> num_nz(num_x);

    #pragma omp parallel for
    for(int i=0; i<num_x; i++){
        std::vector<int> sindx;
        std::vector<double> svals;
        walkTree<1>(work, xwrap.getStrip(i), sindx, svals, nullptr);
        num_nz[i] = (int) sindx.size();
    }

    return std::accumulate(num_nz.begin(), num_nz.end(), 0);
}

void GridLocalPolynomial::buildSparseMatrixBlockForm(const double x[], int num_x, int num_chunk, std::vector<int> &numnz, std::vector<std::vector<int>> &tindx, std::vector<std::vector<double>> &tvals) const{
    // numnz will be resized to (num_x + 1) with the last entry set to a dummy zero
    numnz.resize((size_t) num_x);
    int num_blocks = num_x / num_chunk + ((num_x % num_chunk != 0) ? 1 : 0);

    tindx.resize(num_blocks);
    tvals.resize(num_blocks);

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);

    #pragma omp parallel for
    for(int b=0; b<num_blocks; b++){
        tindx[b].resize(0);
        tvals[b].resize(0);
        int chunk_size = (b < num_blocks - 1) ? num_chunk : (num_x - (num_blocks - 1) * num_chunk);
        for(int i = b * num_chunk; i < b * num_chunk + chunk_size; i++){
            numnz[i] = (int) tindx[b].size();
            walkTree<1>(work, xwrap.getStrip(i), tindx[b], tvals[b], nullptr);
            numnz[i] = (int) tindx[b].size() - numnz[i];
        }
    }
}

void GridLocalPolynomial::buildTree(){
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();

    Data2D<int> kids;
    std::vector<int> level;

    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            kids = HierarchyManipulations::computeDAGDown<RuleLocal::erule::pwc>(work);
            level = HierarchyManipulations::computeLevels<RuleLocal::erule::pwc>(work);
            break;
        case RuleLocal::erule::localp:
            kids = HierarchyManipulations::computeDAGDown<RuleLocal::erule::localp>(work);
            level = HierarchyManipulations::computeLevels<RuleLocal::erule::localp>(work);
            break;
        case RuleLocal::erule::semilocalp:
            kids = HierarchyManipulations::computeDAGDown<RuleLocal::erule::semilocalp>(work);
            level = HierarchyManipulations::computeLevels<RuleLocal::erule::semilocalp>(work);
            break;
        case RuleLocal::erule::localp0:
            kids = HierarchyManipulations::computeDAGDown<RuleLocal::erule::localp0>(work);
            level = HierarchyManipulations::computeLevels<RuleLocal::erule::localp0>(work);
            break;
        default: // case RuleLocal::erule::localpb:
            kids = HierarchyManipulations::computeDAGDown<RuleLocal::erule::localpb>(work);
            level = HierarchyManipulations::computeLevels<RuleLocal::erule::localpb>(work);
            break;
    };
    top_level = *std::max_element(level.begin(), level.end());

    int max_kids = (int) kids.getStride();

    Data2D<int> tree(max_kids, num_points, -1);
    std::vector<bool> free(num_points, true);

    std::vector<int> monkey_count((size_t) (top_level + 1));
    std::vector<int> monkey_tail(monkey_count.size());

    roots = std::vector<int>();
    int next_root = 0; // zero is always a root and is included at index 0

    while(next_root != -1){
        roots.push_back(next_root);
        free[next_root] = false;

        monkey_tail[0] = next_root;
        monkey_count[0] = 0;
        size_t current = 0;

        while(monkey_count[0] < max_kids){
            if (monkey_count[current] < max_kids){
                int kid = kids.getStrip(monkey_tail[current])[monkey_count[current]];
                if ((kid == -1) || (!free[kid])){
                    monkey_count[current]++; // no kid, keep counting
                }else{
                    tree.getStrip(monkey_tail[current])[monkey_count[current]] = kid;
                    monkey_count[++current] = 0;
                    monkey_tail[current] = kid;
                    free[kid] = false;
                }
            }else{
                monkey_count[--current]++; // done with all kids here
            }
        }

        next_root = -1;
        int next_level = top_level + 1;
        for(int i=0; i<num_points; i++){
            if (free[i] && (level[i] < next_level)){
                next_root = i;
                next_level = level[i];
            }
        }
    }

    pntr = std::vector<int>((size_t) (num_points + 1), 0);
    for(int i=0; i<num_points; i++)
        pntr[i+1] = pntr[i] + (int) std::count_if(tree.getIStrip(i), tree.getIStrip(i) + max_kids, [](int k)->bool{ return (k > -1); });

    indx = std::vector<int>((size_t) ((pntr[num_points] > 0) ? pntr[num_points] : 1));
    std::copy_if(tree.begin(), tree.end(), indx.begin(), [](int t)->bool{ return (t > -1); });
}

void GridLocalPolynomial::integrateHierarchicalFunctions(double integrals[]) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;

    std::vector<double> w, x;
    if (order == -1 or order > 3)
        OneDimensionalNodes::getGaussLegendre(((order == -1) ? top_level : order) / 2 + 1, w, x);

    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            #pragma omp parallel for
            for(int i=0; i<work.getNumIndexes(); i++){
                const int* p = work.getIndex(i);
                integrals[i] = RuleLocal::getArea<RuleLocal::erule::pwc>(order, p[0], w, x);
                for(int j=1; j<num_dimensions; j++)
                    integrals[i] *= RuleLocal::getArea<RuleLocal::erule::pwc>(order, p[j], w, x);
            }
            break;
        case RuleLocal::erule::localp:
            #pragma omp parallel for
            for(int i=0; i<work.getNumIndexes(); i++){
                const int* p = work.getIndex(i);
                integrals[i] = RuleLocal::getArea<RuleLocal::erule::localp>(order, p[0], w, x);
                for(int j=1; j<num_dimensions; j++)
                    integrals[i] *= RuleLocal::getArea<RuleLocal::erule::localp>(order, p[j], w, x);
            }
            break;
        case RuleLocal::erule::semilocalp:
            #pragma omp parallel for
            for(int i=0; i<work.getNumIndexes(); i++){
                const int* p = work.getIndex(i);
                integrals[i] = RuleLocal::getArea<RuleLocal::erule::semilocalp>(order, p[0], w, x);
                for(int j=1; j<num_dimensions; j++)
                    integrals[i] *= RuleLocal::getArea<RuleLocal::erule::semilocalp>(order, p[j], w, x);
            }
            break;
        case RuleLocal::erule::localp0:
            #pragma omp parallel for
            for(int i=0; i<work.getNumIndexes(); i++){
                const int* p = work.getIndex(i);
                integrals[i] = RuleLocal::getArea<RuleLocal::erule::localp0>(order, p[0], w, x);
                for(int j=1; j<num_dimensions; j++)
                    integrals[i] *= RuleLocal::getArea<RuleLocal::erule::localp0>(order, p[j], w, x);
            }
            break;
        default: // case RuleLocal::erule::localpb:
            #pragma omp parallel for
            for(int i=0; i<work.getNumIndexes(); i++){
                const int* p = work.getIndex(i);
                integrals[i] = RuleLocal::getArea<RuleLocal::erule::localpb>(order, p[0], w, x);
                for(int j=1; j<num_dimensions; j++)
                    integrals[i] *= RuleLocal::getArea<RuleLocal::erule::localpb>(order, p[j], w, x);
            }
            break;
    };
}
template<RuleLocal::erule effrule>
void GridLocalPolynomial::getQuadratureWeights(double *weights) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    integrateHierarchicalFunctions(weights);

    std::vector<int> monkey_count(top_level+1);
    std::vector<int> monkey_tail(top_level+1);

    double basis_value;

    Data2D<int> lparents;
    if (parents.getNumStrips() != work.getNumIndexes())
        lparents = HierarchyManipulations::computeDAGup<effrule>(work);

    const Data2D<int> &dagUp = (parents.getNumStrips() != work.getNumIndexes()) ? lparents : parents;

    int num_points = work.getNumIndexes();
    std::vector<int> level = HierarchyManipulations::computeLevels<effrule>(work);

    int max_parents = RuleLocal::getMaxNumParents<effrule>() * num_dimensions;

    std::vector<double> node(num_dimensions);
    std::vector<bool> used(work.getNumIndexes(), false);

    for(int l=top_level; l>0; l--){
        for(int i=0; i<num_points; i++){
            if (level[i] == l){
                int const *pnt = work.getIndex(i);
                for(int j=0; j<num_dimensions; j++)
                    node[j] = RuleLocal::getNode<effrule>(pnt[j]);

                std::fill(used.begin(), used.end(), false);

                monkey_count[0] = 0;
                monkey_tail[0] = i;
                int current = 0;

                while(monkey_count[0] < max_parents){
                    if (monkey_count[current] < max_parents){
                        int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                        if ((branch == -1) || used[branch]){
                            monkey_count[current]++;
                        }else{
                            const int *func = work.getIndex(branch);
                            basis_value = RuleLocal::evalRaw<effrule>(order, func[0], node[0]);
                            for(int j=1; j<num_dimensions; j++) basis_value *= RuleLocal::evalRaw<effrule>(order, func[j], node[j]);
                            weights[branch] -= weights[i] * basis_value;
                            used[branch] = true;

                            monkey_count[++current] = 0;
                            monkey_tail[current] = branch;
                        }
                    }else{
                        monkey_count[--current]++;
                    }
                }
            }
        }
    }
}
void GridLocalPolynomial::getQuadratureWeights(double *weights) const{
    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            getQuadratureWeights<RuleLocal::erule::pwc>(weights);
            break;
        case RuleLocal::erule::localp:
            getQuadratureWeights<RuleLocal::erule::localp>(weights);
            break;
        case RuleLocal::erule::semilocalp:
            getQuadratureWeights<RuleLocal::erule::semilocalp>(weights);
            break;
        case RuleLocal::erule::localp0:
            getQuadratureWeights<RuleLocal::erule::localp0>(weights);
            break;
        default: // case RuleLocal::erule::localpb:
            getQuadratureWeights<RuleLocal::erule::localpb>(weights);
            break;
    };
}

void GridLocalPolynomial::integrate(double q[], double *conformal_correction) const{
    int num_points = points.getNumIndexes();
    std::fill(q, q + num_outputs, 0.0);

    if (conformal_correction == 0){
        std::vector<double> integrals(num_points);
        integrateHierarchicalFunctions(integrals.data());
        for(int i=0; i<num_points; i++){
            const double *s = surpluses.getStrip(i);
            double wi = integrals[i];
            for(int k=0; k<num_outputs; k++) q[k] += wi * s[k];
        }
    }else{
        std::vector<double> w(num_points);
        getQuadratureWeights(w.data());
        for(int i=0; i<num_points; i++){
            double wi = w[i] * conformal_correction[i];
            const double *vals = values.getValues(i);
            for(int k=0; k<num_outputs; k++) q[k] += wi * vals[k];
        }
    }
}

void GridLocalPolynomial::differentiate(const double x[], double jacobian[]) const {
    std::fill_n(jacobian, num_outputs * num_dimensions, 0.0);
    // dummy variables, never references in mode 3 below
    std::vector<int> sindx;
    std::vector<double> svals;
    walkTree<3>(points, x, sindx, svals, jacobian);
}

std::vector<double> GridLocalPolynomial::getNormalization() const{
    std::vector<double> norms(num_outputs);
    for(int i=0; i<points.getNumIndexes(); i++){
        const double *v = values.getValues(i);
        for(int j=0; j<num_outputs; j++){
            if (norms[j] < std::abs(v[j])) norms[j] = std::abs(v[j]);
        }
    }
    return norms;
}

template<RuleLocal::erule effrule>
Data2D<int> GridLocalPolynomial::buildUpdateMap(double tolerance, TypeRefinement criteria, int output, const double *scale_correction) const{
    int num_points = points.getNumIndexes();
    Data2D<int> pmap(num_dimensions, num_points,
                     std::vector<int>(Utils::size_mult(num_dimensions, num_points),
                                      (tolerance == 0.0) ? 1 : 0) // tolerance 0 means "refine everything"
                    );
    if (tolerance == 0.0) return pmap;

    std::vector<double> norm = getNormalization();

    int active_outputs = (output == -1) ? num_outputs : 1;
    Utils::Wrapper2D<double const> scale(active_outputs, scale_correction);
    std::vector<double> default_scale;
    if (scale_correction == nullptr){
        default_scale = std::vector<double>(Utils::size_mult(active_outputs, num_points), 1.0);
        scale = Utils::Wrapper2D<double const>(active_outputs, default_scale.data());
    }

    if ((criteria == refine_classic) || (criteria == refine_parents_first)){
        #pragma omp parallel for
        for(int i=0; i<num_points; i++){
            bool small = true;
            const double *s = surpluses.getStrip(i);
            const double *c = scale.getStrip(i);
            if (output == -1){
                for(int k=0; k<num_outputs; k++) small = small && ((c[k] * std::abs(s[k]) / norm[k]) <= tolerance);
            }else{
                small = ((c[0] * std::abs(s[output]) / norm[output]) <= tolerance);
            }
            if (!small)
                std::fill_n(pmap.getStrip(i), num_dimensions, 1);
        }
    }else{
        // construct a series of 1D interpolants and use a refinement criteria that is a combination of the two hierarchical coefficients
        Data2D<int> dagUp = HierarchyManipulations::computeDAGup<effrule>(points);

        int max_1D_parents = RuleLocal::getMaxNumParents<effrule>();

        HierarchyManipulations::SplitDirections split(points);

        #pragma omp parallel
        {
            int max_nump = split.getMaxNumPoints();
            std::vector<int> global_to_pnts(num_points);
            std::vector<int> levels(max_nump);

            std::vector<int> monkey_count;
            std::vector<int> monkey_tail;
            std::vector<bool> used;
            if (max_1D_parents > 1) {
                monkey_count.resize(top_level + 1);
                monkey_tail.resize(top_level + 1);
                used.resize(max_nump, false);
            }

            #pragma omp for
            for(int j=0; j<split.getNumJobs(); j++){ // split.getNumJobs() gives the number of 1D interpolants to construct
                int d = split.getJobDirection(j);
                int nump = split.getJobNumPoints(j);
                const int *pnts = split.getJobPoints(j);

                int max_level = 0;

                Data2D<double> vals(active_outputs, nump);

                if (output == -1) {
                    for(int i=0; i<nump; i++){
                        std::copy_n(values.getValues(pnts[i]), num_outputs, vals.getStrip(i));
                        global_to_pnts[pnts[i]] = i;
                        levels[i] = RuleLocal::getLevel<effrule>(points.getIndex(pnts[i])[d]);
                        if (max_level < levels[i]) max_level = levels[i];
                    }
                } else {
                    for(int i=0; i<nump; i++){
                        vals.getStrip(i)[0] = values.getValues(pnts[i])[output];
                        global_to_pnts[pnts[i]] = i;
                        levels[i] = RuleLocal::getLevel<effrule>(points.getIndex(pnts[i])[d]);
                        if (max_level < levels[i]) max_level = levels[i];
                    }
                }

                if (max_1D_parents == 1) {
                    for(int l=1; l<=max_level; l++){
                        for(int i=0; i<nump; i++){
                            if (levels[i] == l){
                                double x = RuleLocal::getNode<effrule>(points.getIndex(pnts[i])[d]);
                                double *valsi = vals.getStrip(i);

                                int branch = dagUp.getStrip(pnts[i])[d];
                                while(branch != -1) {
                                    const int *branch_point = points.getIndex(branch);
                                    double basis_value = RuleLocal::evalRaw<effrule>(order, branch_point[d], x);
                                    const double *branch_vals = vals.getStrip(global_to_pnts[branch]);
                                    for(int k=0; k<active_outputs; k++) valsi[k] -= basis_value * branch_vals[k];
                                    branch = dagUp.getStrip(branch)[d];
                                }
                            }
                        }
                    }
                } else {
                    for(int l=1; l<=max_level; l++){
                        for(int i=0; i<nump; i++){
                            if (levels[i] == l){
                                double x = RuleLocal::getNode<effrule>(points.getIndex(pnts[i])[d]);
                                double *valsi = vals.getStrip(i);

                                int current = 0;
                                monkey_count[0] = d * max_1D_parents;
                                monkey_tail[0] = pnts[i]; // uses the global indexes
                                std::fill_n(used.begin(), nump, false);

                                while(monkey_count[0] < (d+1) * max_1D_parents){
                                    if (monkey_count[current] < (d+1) * max_1D_parents){
                                        int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                                        if ((branch == -1) || (used[global_to_pnts[branch]])){
                                            monkey_count[current]++;
                                        }else{
                                            const int *branch_point = points.getIndex(branch);
                                            double basis_value = RuleLocal::evalRaw<effrule>(order, branch_point[d], x);
                                            const double *branch_vals = vals.getStrip(global_to_pnts[branch]);
                                            for(int k=0; k<active_outputs; k++) valsi[k] -= basis_value * branch_vals[k];

                                            used[global_to_pnts[branch]] = true;
                                            monkey_count[++current] = d * max_1D_parents;
                                            monkey_tail[current] = branch;
                                        }
                                    }else{
                                        monkey_count[--current]++;
                                    }
                                }
                            }
                        }
                    }
                }

                // at this point, vals contains the one directional surpluses
                for(int i=0; i<nump; i++){
                    const double *s = surpluses.getStrip(pnts[i]);
                    const double *c = scale.getStrip(pnts[i]);
                    const double *v = vals.getStrip(i);
                    bool small = true;
                    if (output == -1){
                        for(int k=0; k<num_outputs; k++){
                            small = small && (((c[k] * std::abs(s[k]) / norm[k]) <= tolerance) || ((c[k] * std::abs(v[k]) / norm[k]) <= tolerance));
                        }
                    }else{
                        small = ((c[0] * std::abs(s[output]) / norm[output]) <= tolerance) || ((c[0] * std::abs(v[0]) / norm[output]) <= tolerance);
                    }
                    pmap.getStrip(pnts[i])[d] = (small) ? 0 : 1;;
                }
            }
        }
    }
    return pmap;
}
template<RuleLocal::erule effrule>
MultiIndexSet GridLocalPolynomial::getRefinementCanidates(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits, const double *scale_correction) const{
    Data2D<int> pmap = buildUpdateMap<effrule>(tolerance, criteria, output, scale_correction);

    bool useParents = (criteria == refine_fds) || (criteria == refine_parents_first);

    Data2D<int> refined(num_dimensions, 0);

    int num_points = points.getNumIndexes();

    #ifdef _OPENMP
    #pragma omp parallel
    {
        Data2D<int> lrefined(num_dimensions, 0);

        if (level_limits.empty()){
            #pragma omp for
            for(int i=0; i<num_points; i++){
                const int *map = pmap.getStrip(i);
                for(int j=0; j<num_dimensions; j++){
                    if (map[j] == 1){ // if this dimension needs to be refined
                        if (!(useParents && addParent<effrule>(points.getIndex(i), j, points, lrefined))){
                            addChild<effrule>(points.getIndex(i), j, points, lrefined);
                        }
                    }
                }
            }
        }else{
            #pragma omp for
            for(int i=0; i<num_points; i++){
                const int *map = pmap.getStrip(i);
                for(int j=0; j<num_dimensions; j++){
                    if (map[j] == 1){ // if this dimension needs to be refined
                        if (!(useParents && addParent<effrule>(points.getIndex(i), j, points, lrefined))){
                            addChildLimited<effrule>(points.getIndex(i), j, points, level_limits, lrefined);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            refined.append(lrefined);
        }
    }
    #else
    if (level_limits.empty()){
        for(int i=0; i<num_points; i++){
            const int *map = pmap.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (map[j] == 1){ // if this dimension needs to be refined
                    if (!(useParents && addParent<effrule>(points.getIndex(i), j, points, refined))){
                        addChild<effrule>(points.getIndex(i), j, points, refined);
                    }
                }
            }
        }
    }else{
        for(int i=0; i<num_points; i++){
            const int *map = pmap.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (map[j] == 1){ // if this dimension needs to be refined
                    if (!(useParents && addParent<effrule>(points.getIndex(i), j, points, refined))){
                        addChildLimited<effrule>(points.getIndex(i), j, points, level_limits, refined);
                    }
                }
            }
        }
    }
    #endif

    MultiIndexSet result(refined);
    if (criteria == refine_stable)
        HierarchyManipulations::completeToLower<effrule>(points, result);

    return result;
}

template<RuleLocal::erule effrule>
bool GridLocalPolynomial::addParent(const int point[], int direction, const MultiIndexSet &exclude, Data2D<int> &destination) const{
    std::vector<int> dad(point, point + num_dimensions);
    bool added = false;
    dad[direction] = RuleLocal::getParent<effrule>(point[direction]);
    if ((dad[direction] != -1) && exclude.missing(dad)){
        destination.appendStrip(dad);
        added = true;
    }
    dad[direction] = RuleLocal::getStepParent<effrule>(point[direction]);
    if ((dad[direction] != -1) && exclude.missing(dad)){
        destination.appendStrip(dad);
        added = true;
    }
    return added;
}
template<RuleLocal::erule effrule>
void GridLocalPolynomial::addChild(const int point[], int direction, const MultiIndexSet &exclude, Data2D<int> &destination) const{
    std::vector<int> kid(point, point + num_dimensions);
    int max_1d_kids = RuleLocal::getMaxNumKids<effrule>();
    for(int i=0; i<max_1d_kids; i++){
        kid[direction] = RuleLocal::getKid<effrule>(point[direction], i);
        if ((kid[direction] != -1) && exclude.missing(kid)){
            destination.appendStrip(kid);
        }
    }
}
template<RuleLocal::erule effrule>
void GridLocalPolynomial::addChildLimited(const int point[], int direction, const MultiIndexSet &exclude, const std::vector<int> &level_limits, Data2D<int> &destination) const{
    std::vector<int> kid(point, point + num_dimensions);
    int max_1d_kids = RuleLocal::getMaxNumKids<effrule>();
    for(int i=0; i<max_1d_kids; i++){
        kid[direction] = RuleLocal::getKid<effrule>(point[direction], i);
        if ((kid[direction] != -1)
            && ((level_limits[direction] == -1) || (RuleLocal::getLevel<effrule>(kid[direction]) <= level_limits[direction]))
            && exclude.missing(kid)){
            destination.appendStrip(kid);
        }
    }
}

void GridLocalPolynomial::clearRefinement(){ needed = MultiIndexSet(); }
const double* GridLocalPolynomial::getSurpluses() const{
    return surpluses.data();
}
const int* GridLocalPolynomial::getNeededIndexes() const{
    return (needed.empty()) ? 0 : needed.getIndex(0);
}
std::vector<double> GridLocalPolynomial::getSupport() const{
    MultiIndexSet const &work = (points.empty()) ? needed : points;

    std::vector<double> support(Utils::size_mult(work.getNumIndexes(), work.getNumDimensions()));

    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            std::transform(work.begin(), work.end(), support.begin(),
                           [&](int p)->double{ return RuleLocal::getSupport<RuleLocal::erule::pwc>(p); });
            break;
        case RuleLocal::erule::localp:
            std::transform(work.begin(), work.end(), support.begin(),
                           [&](int p)->double{ return RuleLocal::getSupport<RuleLocal::erule::localp>(p); });
            break;
        case RuleLocal::erule::semilocalp:
            std::transform(work.begin(), work.end(), support.begin(),
                           [&](int p)->double{ return RuleLocal::getSupport<RuleLocal::erule::semilocalp>(p); });
            break;
        case RuleLocal::erule::localp0:
            std::transform(work.begin(), work.end(), support.begin(),
                           [&](int p)->double{ return RuleLocal::getSupport<RuleLocal::erule::localp0>(p); });
            break;
        default: // case RuleLocal::erule::localpb:
            std::transform(work.begin(), work.end(), support.begin(),
                           [&](int p)->double{ return RuleLocal::getSupport<RuleLocal::erule::localpb>(p); });
            break;
    };
    return support;
}

void GridLocalPolynomial::setSurplusRefinement(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits, const double *scale_correction){
    clearRefinement();

    switch(effective_rule) {
        case RuleLocal::erule::pwc:
            needed = getRefinementCanidates<RuleLocal::erule::pwc>(tolerance, criteria, output, level_limits, scale_correction);
            break;
        case RuleLocal::erule::localp:
            needed = getRefinementCanidates<RuleLocal::erule::localp>(tolerance, criteria, output, level_limits, scale_correction);
            break;
        case RuleLocal::erule::semilocalp:
            needed = getRefinementCanidates<RuleLocal::erule::semilocalp>(tolerance, criteria, output, level_limits, scale_correction);
            break;
        case RuleLocal::erule::localp0:
            needed = getRefinementCanidates<RuleLocal::erule::localp0>(tolerance, criteria, output, level_limits, scale_correction);
            break;
        default: // case RuleLocal::erule::localpb:
            needed = getRefinementCanidates<RuleLocal::erule::localpb>(tolerance, criteria, output, level_limits, scale_correction);
            break;
    };

}
std::vector<double> GridLocalPolynomial::getScaledCoefficients(int output, const double *scale_correction){
    int num_points = points.getNumIndexes();
    std::vector<double> norm = getNormalization();
    int active_outputs = (output == -1) ? num_outputs : 1;
    Utils::Wrapper2D<double const> scale(active_outputs, scale_correction);

    std::vector<double> rescaled(num_points);

    for(int i=0; i<num_points; i++){
        const double *s = surpluses.getStrip(i);
        const double *c = scale.getStrip(i);
        if (output == -1){
            rescaled[i] = 0.0;
            for(int k=0; k<num_outputs; k++) rescaled[i] = std::max(rescaled[i], c[k] * std::abs(s[k]) / norm[k]);
        }else{
            rescaled[i] = c[0] * std::abs(s[output]) / norm[output];
        }
    }
    return rescaled;
}
int GridLocalPolynomial::removePointsByHierarchicalCoefficient(double tolerance, int output, const double *scale_correction){
    clearRefinement();
    int num_points = points.getNumIndexes();

    int active_outputs = (output == -1) ? num_outputs : 1;
    std::vector<double> rescaled = getScaledCoefficients(output, (scale_correction == nullptr) ?
                                        std::vector<double>(num_points * active_outputs, 1.0).data() : scale_correction);

    std::vector<bool> pmap(num_points); // point map, set to true if the point is to be kept, false otherwise
    for(int i=0; i<num_points; i++) pmap[i] = (rescaled[i] > tolerance);

    return removeMappedPoints(pmap);
}

void GridLocalPolynomial::removePointsByHierarchicalCoefficient(int new_num_points, int output, const double *scale_correction){
    clearRefinement();
    int num_points = points.getNumIndexes();

    int active_outputs = (output == -1) ? num_outputs : 1;
    std::vector<double> rescaled = getScaledCoefficients(output, (scale_correction == nullptr) ?
                                        std::vector<double>(num_points * active_outputs, 1.0).data() : scale_correction);

    std::vector<std::pair<double, int>> ordered(num_points);
    for(int i=0; i<num_points; i++)
        ordered[i] = std::make_pair(rescaled[i], i);

    std::sort(ordered.begin(), ordered.end(),
              [](std::pair<double, int> const& a, std::pair<double, int> const& b) ->bool{ return (a.first > b.first); });

    std::vector<bool> pmap(num_points, false);
    for(int i=0; i<new_num_points; i++) pmap[ordered[i].second] = true;

    removeMappedPoints(pmap);
}

int GridLocalPolynomial::removeMappedPoints(std::vector<bool> const &pmap){
    int num_points = points.getNumIndexes();
    int num_kept = 0; for(int i=0; i<num_points; i++) num_kept += (pmap[i]) ? 1 : 0;

    if (num_kept == num_points) return num_points; // trivial case, remove nothing
    clearGpuBasisHierarchy();
    clearGpuSurpluses();

    // save a copy of the points and the values
    Data2D<int> point_kept(num_dimensions, num_kept);

    StorageSet values_kept(num_outputs, num_kept, std::vector<double>(Utils::size_mult(num_kept, num_outputs)));
    Data2D<double> surpluses_kept(num_outputs, num_kept);

    for(int i=0, kept = 0; i<num_points; i++){
        if (pmap[i]){
            std::copy_n(points.getIndex(i), num_dimensions, point_kept.getStrip(kept));
            std::copy_n(values.getValues(i), num_outputs, values_kept.getValues(kept));
            std::copy_n(surpluses.getStrip(i), num_outputs, surpluses_kept.getStrip(kept));
            kept++;
        }
    }

    // reset the grid
    needed = MultiIndexSet();

    if (num_kept == 0){ // trivial case, remove all
        points = MultiIndexSet();
        values = StorageSet();
        parents = Data2D<int>();
        surpluses.clear();
        return 0;
    }

    points = MultiIndexSet(point_kept);

    values = std::move(values_kept);

    surpluses = std::move(surpluses_kept);

    buildTree();

    return points.getNumIndexes();
}

void GridLocalPolynomial::setHierarchicalCoefficients(const double c[]){
    clearGpuSurpluses();
    if (points.empty()){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        clearRefinement();
    }
    surpluses = Data2D<double>(num_outputs, points.getNumIndexes(), std::vector<double>(c, c + Utils::size_mult(num_outputs, points.getNumIndexes())));

    std::vector<double> x(Utils::size_mult(num_dimensions, points.getNumIndexes()));
    std::vector<double> y(Utils::size_mult(num_outputs,    points.getNumIndexes()));

    getPoints(x.data());
    evaluateBatch(x.data(), points.getNumIndexes(), y.data());

    values = StorageSet(num_outputs, points.getNumIndexes(), std::move(y));
}

#ifdef Tasmanian_ENABLE_GPU
void GridLocalPolynomial::updateAccelerationData(AccelerationContext::ChangeType change) const{
    switch(change){
        case AccelerationContext::change_gpu_device:
            gpu_cache.reset();
            gpu_cachef.reset();
            break;
        case AccelerationContext::change_sparse_dense:
            if (acceleration->algorithm_select == AccelerationContext::algorithm_dense){
                // if forcing the dense algorithm then clear the hierarchy cache
                // note: the sparse algorithm uses both the hierarchy and the basis cache used in the dense mode
                if (gpu_cache) gpu_cache->clearHierarchy();
                if (gpu_cachef) gpu_cachef->clearHierarchy();
            }
        default:
            break;
    }
}
#else
void GridLocalPolynomial::updateAccelerationData(AccelerationContext::ChangeType) const{}
#endif

}

#endif
