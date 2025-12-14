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

#ifndef __TASMANIAN_SPARSE_GRID_GLOBAL_NESTED_CPP
#define __TASMANIAN_SPARSE_GRID_GLOBAL_NESTED_CPP

#include "tsgGridSequence.hpp"
#include "tsgTPLWrappers.hpp"

namespace TasGrid{

template<bool iomode> void GridSequence::write(std::ostream &os) const{
    if (iomode == mode_ascii){ os << std::scientific; os.precision(17); }
    IO::writeNumbers<iomode, IO::pad_rspace>(os, num_dimensions, num_outputs);
    IO::writeRule<iomode>(rule, os);

    IO::writeFlag<iomode, IO::pad_auto>(!points.empty(), os);
    if (!points.empty()) points.write<iomode>(os);
    IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
    if (!needed.empty()) needed.write<iomode>(os);

    IO::writeFlag<iomode, IO::pad_auto>(!surpluses.empty(), os);
    if (!surpluses.empty()) surpluses.writeVector<iomode, IO::pad_line>(os);

    if (num_outputs > 0) values.write<iomode>(os);
}

template void GridSequence::write<mode_ascii>(std::ostream &) const;
template void GridSequence::write<mode_binary>(std::ostream &) const;

void GridSequence::clearRefinement(){ needed = MultiIndexSet(); }

inline MultiIndexSet makeSequenceSet(int cnum_dimensions, int depth, TypeDepth type, TypeOneDRule crule,
                                     const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    return (OneDimensionalMeta::isExactQuadrature(type)) ?
        MultiIndexManipulations::selectTensors((size_t) cnum_dimensions, depth, type,
                                               [&](int i) -> int{ return OneDimensionalMeta::getQExact(i, crule); },
                                               anisotropic_weights, level_limits) :
        MultiIndexManipulations::selectTensors((size_t) cnum_dimensions, depth, type,
                                               [&](int i) -> int{ return i; }, anisotropic_weights, level_limits);
}

GridSequence::GridSequence(AccelerationContext const *acc, int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, TypeOneDRule crule,
                            const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits)
    : BaseCanonicalGrid(acc, cnum_dimensions, cnum_outputs, MultiIndexSet(),
                        makeSequenceSet(cnum_dimensions, depth, type, crule, anisotropic_weights, level_limits),
                        StorageSet()),
      rule(crule){
    values.resize(num_outputs, needed.getNumIndexes());
    prepareSequence(0);
}
GridSequence::GridSequence(AccelerationContext const *acc, int cnum_dimensions, int depth, TypeDepth type, TypeOneDRule crule,
                            const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits)
    : BaseCanonicalGrid(acc, cnum_dimensions, 0, makeSequenceSet(cnum_dimensions, depth, type, crule, anisotropic_weights, level_limits),
                        MultiIndexSet(), StorageSet()),
      rule(crule){
    prepareSequence(0);
}
GridSequence::GridSequence(AccelerationContext const *acc, MultiIndexSet &&pset, int cnum_outputs, TypeOneDRule crule)
    : BaseCanonicalGrid(acc, static_cast<int>(pset.getNumDimensions()), cnum_outputs,
                        (cnum_outputs == 0) ? std::move(pset) : MultiIndexSet(),
                        (cnum_outputs == 0) ? MultiIndexSet() : std::move(pset),
                        StorageSet()),
    rule(crule)
{
    if (num_outputs > 0) values.resize(num_outputs, needed.getNumIndexes());
    prepareSequence(0);
}

GridSequence::GridSequence(AccelerationContext const *acc, GridSequence const *seq, int ibegin, int iend) :
    BaseCanonicalGrid(acc, *seq, ibegin, iend),
    rule(seq->rule),
    surpluses((num_outputs == seq->num_outputs) ? seq->surpluses : seq->surpluses.splitData(ibegin, iend)),
    nodes(seq->nodes),
    coeff(seq->coeff),
    max_levels(seq->max_levels){

    if (seq->dynamic_values){
        dynamic_values = std::make_unique<SimpleConstructData>(*seq->dynamic_values);
        if (num_outputs != seq->num_outputs) dynamic_values->restrictData(ibegin, iend);
    }
}

void GridSequence::updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    clearRefinement();

    MultiIndexSet pset = makeSequenceSet(num_dimensions, depth, type, rule, anisotropic_weights, level_limits);

    if ((num_outputs == 0) || (points.empty())){
        if (num_outputs == 0){
            points = std::move(pset);
            needed = MultiIndexSet();
        }else{
            points = MultiIndexSet();
            needed = std::move(pset);
            values.resize(num_outputs, needed.getNumIndexes());
        }
        nodes = std::vector<double>();
        coeff = std::vector<double>();
        surpluses = Data2D<double>();
        prepareSequence(0);
    }else{
        pset += points;
        needed = pset - points;

        if (!needed.empty()) prepareSequence(0);
    }
}

void GridSequence::getLoadedPoints(double *x) const{
    MultiIndexManipulations::indexesToNodes(points, *this, x);
}
void GridSequence::getNeededPoints(double *x) const{
    MultiIndexManipulations::indexesToNodes(needed, *this, x);
}
void GridSequence::getPoints(double *x) const{
    if (points.empty()){ getNeededPoints(x); }else{ getLoadedPoints(x); }
}

void GridSequence::getQuadratureWeights(double *weights) const{
    const MultiIndexSet& work = (points.empty()) ? needed : points;
    std::vector<double> integ = cacheBasisIntegrals();
    int n = work.getNumIndexes();
    for(int i=0; i<n; i++){
        const int* p = work.getIndex(i);
        weights[i] = integ[p[0]];
        for(int j=1; j<num_dimensions; j++){
            weights[i] *= integ[p[j]];
        }
    }

    applyTransformationTransposed<0>(weights);
}

void GridSequence::getInterpolationWeights(const double x[], double *weights) const{
    std::vector<std::vector<double>> cache = cacheBasisValues<double>(x);
    const MultiIndexSet& work = (points.empty()) ? needed : points;
    int n = work.getNumIndexes();
    weights[0] = 1.0;
    for(int i=1; i<n; i++){
        const int* p = work.getIndex(i);
        weights[i] = cache[0][p[0]];
        for(int j=1; j<num_dimensions; j++){
            weights[i] *= cache[j][p[j]];
        }
    }
    applyTransformationTransposed<0>(weights);
}

void GridSequence::getDifferentiationWeights(const double x[], double weights[]) const {
    std::vector<std::vector<double>> value_cache = cacheBasisValues<double>(x);
    std::vector<std::vector<double>> derivative_cache = cacheBasisDerivatives<double>(x);
    std::vector<double> diff_values(num_dimensions);
    const MultiIndexSet& work = (points.empty()) ? needed : points;
    int n = work.getNumIndexes();
    std::fill_n(weights, n * num_dimensions, 0.0);
    for(int i=0; i<n; i++) {
        const int* p = work.getIndex(i);
        diff_values[0] = derivative_cache[0][p[0]];
        for(int j=1; j<num_dimensions; j++) diff_values[j] = value_cache[0][p[0]];

        for(int k=1; k<num_dimensions; k++) {
            for(int j=0; j<k; j++) diff_values[j] *= value_cache[k][p[k]];
            diff_values[k] *= derivative_cache[k][p[k]];
            for(int j=k+1; j<num_dimensions; j++) diff_values[j] *= value_cache[k][p[k]];
        }

        for(int j=0; j<num_dimensions; j++) weights[i * num_dimensions + j] +=  diff_values[j];
    }
    applyTransformationTransposed<1>(weights);
}

void GridSequence::loadNeededValues(const double *vals){
    clearGpuSurpluses(); // changing values and surpluses, clear the cache
    if (needed.empty()){ // overwrite the existing values
        values.setValues(vals);
    }else{
        clearGpuNodes(); // the points and needed will change, clear the cache
        if (points.empty()){ // initial grid, just relabel needed as points (loaded)
            values.setValues(vals);
            points = std::move(needed);
            needed = MultiIndexSet();
        }else{ // merge needed and points
            values.addValues(points, needed, vals);
            points += needed;
            needed = MultiIndexSet();
            prepareSequence(0);
        }
    }
    recomputeSurpluses();
}
void GridSequence::mergeRefinement(){
    if (needed.empty()) return; // nothing to do
    clearGpuSurpluses(); // clear the surpluses (all values have cleared)
    int num_all_points = getNumLoaded() + getNumNeeded();
    size_t num_vals = ((size_t) num_all_points) * ((size_t) num_outputs);
    values.setValues(std::vector<double>(num_vals, 0.0));
    if (points.empty()){ // relabel needed as points (loaded)
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        clearGpuNodes(); // the points will change, clear cache
        points += needed;
        needed = MultiIndexSet();
        prepareSequence(0);
    }
    surpluses = Data2D<double>(num_outputs, num_all_points);
}

void GridSequence::beginConstruction(){
    dynamic_values = std::make_unique<SimpleConstructData>();
    if (points.empty()){
        dynamic_values->initial_points = std::move(needed);
        needed = MultiIndexSet();
    }
}
void GridSequence::writeConstructionData(std::ostream &os, bool iomode) const{
    if (iomode == mode_ascii) dynamic_values->write<mode_ascii>(os); else dynamic_values->write<mode_binary>(os);
}
void GridSequence::readConstructionData(std::istream &is, bool iomode){
    if (iomode == mode_ascii)
        dynamic_values = std::make_unique<SimpleConstructData>(is, num_dimensions, num_outputs, IO::mode_ascii_type());
    else
        dynamic_values = std::make_unique<SimpleConstructData>(is, num_dimensions, num_outputs, IO::mode_binary_type());
}
std::vector<double> GridSequence::getCandidateConstructionPoints(TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    MultiIndexManipulations::ProperWeights weights((size_t) num_dimensions, type, anisotropic_weights);

    auto level_exact = [&](int l) -> int{ return l; };
    auto quad_exact = [&](int l) -> int{ return OneDimensionalMeta::getQExact(l, rule); };

    if (weights.contour == type_level){
        std::vector<std::vector<int>> cache;
        return getCandidateConstructionPoints([&](int const *t) -> double{
            // see the same named function in GridGlobal
            if (cache.empty()){
                if (OneDimensionalMeta::isExactQuadrature(type)){
                    cache = MultiIndexManipulations::generateLevelWeightsCache<int, type_level, true>(weights, quad_exact, (int) nodes.size());
                }else{
                    cache = MultiIndexManipulations::generateLevelWeightsCache<int, type_level, true>(weights, level_exact, (int) nodes.size());
                }
            }

            int w = 0;
            for(int j=0; j<num_dimensions; j++) w += cache[j][t[j]];
            return (double) w;
        }, level_limits);
    }else if (weights.contour == type_curved){
        std::vector<std::vector<double>> cache;
        return getCandidateConstructionPoints([&](int const *t) -> double{
            // see the same named function in GridGlobal
            if (cache.empty()){
                if (OneDimensionalMeta::isExactQuadrature(type)){
                    cache = MultiIndexManipulations::generateLevelWeightsCache<double, type_curved, true>(weights, quad_exact, (int) nodes.size());
                }else{
                    cache = MultiIndexManipulations::generateLevelWeightsCache<double, type_curved, true>(weights, level_exact, (int) nodes.size());
                }
            }

            double w = 0.0;
            for(int j=0; j<num_dimensions; j++) w += cache[j][t[j]];
            return w;
        }, level_limits);
    }else{
        std::vector<std::vector<double>> cache;
        return getCandidateConstructionPoints([&](int const *t) -> double{
            // see the same named function in GridGlobal
            if (cache.empty()){
                if (OneDimensionalMeta::isExactQuadrature(type)){
                    cache = MultiIndexManipulations::generateLevelWeightsCache<double, type_hyperbolic, true>(weights, quad_exact, (int) nodes.size());
                }else{
                    cache = MultiIndexManipulations::generateLevelWeightsCache<double, type_hyperbolic, true>(weights, level_exact, (int) nodes.size());
                }
            }

            double w = 1.0;
            for(int j=0; j<num_dimensions; j++) w *= cache[j][t[j]];
            return w;
        }, level_limits);
    }
}
std::vector<double> GridSequence::getCandidateConstructionPoints(TypeDepth type, int output, const std::vector<int> &level_limits){
    std::vector<int> weights;
    if ((type == type_iptotal) || (type == type_ipcurved) || (type == type_qptotal) || (type == type_qpcurved)){
        int min_needed_points = ((type == type_ipcurved) || (type == type_qpcurved)) ? 4 * num_dimensions : 2 * num_dimensions;
        if (points.getNumIndexes() > min_needed_points) // if there are enough points to estimate coefficients
            estimateAnisotropicCoefficients(type, output, weights);
    }
    return getCandidateConstructionPoints(type, weights, level_limits);
}
std::vector<double> GridSequence::getCandidateConstructionPoints(std::function<double(const int *)> getTensorWeight, const std::vector<int> &level_limits){
    // get the new candidate points that will ensure lower completeness and are not included in the initial set
    MultiIndexSet new_points = (level_limits.empty()) ?
        MultiIndexManipulations::addExclusiveChildren<false>(points, dynamic_values->initial_points, level_limits) :
        MultiIndexManipulations::addExclusiveChildren<true>(points, dynamic_values->initial_points, level_limits);

    prepareSequence(std::max(new_points.getMaxIndex(), dynamic_values->initial_points.getMaxIndex()));

    std::forward_list<NodeData> weighted_points; // use the values as the weight
    for(int i=0; i<dynamic_values->initial_points.getNumIndexes(); i++){
        std::vector<int> p = dynamic_values->initial_points.copyIndex(i);
        weighted_points.push_front({p, {-1.0 / ((double) std::accumulate(p.begin(), p.end(), 0))}});
    }
    for(int i=0; i<new_points.getNumIndexes(); i++){
        std::vector<int> p = new_points.copyIndex(i);
        weighted_points.push_front({p, {getTensorWeight(p.data())}});
    }

    weighted_points.sort([&](const NodeData &a, const NodeData &b)->bool{ return (a.value[0] < b.value[0]); });

    return listToNodes(weighted_points, num_dimensions, *this);
}
std::vector<int> GridSequence::getMultiIndex(const double x[]){
    std::vector<int> p(num_dimensions);
    for(int j=0; j<num_dimensions; j++){
        int i = 0;
        while(std::abs(nodes[i] - x[j]) > Maths::num_tol){
            i++; // convert canonical node to index
            if (i == (int) nodes.size())
                prepareSequence(i);
        }
        p[j] = i;
    }
    return p;
}
void GridSequence::loadConstructedPoint(const double x[], const std::vector<double> &y){
    std::vector<int> p = getMultiIndex(x);
    std::vector<int> scratch(num_dimensions);

    if (MultiIndexManipulations::isLowerComplete(p, points, scratch)){
        std::vector<double> approx_value(num_outputs), surplus(num_outputs);;
        if (!points.empty()){
            evaluate(x, approx_value.data());
            std::transform(approx_value.begin(), approx_value.end(), y.begin(), surplus.begin(), [&](double e, double v)->double{ return v - e; });
        }
        expandGrid(p, y, surplus);
        dynamic_values->initial_points.removeIndex(p);

        loadConstructedPoints(); // batch operation, if the new point has unleashed a bunch of previously available ones
    }else{
        dynamic_values->data.push_front({p, y});
        dynamic_values->initial_points.removeIndex(p);
    }
}
void GridSequence::loadConstructedPoint(const double x[], int numx, const double y[]){
    Utils::Wrapper2D<const double> wrapx(num_dimensions, x);
    std::vector<std::vector<int>> pnts(numx);
    for(int i=0; i<numx; i++)
        pnts[i] = getMultiIndex(wrapx.getStrip(i));

    if (!dynamic_values->initial_points.empty()){
        Data2D<int> combined_pnts(num_dimensions, numx);
        for(int i=0; i<numx; i++)
            std::copy_n(pnts[i].begin(), num_dimensions, combined_pnts.getIStrip(i));
        dynamic_values->initial_points = dynamic_values->initial_points - combined_pnts;
    }

    Utils::Wrapper2D<const double> wrapy(num_outputs, y);
    for(int i=0; i<numx; i++)
        dynamic_values->data.push_front({std::move(pnts[i]), std::vector<double>(wrapy.getStrip(i), wrapy.getStrip(i) + num_outputs)});
    loadConstructedPoints();
}
void GridSequence::expandGrid(const std::vector<int> &point, const std::vector<double> &value, const std::vector<double> &surplus){
    if (points.empty()){ // only one point
        points = MultiIndexSet((size_t) num_dimensions, std::vector<int>(point));
        values = StorageSet(num_outputs, 1, std::vector<double>(value));
        surpluses = Data2D<double>(num_outputs, 1, std::vector<double>(value.begin(), value.end())); // the surplus of one point is the value itself
    }else{ // merge with existing points
        MultiIndexSet temp(num_dimensions, std::vector<int>(point));
        values.addValues(points, temp, value.data());

        points.addSortedIndexes(point);
        surpluses.appendStrip(points.getSlot(point), surplus);
    }
    prepareSequence(0); // update the directional max_levels, will not shrink the number of nodes
}
void GridSequence::loadConstructedPoints(){
    Data2D<int> candidates(num_dimensions, 0);
    for(auto &d : dynamic_values->data)
        candidates.appendStrip(d.point);
    auto new_points = MultiIndexManipulations::getLargestCompletion(points, MultiIndexSet(candidates));
    if (new_points.empty()) return;

    clearGpuNodes(); // the points will change, clear the cache
    clearGpuSurpluses();

    auto vals = dynamic_values->extractValues(new_points);
    if (points.empty()){
        points = std::move(new_points);
        values.setValues(std::move(vals));
    }else{
        values.addValues(points, new_points, vals.data());
        points += new_points;
    }
    prepareSequence(0); // update the directional max_levels, will not shrink the number of nodes
    recomputeSurpluses(); // costly, but the only option under the circumstances
}
void GridSequence::finishConstruction(){
    dynamic_values.reset();
}

void GridSequence::evaluate(const double x[], double y[]) const{
    std::vector<std::vector<double>> cache = cacheBasisValues<double>(x);

    std::fill(y, y + num_outputs, 0.0);

    int num_points = points.getNumIndexes();

    for(int i=0; i<num_points; i++){
        const int* p = points.getIndex(i);
        const double *s = surpluses.getStrip(i);
        double basis_value = cache[0][p[0]];
        for(int j=1; j<num_dimensions; j++){
            basis_value *= cache[j][p[j]];
        }

        for(int k=0; k<num_outputs; k++){
            y[k] += basis_value * s[k];
        }
    }
}
void GridSequence::evaluateBatch(const double x[], int num_x, double y[]) const{
    switch(acceleration->mode){
        case accel_gpu_magma:
        case accel_gpu_cuda: {
            acceleration->setDevice();
            GpuVector<double> gpu_x(acceleration, num_dimensions, num_x, x), gpu_result(acceleration, num_outputs, num_x);
            evaluateBatchGPU(gpu_x.data(), num_x, gpu_result.data());
            gpu_result.unload(acceleration, y);
            break;
        }
        case accel_gpu_cublas: {
            acceleration->setDevice();
            loadGpuSurpluses<double>();
            Data2D<double> hweights(points.getNumIndexes(), num_x);
            evaluateHierarchicalFunctions(x, num_x, hweights.data());
            TasGpu::denseMultiplyMixed(acceleration, num_outputs, num_x, points.getNumIndexes(),
                                       1.0, gpu_cache->surpluses, hweights.data(), 0.0, y);
            break;
        }
        case accel_cpu_blas: {
            int num_points = points.getNumIndexes();
            Data2D<double> weights(num_points, num_x);
            if (num_x > 1)
                evaluateHierarchicalFunctions(x, num_x, weights.data());
            else // workaround small OpenMP penalty
                evalHierarchicalFunctions(x, weights.data());
            TasBLAS::denseMultiply(num_outputs, num_x, num_points, 1.0, surpluses.data(), weights.data(), 0.0, y);
            break;
        }
        default: {
            Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
            Utils::Wrapper2D<double> ywrap(num_outputs, y);
            #pragma omp parallel for
            for(int i=0; i<num_x; i++)
                evaluate(xwrap.getStrip(i), ywrap.getStrip(i));
            break;
        }
    }
}

template<typename T> void GridSequence::evaluateBatchGPUtempl(const T gpu_x[], int cpu_num_x, T gpu_y[]) const{
    loadGpuSurpluses<T>();

    GpuVector<T> gpu_basis(acceleration, points.getNumIndexes(), cpu_num_x);
    evaluateHierarchicalFunctionsGPU(gpu_x, cpu_num_x, gpu_basis.data());
    TasGpu::denseMultiply(acceleration, num_outputs, cpu_num_x, points.getNumIndexes(),
                          1.0, getGpuCache<T>()->surpluses, gpu_basis, 0.0, gpu_y);
}
void GridSequence::evaluateBatchGPU(const double gpu_x[], int cpu_num_x, double gpu_y[]) const{
    evaluateBatchGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
void GridSequence::evaluateHierarchicalFunctionsGPU(const double gpu_x[], int num_x, double gpu_y[]) const{
    loadGpuNodes<double>();
    TasGpu::devalseq(acceleration, num_dimensions, num_x, max_levels, gpu_x, gpu_cache->num_nodes, gpu_cache->points, gpu_cache->nodes, gpu_cache->coeff, gpu_y);
}
void GridSequence::evaluateBatchGPU(const float gpu_x[], int cpu_num_x, float gpu_y[]) const{
    evaluateBatchGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
void GridSequence::evaluateHierarchicalFunctionsGPU(const float gpu_x[], int num_x, float gpu_y[]) const{
    loadGpuNodes<float>();
    TasGpu::devalseq(acceleration, num_dimensions, num_x, max_levels, gpu_x, gpu_cachef->num_nodes, gpu_cachef->points, gpu_cachef->nodes, gpu_cachef->coeff, gpu_y);
}
void GridSequence::clearGpuNodes() const{
    if (gpu_cache) gpu_cache->clearNodes();
    if (gpu_cachef) gpu_cachef->clearNodes();
}
void GridSequence::clearGpuSurpluses() const{
    if (gpu_cache) gpu_cache->surpluses.clear();
    if (gpu_cachef) gpu_cachef->surpluses.clear();
}

void GridSequence::integrate(double q[], double *conformal_correction) const{
    int num_points = points.getNumIndexes();
    std::fill(q, q + num_outputs, 0.0);

    // for sequence grids, quadrature weights are expensive,
    // if using simple integration use the basis integral + surpluses, which is fast
    // if using conformal map, then we have to compute the expensive weights
    if (conformal_correction == 0){
        std::vector<double> integ = cacheBasisIntegrals();
        for(int i=0; i<num_points; i++){
            const int* p = points.getIndex(i);
            double w = integ[p[0]];
            const double *s = surpluses.getStrip(i);
            for(int j=1; j<num_dimensions; j++){
                w *= integ[p[j]];
            }
            for(int k=0; k<num_outputs; k++){
                q[k] += w * s[k];
            }
        }
    }else{
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

void GridSequence::differentiate(const double x[], double jacobian[]) const {
    // Based on the logic in the TasGrid::GridSequence::evaluate() and TasGrid::GridSequence::getDifferentiationWeights() functions.
    std::vector<std::vector<double>> value_cache = cacheBasisValues<double>(x);
    std::vector<std::vector<double>> derivative_cache = cacheBasisDerivatives<double>(x);
    std::vector<double> diff_values(num_dimensions);
    std::fill_n(jacobian, num_outputs * num_dimensions, 0.0);
    int n = points.getNumIndexes();
    for(int i=0; i<n; i++) {
        const int* p = points.getIndex(i);
        const double *s = surpluses.getStrip(i);

        diff_values[0] = derivative_cache[0][p[0]];
        for(int j=1; j<num_dimensions; j++) diff_values[j] = value_cache[0][p[0]];

        for(int k=1; k<num_dimensions; k++) {
            for(int j=0; j<k; j++) diff_values[j] *= value_cache[k][p[k]];
            diff_values[k] *= derivative_cache[k][p[k]];
            for(int j=k+1; j<num_dimensions; j++) diff_values[j] *= value_cache[k][p[k]];
        }
        for(int k=0; k<num_outputs; k++)
            for(int j=0; j<num_dimensions; j++)
                jacobian[k * num_dimensions + j] += s[k] * diff_values[j];
    }
}

void GridSequence::evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const{
    int num_points = (points.empty()) ? needed.getNumIndexes() : points.getNumIndexes();
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_points, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++)
        evalHierarchicalFunctions(xwrap.getStrip(i), ywrap.getStrip(i));
}
void GridSequence::evalHierarchicalFunctions(const double x[], double fvalues[]) const{
    const MultiIndexSet& work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();

    std::vector<std::vector<double>> cache = cacheBasisValues<double>(x);

    for(int i=0; i<num_points; i++){
        const int* p = work.getIndex(i);
        fvalues[i] = cache[0][p[0]];
        for(int j=1; j<num_dimensions; j++){
            fvalues[i] *= cache[j][p[j]];
        }
    }
}
void GridSequence::setHierarchicalCoefficients(const double c[]){
    clearGpuSurpluses(); // points have not changed, just clear surpluses
    if (!points.empty()){
        clearRefinement();
    }else{
        points = std::move(needed);
        needed = MultiIndexSet();
    }
    auto num_points = points.getNumIndexes();
    surpluses = Data2D<double>(num_outputs, num_points, std::vector<double>(c, c + Utils::size_mult(num_outputs, num_points)));

    std::vector<double> x(Utils::size_mult(num_dimensions, num_points));
    std::vector<double> y(Utils::size_mult(num_outputs,    num_points));

    getPoints(x.data());
    evaluateBatch(x.data(), points.getNumIndexes(), y.data()); // speed this up later

    values = StorageSet(num_outputs, num_points, std::move(y));
}

void GridSequence::integrateHierarchicalFunctions(double integrals[]) const{
    const MultiIndexSet& work = (points.empty()) ? needed : points;
    int const num_points = work.getNumIndexes();
    std::vector<double> integ = cacheBasisIntegrals();
    for(int i=0; i<num_points; i++){
        const int* p = work.getIndex(i);
        double w = integ[p[0]];
        for(int j=1; j<num_dimensions; j++){
            w *= integ[p[j]];
        }
        integrals[i] = w;
    }
}

void GridSequence::estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const{
    double tol = 1000 * Maths::num_tol;
    int num_points = points.getNumIndexes();
    std::vector<double> max_surp(num_points);

    if (output == -1){
        std::vector<double> nrm(num_outputs, 0.0);
        for(int i=0; i<num_points; i++){
            const double *val = values.getValues(i);
            int k=0;
            for(auto &n : nrm){
                double v = std::abs(val[k++]);
                if (n < v) n = v;
            }
        }
        #pragma omp parallel for
        for(int i=0; i<num_points; i++){
            const double *s = surpluses.getStrip(i);
            double smax = 0.0;
            for(int k=0; k<num_outputs; k++){
                double v = std::abs(s[k]) / nrm[k];
                if (smax < v) smax = v;
            }
            max_surp[i] = smax;
        }
    }else{
        int i = 0;
        for(auto &m : max_surp) m = surpluses.getStrip(i++)[output];
    }

    weights = MultiIndexManipulations::inferAnisotropicWeights(acceleration, rule, type, points, max_surp, tol);
}

void GridSequence::setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits){
    clearRefinement();

    std::vector<int> weights;
    estimateAnisotropicCoefficients(type, output, weights);

    int level = 0;
    do{
        updateGrid(++level, type, weights, level_limits);
    }while(getNumNeeded() < min_growth);
}
void GridSequence::setSurplusRefinement(double tolerance, int output, const std::vector<int> &level_limits){
    clearRefinement();

    int num_points = points.getNumIndexes();
    std::vector<bool> flagged(num_points);

    std::vector<double> norm(num_outputs, 0.0);
    for(int i=0; i<num_points; i++){
        const double *val = values.getValues(i);
        for(int k=0; k<num_outputs; k++){
            double v = std::abs(val[k]);
            if (norm[k] < v) norm[k] = v;
        }
    }

    if (output == -1){
        for(int i=0; i<num_points; i++){
            const double *s = surpluses.getStrip(i);
            double smax = std::abs(s[0]) / norm[0];
            for(int k=1; k<num_outputs; k++){
                double v = std::abs(s[k]) / norm[k];
                if (smax < v) smax = v;
            }
            flagged[i] = (smax > tolerance);
        }
    }else{
        for(int i=0; i<num_points; i++){
            flagged[i] = ((std::abs(surpluses.getStrip(i)[output]) / norm[output]) > tolerance);
        }
    }

    MultiIndexSet kids = MultiIndexManipulations::selectFlaggedChildren(points, flagged, level_limits);
    if (kids.getNumIndexes() > 0){
        kids += points;
        MultiIndexManipulations::completeSetToLower(kids);

        needed = kids - points;
        if (!needed.empty()) prepareSequence(0);
    }
}

std::vector<int> GridSequence::getPolynomialSpace(bool interpolation) const{
    if (interpolation){
        return (points.empty()) ? std::vector<int>(needed.begin(), needed.end()) : std::vector<int>(points.begin(), points.end()); // copy
    }else{
        MultiIndexSet polynomial_set = MultiIndexManipulations::createPolynomialSpace(
            (points.empty()) ? needed : points,
            [&](int l) -> int{ return OneDimensionalMeta::getQExact(l, rule); });
        return polynomial_set.release();
    }
}
const double* GridSequence::getSurpluses() const{
    return surpluses.data();
}

void GridSequence::prepareSequence(int num_external){
    int mp = 0, mn = 0, max_level;
    if (needed.empty()){ // points must be non-empty
        if (points.empty()){
            max_levels.resize(num_dimensions, 0);
        }else{
            max_levels = MultiIndexManipulations::getMaxIndexes(points);
            mp = *std::max_element(max_levels.begin(), max_levels.end());
        }
    }else if (points.empty()){ // only needed, no points (right after creation)
        max_levels = MultiIndexManipulations::getMaxIndexes(needed);
        mn = *std::max_element(max_levels.begin(), max_levels.end());
    }else{ // both points and needed are set
        max_levels = MultiIndexManipulations::getMaxIndexes(points);
        mp = *std::max_element(max_levels.begin(), max_levels.end());
        mn = needed.getMaxIndex();
    }
    max_level = (mp > mn) ? mp : mn;
    if (max_level < num_external) max_level = num_external;
    max_level++;

    if ((size_t) max_level > nodes.size()){
        if (rule == rule_leja){
            nodes = Optimizer::getGreedyNodes<rule_leja>(max_level);
        }else if (rule == rule_maxlebesgue){
            nodes = Optimizer::getGreedyNodes<rule_maxlebesgue>(max_level);
        }else if (rule == rule_minlebesgue){
            nodes = Optimizer::getGreedyNodes<rule_minlebesgue>(max_level);
        }else if (rule == rule_mindelta){
            nodes = Optimizer::getGreedyNodes<rule_mindelta>(max_level);
        }else if (rule == rule_rleja){
            nodes = OneDimensionalNodes::getRLeja(max_level);
        }else if (rule == rule_rlejashifted){
            nodes = OneDimensionalNodes::getRLejaShifted(max_level);
        }
    }
    coeff.resize((size_t) max_level);
    coeff[0] = 1.0;
    for(int i=1; i<max_level; i++){
        coeff[i] = 1.0;
        for(int j=0; j<i; j++) coeff[i] *= (nodes[i] - nodes[j]);
    }
}

std::vector<double> GridSequence::cacheBasisIntegrals() const{
    int max_level = max_levels[0];

    for(auto l: max_levels) if (max_level < l) max_level = l;

    std::vector<double> integ(++max_level, 0.0); // integrals of basis functions

    int n = 1 + max_level / 2; // number of Gauss-Legendre points needed to integrate the basis functions
    std::vector<double> lag_x, lag_w;
    OneDimensionalNodes::getGaussLegendre(n, lag_w, lag_x);

    for(int i=0; i<n; i++){
        double v = 1.0;
        for(int j=1; j<max_level; j++){
            v *= (lag_x[i] - nodes[j-1]);
            integ[j] += lag_w[i] * v / coeff[j];
        }
    }
    integ[0] = 2.0;
    return integ;
}

double GridSequence::evalBasis(const int f[], const int p[]) const{
    double v = 1.0;
    for(int j=0; j<num_dimensions; j++){
        double x = nodes[p[j]];
        double w = 1.0;
        for(int i=0; i<f[j]; i++){
            w *= (x - nodes[i]);
        }
        v *= w / coeff[f[j]];
    }
    return v;
}

void GridSequence::recomputeSurpluses(){
    int num_points = points.getNumIndexes();
    surpluses = Data2D<double>(num_outputs, num_points, std::vector<double>(values.begin(), values.end()));

    int num_levels = 1 + *std::max_element(points.begin(), points.end());

    // construct matrix of basis values at nodes
    std::vector<double> vmatrix(num_levels * num_levels, 0);

    #pragma omp parallel for
    for(int i=0; i<num_levels; i++) {
        double basis = 1.0; // different basis functions
        double x = nodes[i]; // row = i-th output
        vmatrix[i * num_levels] = 1.0;
        for(int j=1; j<i; j++) { // loop over columns (along the row)
            basis *= (x - nodes[j-1]); // numerator of next function
            vmatrix[i * num_levels + j] = basis / coeff[j];
        }
    }

    std::vector<std::vector<int>> map;
    std::vector<std::vector<int>> job_indexes;

    MultiIndexManipulations::resortIndexes(points, map, job_indexes);

    for(int d=num_dimensions-1; d>=0; d--) {
        #pragma omp parallel for
        for(int job = 0; job < static_cast<int>(job_indexes[d].size() - 1); job++) {
            const int offset = job_indexes[d][job] * num_levels + job_indexes[d][job];
            for(int i=job_indexes[d][job]+1; i<job_indexes[d][job+1]; i++) {
                double * istrip = surpluses.getStrip(map[d][i]);
                for(int j=job_indexes[d][job]; j<i; j++) {
                    double * jstrip = surpluses.getStrip(map[d][j]);
                    for(int k=0; k<num_outputs; k++)
                        istrip[k] -= vmatrix[i * num_levels + j - offset] * jstrip[k];
                }
            }
        }
    }
}

template <int mode>
void GridSequence::applyTransformationTransposed(double weights[]) const{
    // mode 0: applies the transposed linear operator for interpolation and quadrature.
    // mode 1: applies the transposed linear operator for differentiation.
    const MultiIndexSet& work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();

    std::vector<int> level = MultiIndexManipulations::computeLevels(work);
    int top_level = *std::max_element(level.begin(), level.end());

    Data2D<int> parents = MultiIndexManipulations::computeDAGup(work);

    std::vector<int> monkey_count(top_level + 1);
    std::vector<int> monkey_tail(top_level + 1);
    std::vector<bool> used(num_points);

    for(int l=top_level; l>0; l--){
        for(int i=0; i<num_points; i++){
            if (level[i] == l){
                const int* p = work.getIndex(i);
                int current = 0;

                monkey_count[0] = 0;
                monkey_tail[0] = i;
                std::fill(used.begin(), used.end(), false);

                while(monkey_count[0] < num_dimensions){
                    if (monkey_count[current] < num_dimensions){
                        int branch = parents.getStrip(monkey_tail[current])[monkey_count[current]];
                        if ((branch == -1) || used[branch]){
                            monkey_count[current]++;
                        }else{
                            if (mode == 0) {
                                weights[branch] -= weights[i] * evalBasis(work.getIndex(branch), p);
                            } else {
                                double weight_mult = evalBasis(work.getIndex(branch), p);
                                for (int d=0; d<num_dimensions; d++)
                                    weights[branch * num_dimensions + d] -= weights[i * num_dimensions + d] * weight_mult;
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

#ifdef Tasmanian_ENABLE_GPU
void GridSequence::updateAccelerationData(AccelerationContext::ChangeType change) const{
    if (change == AccelerationContext::change_gpu_device){
        gpu_cache.reset();
        gpu_cachef.reset();
    }
}
#else
void GridSequence::updateAccelerationData(AccelerationContext::ChangeType) const{}
#endif

}

#endif
