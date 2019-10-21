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

#ifndef __TASMANIAN_SPARSE_GRID_GLOBAL_CPP
#define __TASMANIAN_SPARSE_GRID_GLOBAL_CPP

#include "tsgGridGlobal.hpp"
#include "tsgHiddenExternals.hpp"

namespace TasGrid{

//GridGlobal::GridGlobal() : alpha(0.0), beta(0.0){}
//GridGlobal::~GridGlobal(){}

template<bool iomode> void GridGlobal::write(std::ostream &os) const{
    if (iomode == mode_ascii){ os << std::scientific; os.precision(17); }
    IO::writeNumbers<iomode, IO::pad_rspace>(os, num_dimensions, num_outputs);
    IO::writeNumbers<iomode, IO::pad_line>(os, alpha, beta);
    IO::writeRule<iomode>(rule, os);
    if (rule == rule_customtabulated)
        custom.write<iomode>(os);

    tensors.write<iomode>(os);
    active_tensors.write<iomode>(os);
    if (!active_w.empty())
        IO::writeVector<iomode, IO::pad_line>(active_w, os);

    IO::writeFlag<iomode, IO::pad_auto>(!points.empty(), os);
    if (!points.empty()) points.write<iomode>(os);
    IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
    if (!needed.empty()) needed.write<iomode>(os);

    IO::writeVector<iomode, IO::pad_line>(max_levels, os);

    if (num_outputs > 0) values.write<iomode>(os);

    IO::writeFlag<iomode, IO::pad_line>(!updated_tensors.empty(), os);
    if (!updated_tensors.empty()){
        updated_tensors.write<iomode>(os);
        updated_active_tensors.write<iomode>(os);
        IO::writeVector<iomode, IO::pad_line>(updated_active_w, os);
    }
}

template<bool iomode> void GridGlobal::read(std::istream &is){
    reset(true); // true deletes any custom rule
    num_dimensions = IO::readNumber<iomode, int>(is);
    num_outputs = IO::readNumber<iomode, int>(is);
    alpha = IO::readNumber<iomode, double>(is);
    beta = IO::readNumber<iomode, double>(is);
    rule = IO::readRule<iomode>(is);
    if (rule == rule_customtabulated) custom.read<iomode>(is);
    tensors.read<iomode>(is);
    active_tensors.read<iomode>(is);
    active_w.resize((size_t) active_tensors.getNumIndexes());
    IO::readVector<iomode>(is, active_w);

    if (IO::readFlag<iomode>(is)) points.read<iomode>(is);
    if (IO::readFlag<iomode>(is)) needed.read<iomode>(is);

    max_levels.resize((size_t) num_dimensions);
    IO::readVector<iomode>(is, max_levels);

    if (num_outputs > 0) values.read<iomode>(is);

    int oned_max_level;
    if (IO::readFlag<iomode>(is)){
        updated_tensors.read<iomode>(is);
        oned_max_level = updated_tensors.getMaxIndex();

        updated_active_tensors.read<iomode>(is);

        updated_active_w.resize((size_t) updated_active_tensors.getNumIndexes());
        IO::readVector<iomode>(is, updated_active_w);
    }else{
        oned_max_level = *std::max_element(max_levels.begin(), max_levels.end());
    }

    wrapper.load(custom, oned_max_level, rule, alpha, beta);

    recomputeTensorRefs((points.empty()) ? needed : points);
}

template void GridGlobal::write<mode_ascii>(std::ostream &) const;
template void GridGlobal::write<mode_binary>(std::ostream &) const;
template void GridGlobal::read<mode_ascii>(std::istream &);
template void GridGlobal::read<mode_binary>(std::istream &);

void GridGlobal::reset(bool includeCustom){
    clearAccelerationData();
    tensor_refs = std::vector<std::vector<int>>();
    wrapper = OneDimensionalWrapper();
    tensors = MultiIndexSet();
    active_tensors = MultiIndexSet();
    active_w = std::vector<int>();
    points = MultiIndexSet();
    needed = MultiIndexSet();
    values = StorageSet();
    updated_tensors = MultiIndexSet();
    updated_active_tensors = MultiIndexSet();
    updated_active_w = std::vector<int>();
    if (includeCustom) custom = CustomTabulated();
    num_dimensions = num_outputs = 0;
}

void GridGlobal::clearRefinement(){
    needed = MultiIndexSet();
    updated_tensors = MultiIndexSet();
    updated_active_tensors = MultiIndexSet();
    updated_active_w = std::vector<int>();
}

MultiIndexSet GridGlobal::selectTensors(size_t dims, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights,
                                        TypeOneDRule crule, std::vector<int> const &level_limits) const{
    if (OneDimensionalMeta::isExactLevel(type)){ // no need to know exactness
        return MultiIndexManipulations::selectTensors(dims, depth, type, [&](int l) -> int{ return l; },
                                                      anisotropic_weights, level_limits);
    }else{ // work with exactness specific to the rule
        if (crule == rule_customtabulated){
            if (OneDimensionalMeta::isExactQuadrature(type)){
                return MultiIndexManipulations::selectTensors(dims, depth, type, [&](int l) -> int{ return custom.getQExact(l); },
                                                              anisotropic_weights, level_limits);
            }else{
                return MultiIndexManipulations::selectTensors(dims, depth, type, [&](int l) -> int{ return custom.getIExact(l); },
                                                              anisotropic_weights, level_limits);
            }
        }else{ // using regular OneDimensionalMeta
            if (OneDimensionalMeta::isExactQuadrature(type)){
                return MultiIndexManipulations::selectTensors(dims, depth, type, [&](int l) -> int{ return OneDimensionalMeta::getQExact(l, crule); },
                                                              anisotropic_weights, level_limits);
            }else{
                return MultiIndexManipulations::selectTensors(dims, depth, type, [&](int l) -> int{ return OneDimensionalMeta::getIExact(l, crule); },
                                                              anisotropic_weights, level_limits);
            }
        }
    }
}
void GridGlobal::recomputeTensorRefs(const MultiIndexSet &work){
    int nz_weights = active_tensors.getNumIndexes();
    tensor_refs.resize((size_t) nz_weights);
    if (OneDimensionalMeta::isNonNested(rule)){
        //#pragma omp parallel for schedule(dynamic)
        //for(int i=0; i<nz_weights; i++)
        //    MultiIndexManipulations::referencePoints<false>(active_tensors.getIndex(i), wrapper, work, tensor_refs[i]);
        //Utils::doParallel<int, Utils::SyncParallel>((int) num_threads, nz_weights, [&](int i)->void{
        threaded->compute(nz_weights, [&](int i)->void{
            MultiIndexManipulations::referencePoints<false>(active_tensors.getIndex(i), wrapper, work, tensor_refs[i]);
        });
    }else{
        //#pragma omp parallel for schedule(dynamic)
        //for(int i=0; i<nz_weights; i++)
        //    MultiIndexManipulations::referencePoints<true>(active_tensors.getIndex(i), wrapper, work, tensor_refs[i]);
        //Utils::doParallel<int, Utils::SyncParallel>((int) threaded.get(), nz_weights, [&](int i)->void{
        threaded->compute(nz_weights, [&](int i)->void{
            MultiIndexManipulations::referencePoints<true>(active_tensors.getIndex(i), wrapper, work, tensor_refs[i]);
        });
    }
}

void GridGlobal::makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, TypeOneDRule crule, const std::vector<int> &anisotropic_weights, double calpha, double cbeta, const char* custom_filename, const std::vector<int> &level_limits){
    if (crule == rule_customtabulated){
        custom.read(custom_filename);
    }

    setTensors(selectTensors((size_t) cnum_dimensions, depth, type, anisotropic_weights, crule, level_limits),
               cnum_outputs, crule, calpha, cbeta);
}
void GridGlobal::copyGrid(const GridGlobal *global, int ibegin, int iend){
    custom = CustomTabulated();
    if (global->rule == rule_customtabulated) custom = global->custom;

    num_dimensions = global->num_dimensions;
    num_outputs    = iend - ibegin;
    points = global->points;
    needed = global->needed;

    rule  = global->rule;
    alpha = global->alpha;
    beta  = global->beta;
    wrapper = global->wrapper;

    tensors        = global->tensors;
    active_tensors = global->active_tensors;
    active_w       = global->active_w;

    tensor_refs = global->tensor_refs;
    max_levels  = global->max_levels;

    values = (num_outputs == global->num_outputs) ? global->values : global->values.splitValues(ibegin, iend);

    updated_tensors        = global->updated_tensors;
    updated_active_tensors = global->updated_active_tensors;
    updated_active_w       = global->updated_active_w;

    if (global->dynamic_values){
        dynamic_values = std::unique_ptr<DynamicConstructorDataGlobal>(new DynamicConstructorDataGlobal(*global->dynamic_values));
        if (num_outputs != global->num_outputs) dynamic_values->restrictData(ibegin, iend);
    }
}

void GridGlobal::setTensors(MultiIndexSet &&tset, int cnum_outputs, TypeOneDRule crule, double calpha, double cbeta){
    reset(false);
    tensors = std::move(tset);

    num_dimensions = (int) tensors.getNumDimensions();
    num_outputs = cnum_outputs;
    rule = crule;
    alpha = calpha;  beta = cbeta;

    max_levels = MultiIndexManipulations::getMaxIndexes(tensors);

    wrapper.load(custom, *std::max_element(max_levels.begin(), max_levels.end()), rule, alpha, beta);

    std::vector<int> tensors_w = MultiIndexManipulations::computeTensorWeights(threaded, tensors);
    active_tensors = MultiIndexManipulations::createActiveTensors(tensors, tensors_w);

    active_w.reserve(active_tensors.getNumIndexes());
    for(auto w : tensors_w) if (w != 0) active_w.push_back(w);

    if (OneDimensionalMeta::isNonNested(rule)){
        needed = MultiIndexManipulations::generateNonNestedPoints(threaded, active_tensors, wrapper);
    }else{
        needed = MultiIndexManipulations::generateNestedPoints(threaded, tensors, [&](int l) -> int{ return wrapper.getNumPoints(l); });
    }

    recomputeTensorRefs(needed);

    if (num_outputs == 0){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        values.resize(num_outputs, needed.getNumIndexes());
    }
}

void GridGlobal::proposeUpdatedTensors(){
    int max_level = updated_tensors.getMaxIndex();
    wrapper.load(custom, max_level, rule, alpha, beta);

    std::vector<int> updates_tensor_w = MultiIndexManipulations::computeTensorWeights(threaded, updated_tensors);
    updated_active_tensors = MultiIndexManipulations::createActiveTensors(updated_tensors, updates_tensor_w);

    updated_active_w.reserve(updated_active_tensors.getNumIndexes());
    for(auto w : updates_tensor_w) if (w != 0) updated_active_w.push_back(w);

    MultiIndexSet new_points = (OneDimensionalMeta::isNonNested(rule)) ?
            MultiIndexManipulations::generateNonNestedPoints(threaded, updated_active_tensors, wrapper) :
            MultiIndexManipulations::generateNestedPoints(threaded, updated_tensors, [&](int l) -> int{ return wrapper.getNumPoints(l); });

    needed = new_points.diffSets(points);
}

void GridGlobal::updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    if ((num_outputs == 0) || points.empty()){
        makeGrid(num_dimensions, num_outputs, depth, type, rule, anisotropic_weights, alpha, beta, 0, level_limits);
    }else{
        clearRefinement();

        updated_tensors = selectTensors((size_t) num_dimensions, depth, type, anisotropic_weights, rule, level_limits);

        MultiIndexSet new_tensors = updated_tensors.diffSets(tensors);

        if (!new_tensors.empty()){
            updated_tensors.addMultiIndexSet(tensors);
            proposeUpdatedTensors();
        }
    }
}

void GridGlobal::mapIndexesToNodes(const std::vector<int> &indexes, double *x) const{
    std::transform(indexes.begin(), indexes.end(), x, [&](int i)->double{ return wrapper.getNode(i); });
}

void GridGlobal::getLoadedPoints(double *x) const{
    mapIndexesToNodes(points.getVector(), x);
}
void GridGlobal::getNeededPoints(double *x) const{
    mapIndexesToNodes(needed.getVector(), x);
}
void GridGlobal::getPoints(double *x) const{
    if (points.empty()){ getNeededPoints(x); }else{ getLoadedPoints(x); };
}

void GridGlobal::getQuadratureWeights(double weights[]) const{
    std::fill_n(weights, (points.empty()) ? needed.getNumIndexes() : points.getNumIndexes(), 0.0);

    ThreadEngine tengine(4);

    std::vector<int> num_oned_points(num_dimensions);
    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        const int* levels = active_tensors.getIndex(n);
        num_oned_points[0] = wrapper.getNumPoints(levels[0]);
        int num_tensor_points = num_oned_points[0];
        for(int j=1; j<num_dimensions; j++){
            num_oned_points[j] = wrapper.getNumPoints(levels[j]);
            num_tensor_points *= num_oned_points[j];
        }
        double tensor_weight = (double) active_w[n];

//         #pragma omp parallel for schedule(static)
//         for(int i=0; i<num_tensor_points; i++){
//             int t = i;
//             double w = 1.0;
//             for(int j=num_dimensions-1; j>=0; j--){
//                 w *= wrapper.getWeight(levels[j], t % num_oned_points[j]);
//                 t /= num_oned_points[j];
//             }
//             weights[tensor_refs[n][i]] += tensor_weight * w;
//         }
//         Utils::doParallel<int>((int) num_threads, num_tensor_points, [&](int i)->void{
//             int t = i;
//             double w = 1.0;
//             for(int j=num_dimensions-1; j>=0; j--){
//                 w *= wrapper.getWeight(levels[j], t % num_oned_points[j]);
//                 t /= num_oned_points[j];
//             }
//             weights[tensor_refs[n][i]] += tensor_weight * w;
//         });

        //std::cout << "Start compute" << std::endl;
        tengine.compute(num_tensor_points, [&](int i)->void{
            int t = i;
            double w = 1.0;
            for(int j=num_dimensions-1; j>=0; j--){
                w *= wrapper.getWeight(levels[j], t % num_oned_points[j]);
                t /= num_oned_points[j];
            }
            weights[tensor_refs[n][i]] += tensor_weight * w;
        });
        //std::cout << "End compute" << std::endl;
    }
}

void GridGlobal::getInterpolationWeights(const double x[], double weights[]) const{
    std::fill_n(weights, (points.empty()) ? needed.getNumIndexes() : points.getNumIndexes(), 0.0);

    CacheLagrange<double> lcache(num_dimensions, max_levels, wrapper, x);

    std::vector<int> num_oned_points(num_dimensions);
    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        const int* levels = active_tensors.getIndex(n);
        num_oned_points[0] = wrapper.getNumPoints(levels[0]);
        int num_tensor_points = num_oned_points[0];
        for(int j=1; j<num_dimensions; j++){
            num_oned_points[j] = wrapper.getNumPoints(levels[j]);
            num_tensor_points *= num_oned_points[j];
        }
        double tensor_weight = (double) active_w[n];
        for(int i=0; i<num_tensor_points; i++){
            int t = i;
            double w = 1.0;
            for(int j=num_dimensions-1; j>=0; j--){
                w *= lcache.getLagrange(j, levels[j], t % num_oned_points[j]);
                t /= num_oned_points[j];
            }
            weights[tensor_refs[n][i]] += tensor_weight * w;
        }
    }
}

void GridGlobal::acceptUpdatedTensors(){
    if (points.empty()){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else if (!needed.empty()){
        #ifdef Tasmanian_ENABLE_CUDA
        clearCudaNodes();
        #endif
        points.addMultiIndexSet(needed);
        needed = MultiIndexSet();

        tensors = std::move(updated_tensors);
        updated_tensors = MultiIndexSet();

        active_tensors = std::move(updated_active_tensors);
        updated_active_tensors = MultiIndexSet();

        active_w = std::move(updated_active_w);
        updated_active_w = std::vector<int>();

        max_levels = MultiIndexManipulations::getMaxIndexes(tensors);

        recomputeTensorRefs(points);
    }
}

void GridGlobal::loadNeededPoints(const double *vals){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaValues();
    #endif
    if (points.empty() || needed.empty()){
        values.setValues(vals);
    }else{
        values.addValues(points, needed, vals);
    }
    acceptUpdatedTensors();
}
void GridGlobal::mergeRefinement(){
    if (needed.empty()) return; // nothing to do
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaValues();
    #endif
    int num_all_points = getNumLoaded() + getNumNeeded();
    values.setValues(std::vector<double>(Utils::size_mult(num_outputs, num_all_points), 0.0));
    acceptUpdatedTensors();
}

void GridGlobal::beginConstruction(){
    dynamic_values = std::unique_ptr<DynamicConstructorDataGlobal>(new DynamicConstructorDataGlobal(num_dimensions, num_outputs));
    if (points.empty()){ // if we start dynamic construction from an empty grid
        for(int i=0; i<tensors.getNumIndexes(); i++){
            const int *t = tensors.getIndex(i);
            double weight = -1.0 / (1.0 + (double) std::accumulate(t, t + num_dimensions, 0));
            dynamic_values->addTensor(t, [&](int l)->int{ return wrapper.getNumPoints(l); }, weight);
        }
        tensors = MultiIndexSet();
        active_tensors = MultiIndexSet();
        active_w = std::vector<int>();
        needed = MultiIndexSet();
        values.resize(num_outputs, 0);
    }
}
void GridGlobal::writeConstructionData(std::ostream &os, bool iomode) const{
    if (iomode == mode_ascii) dynamic_values->write<mode_ascii>(os); else dynamic_values->write<mode_binary>(os);
}
void GridGlobal::readConstructionData(std::istream &is, bool iomode){
    dynamic_values = std::unique_ptr<DynamicConstructorDataGlobal>(new DynamicConstructorDataGlobal((size_t) num_dimensions, (size_t) num_outputs));
    if (iomode == mode_ascii)
        dynamic_values->read<mode_ascii>(is);
    else
        dynamic_values->read<mode_binary>(is);
    int max_level = dynamic_values->getMaxTensor();
    if (max_level + 1 > wrapper.getNumLevels())
        wrapper.load(custom, max_level, rule, alpha, beta);
    dynamic_values->reloadPoints([&](int l)->int{ return wrapper.getNumPoints(l); });
}
std::vector<double> GridGlobal::getCandidateConstructionPoints(TypeDepth type, int output, const std::vector<int> &level_limits){
    std::vector<int> weights;
    if ((type == type_iptotal) || (type == type_ipcurved) || (type == type_qptotal) || (type == type_qpcurved)){
        int min_needed_points = ((type == type_ipcurved) || (type == type_qpcurved)) ? 4 * num_dimensions : 2 * num_dimensions;
        if (points.getNumIndexes() > min_needed_points) // if there are enough points to estimate coefficients
            estimateAnisotropicCoefficients(type, output, weights);
    }
    return getCandidateConstructionPoints(type, weights, level_limits);
}
std::vector<double> GridGlobal::getCandidateConstructionPoints(TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    MultiIndexManipulations::ProperWeights weights((size_t) num_dimensions, type, anisotropic_weights);

    // computing the weight for each index requires the cache for the one dimensional indexes
    // the cache for the one dimensional indexes requires the exactness
    // exactness can be one of 5 cases based on level/interpolation/quadrature for custom or known rule
    // the number of required exactness entries is not known until later and we have to be careful not to exceed the levels available for the custom rule
    // thus, we cache the exactness with a delay
    std::vector<int> effective_exactness;
    auto get_exact = [&](int l) -> int{ return effective_exactness[l]; };
    auto build_exactness = [&](size_t num) ->
        void{
            effective_exactness.resize(num);
            if(OneDimensionalMeta::isExactLevel(type)){
                for(size_t i=0; i<num; i++) effective_exactness[i] = (int) i;
            }else if (OneDimensionalMeta::isExactInterpolation(type)){
                if (rule == rule_customtabulated){
                    for(size_t i=0; i<num; i++) effective_exactness[i] = custom.getIExact((int) i);
                }else{
                    for(size_t i=0; i<num; i++) effective_exactness[i] = OneDimensionalMeta::getIExact((int) i, rule);
                }
            }else{ // must be quadrature
                if (rule == rule_customtabulated){
                    for(size_t i=0; i<num; i++) effective_exactness[i] = custom.getQExact((int) i);
                }else{
                    for(size_t i=0; i<num; i++) effective_exactness[i] = OneDimensionalMeta::getQExact((int) i, rule);
                }
            }
        };

    if (weights.contour == type_level){
        std::vector<std::vector<int>> cache;
        return getCandidateConstructionPoints([&](int const *t) -> double{
            if (cache.empty()){
                build_exactness((size_t) wrapper.getNumLevels());
                cache = MultiIndexManipulations::generateLevelWeightsCache<int, type_level, true>(weights, get_exact, wrapper.getNumLevels());
            }

            return (double) MultiIndexManipulations::getIndexWeight<int, type_level>(t, cache);
        }, level_limits);
    }else if (weights.contour == type_curved){
        std::vector<std::vector<double>> cache;
        return getCandidateConstructionPoints([&](int const *t) -> double{
            if (cache.empty()){
                build_exactness((size_t) wrapper.getNumLevels());
                cache = MultiIndexManipulations::generateLevelWeightsCache<double, type_curved, true>(weights, get_exact, wrapper.getNumLevels());
            }

            return MultiIndexManipulations::getIndexWeight<double, type_curved>(t, cache);
        }, level_limits);
    }else{
        std::vector<std::vector<double>> cache;
        return getCandidateConstructionPoints([&](int const *t) -> double{
            if (cache.empty()){
                build_exactness((size_t) wrapper.getNumLevels());
                cache = MultiIndexManipulations::generateLevelWeightsCache<double, type_hyperbolic, true>(weights, get_exact, wrapper.getNumLevels());
            }

            return MultiIndexManipulations::getIndexWeight<double, type_hyperbolic>(t, cache);
        }, level_limits);
    }
}
std::vector<double> GridGlobal::getCandidateConstructionPoints(std::function<double(const int *)> getTensorWeight, const std::vector<int> &level_limits){
    dynamic_values->clearTesnors(); // clear old tensors
    MultiIndexSet init_tensors = dynamic_values->getInitialTensors(); // get the initial tensors (created with make grid)

    MultiIndexSet new_tensors = (level_limits.empty()) ?
        MultiIndexManipulations::addExclusiveChildren<false>(tensors, init_tensors, level_limits) :
        MultiIndexManipulations::addExclusiveChildren<true>(tensors, init_tensors, level_limits);

    if (!new_tensors.empty()){
        auto max_indexes = MultiIndexManipulations::getMaxIndexes(new_tensors);
        int max_level = *std::max_element(max_indexes.begin(), max_indexes.end());
        if (max_level+1 > wrapper.getNumLevels())
            wrapper.load(custom, max_level, rule, alpha, beta);
    }

    std::vector<double> tweights(new_tensors.getNumIndexes());
    for(int i=0; i<new_tensors.getNumIndexes(); i++)
        tweights[i] = (double) getTensorWeight(new_tensors.getIndex(i));

    for(int i=0; i<new_tensors.getNumIndexes(); i++){
        const int *t = new_tensors.getIndex(i);
        dynamic_values->addTensor(t, [&](int l)->int{ return wrapper.getNumPoints(l); }, tweights[i]);
    }
    std::vector<int> node_indexes;
    dynamic_values->getNodesIndexes(node_indexes);
    std::vector<double> x(node_indexes.size());
    mapIndexesToNodes(node_indexes, x.data());
    return x;
}
std::vector<int> GridGlobal::getMultiIndex(const double x[]){
    std::vector<int> p(num_dimensions);
    for(int j=0; j<num_dimensions; j++){
        int i = 0;
        while(std::abs(wrapper.getNode(i) - x[j]) > Maths::num_tol){
            i++; // convert canonical node to index
            if (i == wrapper.getNumNodes())
                wrapper.load(custom, wrapper.getNumLevels(), wrapper.getType(), alpha, beta);
        }
        p[j] = i;
    }
    return p;
}
void GridGlobal::loadConstructedPoint(const double x[], const std::vector<double> &y){
    if (dynamic_values->addNewNode(getMultiIndex(x), y)) // if a new tensor is complete
        loadConstructedTensors();
}
void GridGlobal::loadConstructedPoint(const double x[], int numx, const double y[]){
    Utils::Wrapper2D<const double> wrapx(num_dimensions, x);
    Utils::Wrapper2D<const double> wrapy(num_outputs, y);
    for(int i=0; i<numx; i++)
        dynamic_values->addNewNode(getMultiIndex(wrapx.getStrip(i)), std::vector<double>(wrapy.getStrip(i), wrapy.getStrip(i) + num_outputs));
    loadConstructedTensors();
}
void GridGlobal::loadConstructedTensors(){
    MultiIndexSet new_tensors, new_points;
    StorageSet new_values;
    dynamic_values->ejectCompleteTensor(tensors, new_tensors, new_points, new_values);
    if (new_tensors.empty()) return; // nothing to do

    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaNodes();
    clearCudaValues();
    #endif

    if (points.empty()){ // no loaded points yet
        values = std::move(new_values);
        points = std::move(new_points);
    }else{
        values.addValues(points, new_points, new_values.getValues(0));
        points.addMultiIndexSet(new_points);
    }

    tensors.addMultiIndexSet(new_tensors);
    // recompute the tensor weights
    auto tensors_w = MultiIndexManipulations::computeTensorWeights(threaded, tensors);
    active_tensors = MultiIndexManipulations::createActiveTensors(tensors, tensors_w);

    active_w = std::vector<int>();
    active_w.reserve(active_tensors.getNumIndexes());
    for(auto w : tensors_w) if (w != 0) active_w.push_back(w);

    max_levels = MultiIndexManipulations::getMaxIndexes(active_tensors);

    recomputeTensorRefs(points);
}
void GridGlobal::finishConstruction(){
    dynamic_values = std::unique_ptr<DynamicConstructorDataGlobal>();
}

const double* GridGlobal::getLoadedValues() const{
    if (getNumLoaded() == 0) return 0;
    return values.getValues(0);
}

void GridGlobal::evaluate(const double x[], double y[]) const{
    std::vector<double> w(points.getNumIndexes());
    getInterpolationWeights(x, w.data());
    std::fill_n(y, num_outputs, 0.0);
    for(int i=0; i<points.getNumIndexes(); i++){
        const double *v = values.getValues(i);
        double wi = w[i];
        for(int k=0; k<num_outputs; k++) y[k] += wi * v[k];
    }
}
void GridGlobal::evaluateBatch(const double x[], int num_x, double y[]) const{
    Utils::Wrapper2D<const double> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_outputs, y);
    //#pragma omp parallel for
    //for(int i=0; i<num_x; i++)
    //    evaluate(xwrap.getStrip(i), ywrap.getStrip(i));
    //Utils::doParallel<int>((int) num_threads, num_x, [&](int i)->void{
    threaded->compute(num_x, [&](int i)->void{
        evaluate(xwrap.getStrip(i), ywrap.getStrip(i));
    });
}

#ifdef Tasmanian_ENABLE_BLAS
void GridGlobal::evaluateBlas(const double x[], int num_x, double y[]) const{
    int num_points = points.getNumIndexes();
    Data2D<double> weights(num_points, num_x);
    if (num_x > 1){
        evaluateHierarchicalFunctions(x, num_x, weights.getStrip(0));
    }else{ // skips small OpenMP overhead
        getInterpolationWeights(x, weights.getStrip(0));
    }

    TasBLAS::denseMultiply(num_outputs, num_x, num_points, 1.0, values.getValues(0), weights.getStrip(0), 0.0, y);
}
#endif // Tasmanian_ENABLE_BLAS

#ifdef Tasmanian_ENABLE_CUDA
void GridGlobal::loadNeededPointsCuda(CudaEngine *, const double *vals){
    loadNeededPoints(vals);
}
void GridGlobal::evaluateCudaMixed(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    loadCudaValues<double>();

    int num_points = points.getNumIndexes();
    Data2D<double> weights(num_points, num_x);
    evaluateHierarchicalFunctions(x, num_x, weights.getStrip(0));

    engine->denseMultiply(num_outputs, num_x, num_points, 1.0, cuda_cache->values, weights.getVector(), y);
}
void GridGlobal::evaluateCuda(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    CudaVector<double> gpu_x(num_dimensions, num_x, x), gpu_result(num_x, num_outputs);
    evaluateBatchGPU(engine, gpu_x.data(), num_x, gpu_result.data());
    gpu_result.unload(y);
}
template<typename T> void GridGlobal::evaluateBatchGPUtempl(CudaEngine *engine, T const gpu_x[], int cpu_num_x, T gpu_y[]) const{
    loadCudaValues<T>();
    int num_points = points.getNumIndexes();

    CudaVector<T> gpu_basis(cpu_num_x, num_points);
    evaluateHierarchicalFunctionsGPU(gpu_x, cpu_num_x, gpu_basis.data());
    engine->denseMultiply(num_outputs, cpu_num_x, num_points, 1.0, getCudaCache(static_cast<T>(0.0))->values, gpu_basis, 0.0, gpu_y);
}
void GridGlobal::evaluateBatchGPU(CudaEngine *engine, const double *gpu_x, int cpu_num_x, double gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
void GridGlobal::evaluateBatchGPU(CudaEngine *engine, const float gpu_x[], int cpu_num_x, float gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
template<typename T> void GridGlobal::evaluateHierarchicalFunctionsGPUtempl(T const gpu_x[], int cpu_num_x, T *gpu_y) const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    loadCudaNodes<T>();
    TasCUDA::devalglo(!OneDimensionalMeta::isNonNested(rule), (rule == rule_clenshawcurtis0), num_dimensions, cpu_num_x, getNumPoints(),
                      ccache->num_basis,
                      gpu_x, ccache->nodes, ccache->coeff, ccache->tensor_weights,
                      ccache->nodes_per_level, ccache->offset_per_level, ccache->map_dimension, ccache->map_level,
                      ccache->active_tensors, ccache->active_num_points, ccache->dim_offsets,
                      ccache->map_tensor, ccache->map_index, ccache->map_reference, gpu_y);
}
void GridGlobal::evaluateHierarchicalFunctionsGPU(const double gpu_x[], int cpu_num_x, double *gpu_y) const{
    evaluateHierarchicalFunctionsGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
void GridGlobal::evaluateHierarchicalFunctionsGPU(const float gpu_x[], int cpu_num_x, float *gpu_y) const{
    evaluateHierarchicalFunctionsGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
template<typename T> void GridGlobal::loadCudaValues() const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaGlobalData<T>>();
    if (ccache->values.empty()) ccache->values.load(values.getVector());
}
void GridGlobal::clearCudaValues() const{ if (cuda_cache) cuda_cache->values.clear(); }
template<typename T> void GridGlobal::loadCudaNodes() const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaGlobalData<T>>();
    if (!ccache->nodes.empty()) return; // already loaded
    // data for stage 1 (Lagrange caching)
    ccache->nodes.load((OneDimensionalMeta::isNonNested(rule)) ? wrapper.getAllNodes() : wrapper.getUnique());
    ccache->coeff.load(wrapper.getAllCoeff());
    ccache->nodes_per_level.load(wrapper.getNumNodesPerLevel());
    ccache->offset_per_level.load(wrapper.getOffsetNodesPerLevel());

    std::vector<int> map_dims, map_level, dim_offsets;
    int num_basis = 0;
    for(int dim=0; dim<num_dimensions; dim++){
        dim_offsets.push_back(num_basis);
        for(int level=0; level <= max_levels[dim]; level++){
            map_dims.push_back(dim);
            map_level.push_back(level);
            num_basis += wrapper.getNumPoints(level);
        }
    }
    ccache->num_basis = num_basis;
    ccache->map_dimension.load(map_dims);
    ccache->map_level.load(map_level);

    std::vector<double> tweights(active_w.size());
    std::transform(active_w.begin(), active_w.end(), tweights.begin(), [](int i)->double{ return static_cast<double>(i); });
    ccache->tensor_weights.load(tweights);

    ccache->active_tensors.load(active_tensors.getVector());

    std::vector<int> active_num_points(active_tensors.getVector().size());
    std::transform(active_tensors.getVector().begin(), active_tensors.getVector().end(), active_num_points.begin(), [&](int i)->int{ return wrapper.getNumPoints(i); });
    ccache->active_num_points.load(active_num_points);

    ccache->dim_offsets.load(dim_offsets);

    std::vector<int> map_tensor, map_index, map_reference;
    auto inump = active_num_points.begin();
    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        int num_tensor_points = *inump++;
        for(int j=1; j<num_dimensions; j++)
            num_tensor_points *= *inump++;

        for(int i=0; i<num_tensor_points; i++){
            map_tensor.push_back(n);
            map_index.push_back(i);
            map_reference.push_back(tensor_refs[n][i]);
        }
    }
    ccache->map_tensor.load(map_tensor);
    ccache->map_index.load(map_index);
    ccache->map_reference.load(map_reference);
}
void GridGlobal::clearCudaNodes() const{
    if (cuda_cache) cuda_cache->clearNodes();
    if (cuda_cachef) cuda_cachef->clearNodes();
}
#endif // Tasmanian_ENABLE_CUDA

void GridGlobal::integrate(double q[], double *conformal_correction) const{
    std::vector<double> w(getNumPoints());
    getQuadratureWeights(w.data());
    if (conformal_correction != 0) for(int i=0; i<points.getNumIndexes(); i++) w[i] *= conformal_correction[i];
    std::fill(q, q+num_outputs, 0.0);
    //#pragma omp parallel for schedule(static)
    for(int i=0; i<points.getNumIndexes(); i++){
        const double *v = values.getValues(i);
        for(int k=0; k<num_outputs; k++) q[k] += w[i] * v[k];
    }
}

void GridGlobal::evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const{
    int num_points = (points.empty()) ? needed.getNumIndexes() : points.getNumIndexes();
    Utils::Wrapper2D<const double> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_points, y);
    //#pragma omp parallel for
    //for(int i=0; i<num_x; i++)
    //    getInterpolationWeights(xwrap.getStrip(i), ywrap.getStrip(i));
    //Utils::doParallel<int>((int) num_threads, num_x, [&](int i)->void{
    threaded->compute(num_x, [&](int i)->void{
        getInterpolationWeights(xwrap.getStrip(i), ywrap.getStrip(i));
    });
}

std::vector<double> GridGlobal::computeSurpluses(int output, bool normalize) const{
    int num_points = points.getNumIndexes();
    std::vector<double> surp((size_t) num_points);

    if (OneDimensionalMeta::isSequence(rule)){
        double max_surp = 0.0;
        for(int i=0; i<num_points; i++){
            surp[i] = values.getValues(i)[output];
            if (std::abs(surp[i]) > max_surp) max_surp = std::abs(surp[i]);
        }

        GridSequence seq(threaded); // there is an extra copy here, but the sequence grid does the surplus computation automatically
        MultiIndexSet spoints = points;
        seq.setPoints(spoints, 1, rule);

        seq.loadNeededPoints(surp.data());

        std::copy_n(seq.getSurpluses(), surp.size(), surp.data());

        if (normalize) for(auto &s : surp) s /= max_surp;
    }else{
        MultiIndexSet polynomial_set = getPolynomialSpaceSet(true);

        MultiIndexSet quadrature_tensors =
            MultiIndexManipulations::generateLowerMultiIndexSet((size_t) num_dimensions, [&](const std::vector<int> &index) ->
                bool{
                    std::vector<int> qindex = index;
                    for(auto &i : qindex) i = (i > 0) ? 1 + OneDimensionalMeta::getQExact(i - 1, rule_gausspatterson) : 0;
                    return !polynomial_set.missing(qindex);
                });

        int getMaxQuadLevel = quadrature_tensors.getMaxIndex();

        GridGlobal QuadGrid(threaded);
        if (getMaxQuadLevel < TableGaussPatterson::getNumLevels()-1){
            QuadGrid.setTensors(std::move(quadrature_tensors), 0, rule_gausspatterson, 0.0, 0.0);
        }else{
            quadrature_tensors =
                MultiIndexManipulations::generateLowerMultiIndexSet((size_t) num_dimensions, [&](const std::vector<int> &index) ->
                    bool{
                        std::vector<int> qindex = index;
                        for(auto &i : qindex) i = (i > 0) ? 1 + OneDimensionalMeta::getQExact(i - 1, rule_clenshawcurtis) : 0;
                        return polynomial_set.missing(qindex);
                    });
            QuadGrid.setTensors(std::move(quadrature_tensors), 0, rule_clenshawcurtis, 0.0, 0.0);
        }

        size_t qn = (size_t) QuadGrid.getNumPoints();
        std::vector<double> w(qn);
        QuadGrid.getQuadratureWeights(w.data());
        std::vector<double> x(qn * ((size_t) num_dimensions));
        std::vector<double> y(qn * ((size_t) num_outputs));
        QuadGrid.getPoints(x.data());
        std::vector<double> I(qn);

        evaluateBatch(x.data(), (int) qn, y.data());
        auto iter_y = y.begin();
        std::advance(iter_y, output);
        for(auto &ii : I){
            ii = *iter_y;
            std::advance(iter_y, num_outputs);
        }

        //#pragma omp parallel for schedule(dynamic)
        //for(int i=0; i<num_points; i++){
        //    // for each surp, do the quadrature
        //    const int* p = points.getIndex(i);
        //    double c = 0.0;
        //    auto iter_x = x.begin();
        //    for(auto ii: I){
        //        double v = 1.0;
        //        for(int j=0; j<num_dimensions; j++) v *= legendre(p[j], *iter_x++);
        //        c += v * ii;
        //    }
        //    double nrm = std::sqrt((double) p[0] + 0.5);
        //    for(int j=1; j<num_dimensions; j++){
        //        nrm *= std::sqrt((double) p[j] + 0.5);
        //    }
        //    surp[i] = c * nrm;
        //}
        //Utils::doParallel<int, Utils::SyncParallel>((int) num_threads, num_points, [&](int i)->void{
        threaded->compute(num_points, [&](int i)->void{
            //for each surp, do the quadrature
            const int* p = points.getIndex(i);
            double c = 0.0;
            auto iter_x = x.begin();
            for(auto ii: I){
                double v = 1.0;
                for(int j=0; j<num_dimensions; j++) v *= legendre(p[j], *iter_x++);
                c += v * ii;
            }
            double nrm = std::sqrt((double) p[0] + 0.5);
            for(int j=1; j<num_dimensions; j++){
                nrm *= std::sqrt((double) p[j] + 0.5);
            }
            surp[i] = c * nrm;
        });
    }
    return surp;
}

void GridGlobal::estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const{
    double tol = 1000.0 * Maths::num_tol;
    std::vector<double> surp = computeSurpluses(output, false);

    int num_points = points.getNumIndexes();

    int n = 0, m;
    for(int j=0; j<num_points; j++){
        surp[j] = std::abs(surp[j]);
        if (surp[j] > tol) n++;
    }

    std::vector<double> b(n);
    Data2D<double> A;

    if ((type == type_curved) || (type == type_ipcurved) || (type == type_qpcurved)){
        m = 2*num_dimensions + 1;
        A.resize(n, m);

        int count = 0;
        for(int c=0; c<num_points; c++){
            const int *indx = points.getIndex(c);
            if (surp[c] > tol){
                for(int j=0; j<num_dimensions; j++){
                    A.getStrip(j)[count] = ((double) indx[j]);
                }
                for(int j=0; j<num_dimensions; j++){
                    A.getStrip(num_dimensions + j)[count] = log((double) (indx[j] + 1));
                }
                A.getStrip(2*num_dimensions)[count] = 1.0;
                b[count++] = -log(surp[c]);
            }
        }
    }else{
        m = num_dimensions + 1;
        A.resize(n, m);

        int count = 0;
        for(int c=0; c<num_points; c++){
            const int *indx = points.getIndex(c);
            if (surp[c] > tol){
                for(int j=0; j<num_dimensions; j++){
                    A.getStrip(j)[count] = ((double) indx[j]);
                }
                A.getStrip(num_dimensions)[count] = 1.0;
                b[count++] = - log(surp[c]);
            }
        }
    }

    std::vector<double> x(m);
    TasmanianDenseSolver::solveLeastSquares(n, m, A.getStrip(0), b.data(), 1.E-5, x.data());

    weights.resize(--m);
    for(int j=0; j<m; j++){
        weights[j] = (int)(x[j] * 1000.0 + 0.5);
    }

    int min_weight = weights[0];  for(int j=1; j<num_dimensions; j++){ if (min_weight < weights[j]) min_weight = weights[j];  } // start by finding the largest weight
    if (min_weight < 0){ // all directions are diverging, default to isotropic total degree
        for(int j=0; j<num_dimensions; j++)  weights[j] = 1;
        if (m == 2*num_dimensions) for(int j=num_dimensions; j<2*num_dimensions; j++)  weights[j] = 0;
    }else{
        for(int j=0; j<num_dimensions; j++)  if((weights[j]>0) && (weights[j]<min_weight)) min_weight = weights[j];  // find the smallest positive weight
        for(int j=0; j<num_dimensions; j++){
            if (weights[j] <= 0){
                weights[j] = min_weight;
                if (m == 2*num_dimensions){
                    if (std::abs(weights[num_dimensions + j]) > weights[j]){
                        weights[num_dimensions + j] = (weights[num_dimensions + j] > 0.0) ? weights[j] : -weights[j];
                    }
                }
            }
        }
    }
}

void GridGlobal::setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits){
    clearRefinement();
    std::vector<int> weights;
    estimateAnisotropicCoefficients(type, output, weights);

    int level = 0;
    do{
        updateGrid(++level, type, weights, level_limits);
    }while(getNumNeeded() < min_growth);
}

void GridGlobal::setSurplusRefinement(double tolerance, int output, const std::vector<int> &level_limits){
    clearRefinement();
    std::vector<double> surp = computeSurpluses(output, true);

    int n = points.getNumIndexes();
    std::vector<bool> flagged(n);

    for(int i=0; i<n; i++)
        flagged[i] = (std::abs(surp[i]) > tolerance);

    MultiIndexSet kids = MultiIndexManipulations::selectFlaggedChildren(points, flagged, level_limits);

    if (kids.getNumIndexes() > 0){
        kids.addMultiIndexSet(points);
        MultiIndexManipulations::completeSetToLower(kids);

        updated_tensors = std::move(kids);
        proposeUpdatedTensors();
    }
}
void GridGlobal::setHierarchicalCoefficients(const double c[], TypeAcceleration){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaValues();
    #endif
    if (!points.empty()) clearRefinement();
    loadNeededPoints(c);
}
void GridGlobal::integrateHierarchicalFunctions(double integrals[]) const{ getQuadratureWeights(integrals); }

double GridGlobal::legendre(int n, double x){
    if (n == 0) return 1.0;
    if (n == 1) return x;

    double lm = 1, l = x;
    for(int i=2; i<=n; i++){
        double lp = ((2*i - 1) * x * l) / ((double) i) - ((i - 1) * lm) / ((double) i);
        lm = l; l = lp;
    }
    return l;
}

void GridGlobal::clearAccelerationData(){
    #ifdef Tasmanian_ENABLE_CUDA
    cuda_cache.reset();
    cuda_cachef.reset();
    #endif
}

MultiIndexSet GridGlobal::getPolynomialSpaceSet(bool interpolation) const{
    if (interpolation){
        if (rule == rule_customtabulated){
            return MultiIndexManipulations::createPolynomialSpace(std::make_unique<ThreadEngine>(1).get(), active_tensors, [&](int l)-> int{ return custom.getIExact(l); });
        }else{
            return MultiIndexManipulations::createPolynomialSpace(std::make_unique<ThreadEngine>(1).get(), active_tensors, [&](int l)-> int{ return OneDimensionalMeta::getIExact(l, rule); });
        }
    }else{
        if (rule == rule_customtabulated){
            return MultiIndexManipulations::createPolynomialSpace(std::make_unique<ThreadEngine>(1).get(), active_tensors, [&](int l)-> int{ return custom.getQExact(l); });
        }else{
            return MultiIndexManipulations::createPolynomialSpace(std::make_unique<ThreadEngine>(1).get(), active_tensors, [&](int l)-> int{ return OneDimensionalMeta::getQExact(l, rule); });
        }
    }
}

std::vector<int> GridGlobal::getPolynomialSpace(bool interpolation) const{
    return std::move(getPolynomialSpaceSet(interpolation).getVector());
}

}

#endif
