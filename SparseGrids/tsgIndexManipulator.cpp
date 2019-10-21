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

#ifndef __TSG_INDEX_MANIPULATOR_CPP
#define __TSG_INDEX_MANIPULATOR_CPP

#include "tsgIndexManipulator.hpp"

namespace TasGrid{

namespace MultiIndexManipulations{

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Generate a series of \b level_sets where each set has the parents/children of the previous one that satisfy the \b inside() criteria.
 *
 * On entry, \b level_sets must constain at least one set.
 * The function takes the last set in \b level_sets and adds a new set of
 * eithe the parents or children that also satisfy the \b inside() condition.
 * The process is repeated until the new set is empty.
 * \endinternal
 */
template<bool use_parents>
void repeatAddIndexes(std::function<bool(const std::vector<int> &index)> inside, std::vector<MultiIndexSet> &level_sets){
    size_t num_dimensions = level_sets.back().getNumDimensions();
    bool adding = true;
    while(adding){
        Data2D<int> level((int) num_dimensions, 0);
        int num_indexes = level_sets.back().getNumIndexes();
        for(int i=0; i<num_indexes; i++){
            std::vector<int> point(num_dimensions);
            std::copy_n(level_sets.back().getIndex(i), num_dimensions, point.data());
            for(auto &p : point){
                p += (use_parents) ? -1 : 1; // parents have lower index, children have higher indexes
                if ( (!use_parents || (p >= 0)) && inside(point) ) level.appendStrip(point);
                p -= (use_parents) ? -1 : 1; // restore p
            }
        }

        adding = (level.getNumStrips() > 0);
        if (adding)
            level_sets.push_back(MultiIndexSet(level));
    }
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Retuns the union of all \b level_sets, all sets are destroyed in the process.
 *
 * \endinternal
 */
inline MultiIndexSet unionSets(std::vector<MultiIndexSet> &level_sets){
    size_t num_levels = level_sets.size();
    while(num_levels > 1){
        size_t stride = num_levels / 2 + (((num_levels % 2) > 0) ? 1 : 0);
        for(size_t i=0; i<stride; i++)
            if (i + stride < num_levels)
                level_sets[i].addMultiIndexSet(level_sets[i + stride]);
        num_levels = stride;
    }
    return std::move(level_sets[0]);
}

void completeSetToLower(MultiIndexSet &set){
    size_t num_dimensions = set.getNumDimensions();
    int num = set.getNumIndexes();
    Data2D<int> completion((int) num_dimensions, 0);
    for(int i=0; i<num; i++){
        std::vector<int> point(num_dimensions);
        std::copy_n(set.getIndex(i), num_dimensions, point.data());
        for(auto &p : point){
            if (p != 0){
                p--;
                if (set.missing(point)) completion.appendStrip(point);
                p++;
            }
        }
    }

    if (completion.getNumStrips() > 0){
        std::vector<MultiIndexSet> level_sets = { MultiIndexSet(completion) };

        repeatAddIndexes<true>([&](std::vector<int> const &p) -> bool{ return set.missing(p); }, level_sets);

        set.addMultiIndexSet(unionSets(level_sets));
    }
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Generate the minimum lower complete multi-index set that includes the indexes satisfying \b criteria(), assumes \b criteria() defines a connected set.
 *
 * \endinternal
 */
inline MultiIndexSet generateGeneralMultiIndexSet(size_t num_dimensions, std::function<bool(const std::vector<int> &index)> criteria){
    std::vector<MultiIndexSet> level_sets = { MultiIndexSet(num_dimensions, std::vector<int>(num_dimensions, 0)) };

    repeatAddIndexes<false>(criteria, level_sets);

    MultiIndexSet set = unionSets(level_sets);

    completeSetToLower(set);
    return set;
}

/*!
 * \internal
 * \brief Generate the multi-index of indexes with weighs less than the \b normalized_offset.
 *
 * The weight of an index uses the \b weights combined with the \b rule_exactness().
 * Called only when the set is guaranteed to be lower complete,
 * then the one dimensional weights can be cached prior to running the selection algorithm.
 *
 * If \b check_limits is \b false, then \b level_limits are ignored for speedup.
 * \endinternal
 */
template<bool check_limits>
MultiIndexSet selectLowerSet(ProperWeights const &weights, std::function<int(int i)> rule_exactness,
                             int normalized_offset, std::vector<int> const &level_limits){
    size_t num_dimensions = weights.getNumDimensions();
    if (weights.contour == type_level){
        auto cache = generateLevelWeightsCache<int, type_level, false>(weights, rule_exactness, normalized_offset);
        return generateLowerMultiIndexSet(num_dimensions,
                [&](std::vector<int> const &index)->bool{
                    if (check_limits) for(size_t j=0; j<num_dimensions; j++) if ((level_limits[j] > -1) && (index[j] > level_limits[j])) return false;
                    return (getIndexWeight<int, type_level>(index.data(), cache) <= normalized_offset);
                });
    }else if (weights.contour == type_curved){
        auto cache = generateLevelWeightsCache<double, type_curved, false>(weights, rule_exactness, normalized_offset);
        double noff = (double) normalized_offset;
        return generateLowerMultiIndexSet(num_dimensions,
                [&](std::vector<int> const &index)->bool{
                    if (check_limits) for(size_t j=0; j<num_dimensions; j++) if ((level_limits[j] > -1) && (index[j] > level_limits[j])) return false;
                    return (std::ceil(getIndexWeight<double, type_curved>(index.data(), cache)) <= noff);
                });
    }else{ // type_hyperbolic
        auto cache = generateLevelWeightsCache<double, type_hyperbolic, false>(weights, rule_exactness, normalized_offset);
        double noff = (double) normalized_offset;
        return generateLowerMultiIndexSet(num_dimensions,
                [&](std::vector<int> const &index)->bool{
                    if (check_limits) for(size_t j=0; j<num_dimensions; j++) if ((level_limits[j] > -1) && (index[j] > level_limits[j])) return false;
                    return (std::ceil(getIndexWeight<double, type_hyperbolic>(index.data(), cache)) <= noff);
                });
    }
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Generates the minimum lower complete set that contains all indexes with weights less than \b normalized_offset.
 *
 * The weight of an index uses the \b weights combined with the \b rule_exactness().
 * Called only for contour \b type_curved and caches values on-the-fly.
 *
 * If \b check_limits is \b false, then \b level_limits are ignored for speedup.
 * \endinternal
 */
template<bool check_limits>
MultiIndexSet selectGeneralSet(ProperWeights const &weights, std::function<int(int i)> rule_exactness,
                                      int normalized_offset, std::vector<int> const &level_limits){
    size_t num_dimensions = weights.getNumDimensions();
    std::vector<std::vector<double>> cache(num_dimensions);
    for(size_t j=0; j<num_dimensions; j++) cache[j].push_back(0.0);
    double noff = (double) normalized_offset;
    return generateGeneralMultiIndexSet(num_dimensions,
                                        [&](std::vector<int> const &index) -> bool{
                                            if (check_limits) for(size_t j=0; j<num_dimensions; j++) if (index[j] > level_limits[j]) return false;
                                            double w = 0;
                                            for(size_t j=0; j<num_dimensions; j++){
                                                while(index[j] >= (int) cache[j].size()){
                                                    int exactness = 1 + rule_exactness((int)(cache[j].size() - 1));
                                                    cache[j].push_back( (double) weights.linear[j] * exactness +
                                                                         weights.curved[j] * std::log1p((double) exactness) );
                                                }
                                                w += cache[j][index[j]];
                                            }
                                            return (std::ceil(w) <= noff);
                                        });
}


MultiIndexSet selectTensors(size_t num_dimensions, int offset, TypeDepth type,
                                                     std::function<int(int i)> rule_exactness, std::vector<int> const &anisotropic_weights,
                                                     std::vector<int> const &level_limits){
    // special case of a full tensor selection
    if ((type == type_tensor) || (type == type_iptensor) || (type == type_qptensor)){ // special case, full tensor
        std::vector<int> max_exactness = (anisotropic_weights.empty()) ? std::vector<int>(num_dimensions, 1) : anisotropic_weights;
        for(auto &e : max_exactness) e *= offset;
        std::vector<int> num_points(num_dimensions, 0); // how many points to have in each direction
        std::transform(max_exactness.begin(), max_exactness.end(), num_points.begin(),
                        [&](int e)-> int{
                            int l = 0; // level
                            while(rule_exactness(l) < e) l++; // get the first level that covers the weight
                            return l+1; // adding extra one to change interpretation from 0-index "level" to 1-index "number of points"
                        });
        if (!level_limits.empty()){
            for(size_t j=0; j<num_dimensions; j++)
                if (level_limits[j] >= 0)
                    num_points[j] = std::min(num_points[j], level_limits[j]+1); // the +1 indicates switch from max-level to number-of-points
        }
        return generateFullTensorSet(num_points);
    }

    ProperWeights weights(num_dimensions, type, anisotropic_weights);

    int normalized_offset = offset * weights.minLinear();
    if (weights.provenLower()){ // if the set is guaranteed to be lower
        if (level_limits.empty()){
            return selectLowerSet<false>(weights, rule_exactness, normalized_offset, level_limits);
        }else{
            return selectLowerSet<true>(weights, rule_exactness, normalized_offset, level_limits);
        }
    }else{
        if (level_limits.empty()){
            return selectGeneralSet<false>(weights, rule_exactness, normalized_offset, level_limits);
        }else{
            return selectGeneralSet<true>(weights, rule_exactness, normalized_offset, level_limits);
        }
    }
}

std::vector<int> computeLevels(size_t num_threads, MultiIndexSet const &mset){
    // cannot add as inline to the public header due to the pragma and possible "unknown pragma" message
    int num_indexes = mset.getNumIndexes();
    size_t num_dimensions = mset.getNumDimensions();
    std::vector<int> levels((size_t) num_indexes);
//     #pragma omp parallel for
//     for(int i=0; i<num_indexes; i++){
    Utils::doParallel<int>((int) num_threads, num_indexes, [&](int i)->void{
        const int* p = mset.getIndex(i);
        levels[i] = std::accumulate(p, p + num_dimensions, 0);
    });
//   }
    return levels;
}

std::vector<int> getMaxIndexes(const MultiIndexSet &mset){
    size_t num_dimensions = mset.getNumDimensions();
    std::vector<int> max_index(num_dimensions, 0);
    int n = mset.getNumIndexes();
    for(int i=0; i<n; i++){
        const int* p = mset.getIndex(i);
        for(size_t j=0; j<num_dimensions; j++) if (max_index[j] < p[j]) max_index[j] = p[j];
    }
    return max_index;
}

Data2D<int> computeDAGup(size_t num_threads, MultiIndexSet const &mset){
    size_t num_dimensions = (size_t) mset.getNumDimensions();
    int n = mset.getNumIndexes();
    Data2D<int> parents(mset.getNumDimensions(), n);
    //#pragma omp parallel for schedule(static)
    //for(int i=0; i<n; i++){
    Utils::doParallel<int>((int) num_threads, n, [&](int i)->void{
        std::vector<int> dad(num_dimensions);
        std::copy_n(mset.getIndex(i), num_dimensions, dad.data());
        int *v = parents.getStrip(i);
        for(auto &d : dad){
            d--;
            *v = (d < 0) ? -1 : mset.getSlot(dad);
            d++;
            v++;
        }
    });
    //}
    return parents;
}

MultiIndexSet selectFlaggedChildren(const MultiIndexSet &mset, const std::vector<bool> &flagged, const std::vector<int> &level_limits){
    size_t num_dimensions = mset.getNumDimensions();

    Data2D<int> children_unsorted(mset.getNumDimensions(), 0);

    std::vector<int> kid(num_dimensions);

    int n = mset.getNumIndexes();
    if (level_limits.empty()){
        for(int i=0; i<n; i++){
            if (flagged[i]){
                std::copy_n(mset.getIndex(i), num_dimensions, kid.data());
                for(auto &k : kid){
                    k++;
                    if (mset.missing(kid)) children_unsorted.appendStrip(kid);
                    k--;
                }
            }
        }
    }else{
        for(int i=0; i<n; i++){
            if (flagged[i]){
                std::copy_n(mset.getIndex(i), num_dimensions, kid.data());
                auto ill = level_limits.begin();
                for(auto &k : kid){
                    k++;
                    if (((*ill == -1) || (k <= *ill)) && mset.missing(kid))
                        children_unsorted.appendStrip(kid);
                    k--;
                    ill++;
                }
            }
        }
    }

    return MultiIndexSet(children_unsorted);
}

MultiIndexSet generateNestedPoints(size_t num_threads, const MultiIndexSet &tensors, std::function<int(int)> getNumPoints){
    size_t num_dimensions = (size_t) tensors.getNumDimensions();
    std::vector<MultiIndexSet> delta_sets((size_t) tensors.getNumIndexes());

//    #pragma omp parallel for
//    for(int i=0; i<tensors.getNumIndexes(); i++){
    Utils::doParallel<int>((int) num_threads, tensors.getNumIndexes(), [&](int i)->void{
        Data2D<int> raw_points(num_dimensions, 0);

        std::vector<int> num_points_delta(num_dimensions);
        std::vector<int> offsets(num_dimensions);
        std::vector<int> index(num_dimensions);

        const int *p = tensors.getIndex(i);
        size_t num_total = 1;
        for(size_t j=0; j<num_dimensions; j++){
            num_points_delta[j] = getNumPoints(p[j]);
            if (p[j] > 0){
                offsets[j] = getNumPoints(p[j]-1);
                num_points_delta[j] -= offsets[j];
            }else{
                offsets[j] = 0;
            }

            num_total *= (size_t) num_points_delta[j];
        }

        for(size_t k=0; k<num_total; k++){
            size_t t = k;
            for(int j = (int) num_dimensions-1; j>=0; j--){
                index[j] = offsets[j] + (int) (t % num_points_delta[j]);
                t /= (size_t) num_points_delta[j];
            }
            raw_points.appendStrip(index);
        }

        delta_sets[i] = MultiIndexSet(raw_points);
    });
//    }

    return unionSets(delta_sets);
}

MultiIndexSet generateNonNestedPoints(size_t num_threads, const MultiIndexSet &tensors, const OneDimensionalWrapper &wrapper){
    size_t num_dimensions = tensors.getNumDimensions();
    int num_tensors = tensors.getNumIndexes();
    std::vector<MultiIndexSet> point_tensors((size_t) num_tensors);

//    #pragma omp parallel for
//    for(int t=0; t<num_tensors; t++){
    Utils::doParallel<int>((int) num_threads, num_tensors, [&](int t)->void{
        std::vector<int> num_entries(num_dimensions);
        const int *p = tensors.getIndex(t);
        std::transform(p, p + num_dimensions, num_entries.begin(), [&](int l)->int{ return wrapper.getNumPoints(l); });

        int num_total = 1;
        for(auto &l : num_entries) num_total *= l;

        Data2D<int> raw_points(num_dimensions, num_total);
        auto iter = raw_points.getVector().rbegin();
        for(int i=num_total-1; i>=0; i--){
            int d = i;
            auto l = num_entries.rbegin();
            // in order to generate indexes in the correct order, the for loop must go backwards
            for(size_t j = 0; j<num_dimensions; j++){
                *iter++ = wrapper.getPointIndex(p[num_dimensions - j - 1], (d % *l));
                d /= *l++;
            }
        }

        point_tensors[t] = MultiIndexSet(raw_points);
    });
//    }

    return unionSets(point_tensors);
}

std::vector<int> computeTensorWeights(size_t num_threads, MultiIndexSet const &mset){
    size_t num_dimensions = (size_t) mset.getNumDimensions();
    int num_tensors = mset.getNumIndexes();

    std::vector<int> level = computeLevels(num_threads, mset);
    int max_level = *std::max_element(level.begin(), level.end());

    Data2D<int> dag_down(num_dimensions, num_tensors);

    std::vector<int> weights((size_t) num_tensors);

//     #pragma omp parallel for schedule(static)
//     for(int i=0; i<num_tensors; i++){
    Utils::doParallel<int>((int) num_threads, num_tensors, [&](int i)->void{
        std::vector<int> kid(num_dimensions);
        std::copy_n(mset.getIndex(i), num_dimensions, kid.data());

        int *ref_kids = dag_down.getStrip(i);
        for(size_t j=0; j<num_dimensions; j++){
            kid[j]++;
            ref_kids[j] = mset.getSlot(kid);
            kid[j]--;
        }

        if (level[i] == max_level) weights[i] = 1;
    });
//    }

    for(int l=max_level-1; l>=0; l--){
//        #pragma omp parallel for schedule(dynamic)
//        for(int i=0; i<num_tensors; i++){
        Utils::doParallel<int, Utils::SyncParallel>((int) num_threads, num_tensors, [&](int i)->void{
            if (level[i] == l){
                std::vector<int> monkey_tail(max_level-l+1);
                std::vector<int> monkey_count(max_level-l+1);
                std::vector<bool> used(num_tensors, false);

                int current = 0;
                monkey_count[0] = 0;
                monkey_tail[0] = i;

                int sum = 0;

                while(monkey_count[0] < (int) num_dimensions){
                    if (monkey_count[current] < (int) num_dimensions){
                        int branch = dag_down.getStrip(monkey_tail[current])[monkey_count[current]];
                        if ((branch == -1) || (used[branch])){
                            monkey_count[current]++;
                        }else{
                            used[branch] = true;
                            sum += weights[branch];
                            monkey_count[++current] = 0;
                            monkey_tail[current] = branch;
                        }
                    }else{
                        monkey_count[--current]++;
                    }
                }

                weights[i] = 1 - sum;
            }
        });
//        }
    }

    return weights;
}

MultiIndexSet createPolynomialSpace(size_t num_threads, const MultiIndexSet &tensors, std::function<int(int)> exactness){
    size_t num_dimensions = (size_t) tensors.getNumDimensions();
    int num_tensors = tensors.getNumIndexes();
    std::vector<MultiIndexSet> polynomial_tensors((size_t) num_tensors);

//    #pragma omp parallel for
//    for(int i=0; i<num_tensors; i++){
    Utils::doParallel<int>((int) num_threads, num_tensors, [&](int i)->void{
        std::vector<int> npoints(num_dimensions);
        const int *p = tensors.getIndex(i);
        for(size_t j=0; j<num_dimensions; j++)
            npoints[j] = exactness(p[j]) + 1;
        polynomial_tensors[i] = generateFullTensorSet(npoints);
    });
//    }

    return unionSets(polynomial_tensors);
}

} // MultiIndexManipulations

} // TasGrid

#endif
