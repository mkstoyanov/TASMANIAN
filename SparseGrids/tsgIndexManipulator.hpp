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

#ifndef __TSG_INDEX_MANIPULATOR_HPP
#define __TSG_INDEX_MANIPULATOR_HPP

#include "tsgIndexSets.hpp"
#include "tsgOneDimensionalWrapper.hpp"

/*!
 * \internal
 * \file tsgIndexManipulator.hpp
 * \brief Algorithms for manipulating sets of multi-indexes.
 * \author Miroslav Stoyanov
 * \ingroup TasmanianMultiIndexManipulations
 *
 * A series of templates, lambda, and regular functions that allow the manipulation of
 * multi-indexes and sets of multi-indexes.
 * \endinternal
 */

/*!
 * \internal
 * \ingroup TasmanianSG
 * \addtogroup TasmanianMultiIndexManipulations Multi-Index manipulation algorithms
 *
 * \par Multi-Index manipulation algorithms
 * A series of templates, lambda, and regular functions that allow the manipulation of
 * multi-indexes and sets of multi-indexes. The algorithms include the selection of
 * multi-indexes according to a criteria, union of a vector of multi-index sets,
 * normalization of anisotropic weights, map parents and children within a set (which
 * generates a DAG structure), complete a set to satisfy the lower-property, etc.
 * \endinternal
 */

namespace TasGrid{

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Collection of algorithm to manipulate multi-indexes.
 */
namespace MultiIndexManipulations{

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Create a full-tensor multi-index set with \b num_entries in each direction.
 *
 * \endinternal
 */
inline MultiIndexSet generateFullTensorSet(std::vector<int> const &num_entries){
    size_t num_dimensions = num_entries.size();
    int num_total = 1;
    for(auto &l : num_entries) num_total *= l;
    std::vector<int> indexes(Utils::size_mult(num_dimensions, num_total));
    auto iter = indexes.rbegin();
    for(int i=num_total-1; i>=0; i--){
        int t = i;
        auto l = num_entries.rbegin();
        // in order to generate indexes in the correct order, the for loop must go backwards
        for(size_t j = 0; j<num_dimensions; j++){
            *iter++ = (t % *l);
            t /= *l++;
        }
    }
    return MultiIndexSet(num_dimensions, std::move(indexes));
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Generate the multi-index with entries satisfying the \b inside(), assumes that \b inside() defines a lower-complete set.
 * \endinternal
 */
inline MultiIndexSet generateLowerMultiIndexSet(size_t num_dimensions, std::function<bool(const std::vector<int> &index)> inside){
    size_t c = num_dimensions -1;
    bool is_in = true;
    std::vector<int> root(num_dimensions, 0);
    std::vector<int> indexes;
    while(is_in || (c > 0)){
        if (is_in){
            indexes.insert(indexes.end(), root.begin(), root.end());
            c = num_dimensions-1;
            root[c]++;
        }else{
            std::fill(root.begin() + c, root.end(), 0);
            root[--c]++;
        }
        is_in = inside(root);
    }
    return MultiIndexSet(num_dimensions, std::move(indexes));
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Expand the \b set with the minimum number of multi-indexes that will result in a lower complete set.
 * \endinternal
 */
void completeSetToLower(MultiIndexSet &set);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Holds anisotropic weights in a usable format.
 *
 * The external API accepts weigths as a single vector of integers or even an empty vector,
 * the \b ProperWeights holds weights that are always defined in a form suitable for computing.
 * \endinternal
 */
struct ProperWeights{
    //! \brief Interpret the single vector and form the proper weights, e.g., use defaults, scale hyperbolic weights, split \b type_ipcurved into linear and curved, etc.
    ProperWeights(size_t num_dimensions, TypeDepth type, std::vector<int> const &weights){
        contour = OneDimensionalMeta::getControurType(type);
        if (weights.empty()) linear = std::vector<int>(num_dimensions, 1);
        if (contour == type_level){ // linear weights only
            if (!weights.empty()){
                linear = weights; // simplest type
            }
        }else if (contour == type_curved){
            if (weights.empty()){
                curved = std::vector<double>(num_dimensions, 0.0); // cannot define default log-correction
            }else{
                linear = std::vector<int>(weights.begin(), weights.begin() + num_dimensions);
                curved = std::vector<double>(num_dimensions);
                std::transform(weights.begin() + num_dimensions, weights.end(), curved.begin(), [&](int i)->double{ return (double) i; });
            }
        }else{ // hyperbolic
            if (weights.empty()){
                curved = std::vector<double>(num_dimensions, 1.0);
            }else{
                linear = std::vector<int>(num_dimensions, 1); // used only for offset normalization
                double exponent_normalization = (double) *std::min_element(weights.begin(), weights.end());
                curved = std::vector<double>(num_dimensions);
                std::transform(weights.begin(), weights.end(), curved.begin(),
                               [&](int i)->double{ return ((double) i) / exponent_normalization; });
            }
        }
    }
    //! \brief Return \b true if the combination of weights and contour is guaranteed to give a lower-complete set.
    bool provenLower() const{
        if (contour == type_curved)
            for(size_t i=0; i<linear.size(); i++)
                if ((double)linear[i] + curved[i] < 0) return false;
        return true;
    }
    //! \brief Return the smallest linear weight, used for normalization of the offset.
    int minLinear() const{ return *std::min_element(linear.begin(), linear.end()); }
    //! \brief Return the number of dimensions.
    size_t getNumDimensions() const{ return linear.size(); }

    //! \brief Always equal to \b type_level, \b type_curved or \b type_hyperbolic, indicates the general contour and how the weights should be used.
    TypeDepth contour;
    //! \brief The linear components of the weights for \b type_level and \b type_curved contours; a vector of ones for \b type_hyperbolic.
    std::vector<int> linear;
    //! \brief The curved components of the weights \b type_curved contours; a scaled vector of exponents for \b type_hyperbolic; empty for \b type_level.
    std::vector<double> curved;
};

/*!
 * \internal
 * \brief Generate weights cache for the given parameters.
 *
 * Complexity chosen here in favor of performance and re-usability.
 * - \b isotropic indicates whether to treat the \b offset as a maximum cache size in each direction (\b true)
 *   of as a cut-off for the weights, i.e., cache until the 1-D weight exceeds the offset.
 * - \b weights.contour must match \b contour
 * - if \b contour is \b type_level, then \b CacheType must be \b int
 * - if \b contour is \b type_curved or \b type_hyperbolic, then \b CacheType must be \b double
 * \endinternal
 */
template<typename CacheType, TypeDepth contour, bool isotropic>
std::vector<std::vector<CacheType>> generateLevelWeightsCache(ProperWeights const &weights, std::function<int(int i)> rule_exactness, int offset){
    size_t num_dimensions = weights.getNumDimensions();
    std::vector<std::vector<CacheType>> cache(num_dimensions);
    std::vector<int> exactness_cache;

    // if we know the sized here, then assign, otherwise dynamically build later
    if (isotropic){
        for(auto &vec : cache) vec.reserve((size_t) offset);
        exactness_cache.resize((size_t) offset);
        exactness_cache[0] = 0;
        for(int i=1; i<offset; i++)
            exactness_cache[i] = 1 + rule_exactness(i - 1);
    }else{
        exactness_cache.push_back(0); // exactness always starts from 0
    }

    for(size_t j=0; j<num_dimensions; j++){
        int wl = weights.linear[j]; // linear weights
        double wc = (contour == type_level) ? 0.0 : weights.curved[j]; // curved weights

        size_t i = 0; // keep track of the index
        CacheType w = (CacheType) ((contour == type_hyperbolic) ? 1 : 0);
        cache[j].push_back(w); // initial entry
        do{ // accumulate the cache
            i++;
            if (!isotropic && (i >= exactness_cache.size()))
                exactness_cache.push_back(1 + rule_exactness((int)(i - 1)));

            int e = exactness_cache[i];

            if (contour == type_level){
                w = (CacheType)(wl * e);
            }else if (contour == type_curved){
                w = (CacheType)(wl * e) + (CacheType)(wc * std::log1p((CacheType)e));
            }else{ // must be hyperbolic
                w = (CacheType)(pow((CacheType) (1 + e), wc));
            }

            cache[j].push_back(w);
        // isotropic mode works until offset, anisotropic mode works until weight reaches the offset
        }while( (!isotropic && (std::ceil(w) <= (CacheType) offset)) || (isotropic && (i + 1 < (size_t) offset)) );
    }

    return cache;
}

/*!
 * \internal
 * \brief Returns the weight for the multi-index using the cache, assuming level contour.
 *
 * \endinternal
 */
template<typename CacheType, TypeDepth contour>
inline CacheType getIndexWeight(int const index[], std::vector<std::vector<CacheType>> const &cache){
    CacheType w = (contour == type_hyperbolic) ? 1 : 0;
    for(size_t j=0; j<cache.size(); j++)
        if (contour == type_hyperbolic)
            w *= cache[j][index[j]];
        else
            w += cache[j][index[j]];
    return w;
}

/*!
 * \internal
 * \brief Generate a lower complete multi-index set that satisfies the given properties.
 *
 * Using the \b num_dimensions select all multi-indexes with weight less than the \b offset.
 * If not empty, \b anisotropic_weights are used to form the weight and the offset is scaled by the minimum linear weight.
 * Rule exactness is used in place of the raw-index to cover the correct polynomial space, see references in \b TypeDepth.
 * \endinternal
 */
MultiIndexSet selectTensors(size_t num_dimensions, int offset, TypeDepth type, std::function<int(int i)> rule_exactness,
                            std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Returns a vector that is the sum of entries of each multi-index in the set.
 *
 * \endinternal
 */
std::vector<int> computeLevels(size_t num_threads, MultiIndexSet const &mset);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Returns a vector with the maximum index in each dimension.
 *
 * \endinternal
 */
std::vector<int> getMaxIndexes(const MultiIndexSet &mset);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Returns a Data2D structure where each strip holds the slot-index of the parents of indexes in \b mset (for each direction), using one-point-growth hierarchy.
 *
 * Adds -1 in places where the parents are missing.
 * \endinternal
 */
Data2D<int> computeDAGup(size_t num_threads, MultiIndexSet const &mset);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Using the \b flagged map, create a set with the flagged children of \b mset but only if they obey the \b level_limits.
 *
 * \endinternal
 */
MultiIndexSet selectFlaggedChildren(const MultiIndexSet &mset, const std::vector<bool> &flagged, const std::vector<int> &level_limits);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Converts a set of nested \b tensors to the actual points, where each level has \b getNumPoints() in each direction.
 *
 * Working with nested points, it is more efficient to interpret the tensors as a surplus operators and generate only the surplus points;
 * then take the union without repeated indexes.
 * This can work if and only if \b tensors is a lower-complete set, i.e., the \b tensors in \b GridGlobal and not the active_tensors.
 * \endinternal
 */
MultiIndexSet generateNestedPoints(size_t num_threads, const MultiIndexSet &tensors, std::function<int(int)> getNumPoints);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Converts a set of non-nested active \b tensors to the actual points, the \b wrapper gives mapping between level and point indexes.
 *
 *  For each index in \b tensor, generate a local set of points and then remap them to the global index using the \b wrapper.getPointIndex().
 * \endinternal
 */
MultiIndexSet generateNonNestedPoints(size_t num_threads, const MultiIndexSet &tensors, const OneDimensionalWrapper &wrapper);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Find the indexes of all points associated with the tensor with \b levels within the global \b points set.
 *
 * The order in which the tensors are formed here must match the order in which they will be used,
 * e.g., when computing interpolation and quadrature weights.
 */
template<bool nested>
void referencePoints(const int levels[], const OneDimensionalWrapper &wrapper, const MultiIndexSet &points, std::vector<int> &refs){
    size_t num_dimensions = (size_t) points.getNumDimensions();
    std::vector<int> num_points(num_dimensions);
    int num_total = 1; // this will be a subset of all points, no danger of overflow
    for(size_t j=0; j<num_dimensions; j++) num_points[j] = wrapper.getNumPoints(levels[j]);
    for(auto n : num_points) num_total *= n;

    refs.resize(num_total);
    std::vector<int> p(num_dimensions);

    for(int i=0; i<num_total; i++){
        int t = i;
        auto n = num_points.rbegin();
        for(int j=(int) num_dimensions-1; j>=0; j--){
            p[j] = (nested) ? t % *n : wrapper.getPointIndex(levels[j], t % *n);
            t /= *n++;
        }
        refs[i] = points.getSlot(p);
    }
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Computes the weights for the tensor linear combination, \b mset must be a lower complete set.
 *
 * \endinternal
 */
std::vector<int> computeTensorWeights(size_t num_threads, MultiIndexSet const &mset);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Creates a set containing only the entries of \b mset that have non-zero \b weights.
 *
 * \endinternal
 */
inline MultiIndexSet createActiveTensors(const MultiIndexSet &mset, const std::vector<int> &weights){
    size_t num_dimensions = mset.getNumDimensions();
    size_t nz_weights = 0;
    for(auto w: weights) if (w != 0) nz_weights++;

    std::vector<int> indexes(nz_weights * num_dimensions);
    nz_weights = 0;
    auto iter = indexes.begin();
    auto iset = mset.getVector().begin();
    for(auto w: weights){
        if (w != 0){
            std::copy_n(iset, num_dimensions, iter);
            std::advance(iter, num_dimensions);
        }
        std::advance(iset, num_dimensions);
    }

    return MultiIndexSet(num_dimensions, std::move(indexes));
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief For a set of \b tensors compute the corresponding polynomial space assuming the 1D rules have given \b exactness().
 *
 * \endinternal
 */
MultiIndexSet createPolynomialSpace(size_t num_threads, const MultiIndexSet &tensors, std::function<int(int)> exactness);

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Assuming that \b mset is lower complete, return \b true if adding the \b point will preserve completeness.
 *
 * \endinternal
 */
inline bool isLowerComplete(std::vector<int> const &point, MultiIndexSet const &mset){
    auto dad = point;
    for(auto &d : dad){
        if (d > 0){
            d--;
            if (mset.missing(dad)) return false;
            d++;
        }
    }
    return true;
}

/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief Return the largest subset of \b candidates such that adding it to \b current will preserve lower completeness.
 *
 * \endinternal
 */
inline MultiIndexSet getLargestCompletion(MultiIndexSet const &current, MultiIndexSet const &candidates){
    if (candidates.empty()) return MultiIndexSet();
    auto num_dimensions = candidates.getNumDimensions();
    MultiIndexSet result; // start with an empty set
    if (current.empty()){
        if (candidates.missing(std::vector<int>(num_dimensions, 0))){
            return MultiIndexSet(); // current is empty, 0-th index is the only thing that can be added
        }else{
            result = MultiIndexSet(num_dimensions, std::vector<int>(num_dimensions, 0));
        }
    }
    bool loopon = true;
    while(loopon){
        Data2D<int> update(num_dimensions, 0);

        MultiIndexSet total = current;
        if (!result.empty()) total.addMultiIndexSet(result);
        for(int i=0; i<total.getNumIndexes(); i++){
            std::vector<int> kid(total.getIndex(i), total.getIndex(i) + num_dimensions);
            for(auto &k : kid){
                k++; // construct the kid in the new direction
                if (!candidates.missing(kid) && result.missing(kid) && isLowerComplete(kid, total))
                    update.appendStrip(kid);
                k--;
            }
        }

        loopon = (update.getNumStrips() > 0);
        if (loopon) result.addMultiIndexSet(MultiIndexSet(update));
    }
    return result;
}


/*!
 * \internal
 * \ingroup TasmanianMultiIndexManipulations
 * \brief For a set of \b tensors create an \b mset that contain the children of indexes in \b tensors that are missing from \b exclude and obey the \b level_limits.
 *
 * If \b limited is \b false, then the \b level_limits are ignored.
 * \endinternal
 */
template<bool limited>
MultiIndexSet addExclusiveChildren(const MultiIndexSet &tensors, const MultiIndexSet &exclude, const std::vector<int> level_limits){
    int num_dimensions = (int) tensors.getNumDimensions();
    Data2D<int> tens(num_dimensions, 0);
    for(int i=0; i<tensors.getNumIndexes(); i++){ // add the new tensors (so long as they are not included in the initial grid)
        const int *t = tensors.getIndex(i);
        std::vector<int> kid(t, t + num_dimensions);
        auto ilimit = level_limits.begin();
        for(auto &k : kid){
            k++;
            if (exclude.missing(kid) && tensors.missing(kid)){ // if the kid is not to be excluded and if not included in the current set
                if (isLowerComplete(kid, tensors)){
                    if (limited){
                        if ((*ilimit == -1) || (k <= *ilimit))
                            tens.appendStrip(kid);
                        ilimit++;
                    }else{
                        tens.appendStrip(kid);
                    }
                }
            }
            k--;
        }
    }

    return MultiIndexSet(tens);
}

}

}

#endif
