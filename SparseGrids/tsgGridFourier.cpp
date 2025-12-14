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

#ifndef __TASMANIAN_SPARSE_GRID_FOURIER_CPP
#define __TASMANIAN_SPARSE_GRID_FOURIER_CPP

#include "tsgGridFourier.hpp"
#include "tsgTPLWrappers.hpp"

namespace TasGrid{

template<bool iomode> void GridFourier::write(std::ostream &os) const{
    if (iomode == mode_ascii){ os << std::scientific; os.precision(17); }
    IO::writeNumbers<iomode, IO::pad_line>(os, num_dimensions, num_outputs);

    tensors.write<iomode>(os);
    active_tensors.write<iomode>(os);
    if (!active_w.empty())
        IO::writeVector<iomode, IO::pad_line>(active_w, os);

    IO::writeFlag<iomode, IO::pad_auto>(!points.empty(), os);
    if (!points.empty()) points.write<iomode>(os);
    IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
    if (!needed.empty()) needed.write<iomode>(os);

    IO::writeVector<iomode, IO::pad_line>(max_levels, os);

    if (num_outputs > 0){
        values.write<iomode>(os);
        IO::writeFlag<iomode, IO::pad_auto>((fourier_coefs.getNumStrips() != 0), os);
        if (!fourier_coefs.empty()) fourier_coefs.writeVector<iomode, IO::pad_line>(os);
    }

    IO::writeFlag<iomode, IO::pad_line>(!updated_tensors.empty(), os);
    if (!updated_tensors.empty()){
        updated_tensors.write<iomode>(os);
        updated_active_tensors.write<iomode>(os);
        IO::writeVector<iomode, IO::pad_line>(updated_active_w, os);
    }
}

template void GridFourier::write<mode_ascii>(std::ostream &) const;
template void GridFourier::write<mode_binary>(std::ostream &) const;

void GridFourier::makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    setTensors(selectTensors((size_t) cnum_dimensions, depth, type, anisotropic_weights, level_limits), cnum_outputs);
}

GridFourier::GridFourier(AccelerationContext const *acc, GridFourier const *fourier, int ibegin, int iend) :
    BaseCanonicalGrid(acc, *fourier, ibegin, iend),
    wrapper(fourier->wrapper),
    tensors       (fourier->tensors),
    active_tensors(fourier->active_tensors),
    active_w      (fourier->active_w),
    updated_tensors       (fourier->updated_tensors),
    updated_active_tensors(fourier->updated_active_tensors),
    updated_active_w      (fourier->updated_active_w),
    max_levels(fourier->max_levels),
    fourier_coefs((num_outputs == fourier->num_outputs) ? fourier->fourier_coefs : fourier->fourier_coefs.splitData(ibegin, iend)),
    max_power(fourier->max_power){

    if (fourier->dynamic_values){
        dynamic_values = std::make_unique<DynamicConstructorDataGlobal>(*fourier->dynamic_values);
        if (num_outputs != fourier->num_outputs) dynamic_values->restrictData(ibegin, iend);
    }
}

void GridFourier::updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    if ((num_outputs == 0) || points.empty()){
        makeGrid(num_dimensions, num_outputs, depth, type, anisotropic_weights, level_limits);
    }else{
        clearRefinement();

        updated_tensors = selectTensors((size_t) num_dimensions, depth, type, anisotropic_weights, level_limits);

        MultiIndexSet new_tensors = updated_tensors - tensors;

        if (!new_tensors.empty()){
            updated_tensors += tensors;
            proposeUpdatedTensors();
        }
    }
}

MultiIndexSet GridFourier::selectTensors(size_t dims, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights,
                                        std::vector<int> const &level_limits) const{
    return (OneDimensionalMeta::isExactLevel(type)) ?
        MultiIndexManipulations::selectTensors(dims, depth, type,
                                               [&](int i) -> int{ return i; }, anisotropic_weights, level_limits) :
        MultiIndexManipulations::selectTensors(dims, depth, type,
                                               [&](int i) -> int{ return OneDimensionalMeta::getIExact(i, rule_fourier); }, anisotropic_weights, level_limits);
}

void GridFourier::setTensors(MultiIndexSet &&tset, int cnum_outputs){
    clearGpuNodes();
    clearGpuCoefficients();
    points = MultiIndexSet();
    values = StorageSet();
    active_w.clear();
    fourier_coefs.clear();

    tensors = std::move(tset);

    num_dimensions = (int) tensors.getNumDimensions();
    num_outputs = cnum_outputs;

    max_levels = MultiIndexManipulations::getMaxIndexes(tensors);

    wrapper = OneDimensionalWrapper(*std::max_element(max_levels.begin(), max_levels.end()), rule_fourier, 0.0, 0.0);

    MultiIndexManipulations::computeActiveTensorsWeights(tensors, active_tensors, active_w);

    needed = MultiIndexManipulations::generateNestedPoints(tensors, [&](int l) -> int{ return wrapper.getNumPoints(l); });

    if (num_outputs == 0){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        values.resize(num_outputs, needed.getNumIndexes());
    }

    max_power = MultiIndexManipulations::getMaxIndexes(((points.empty()) ? needed : points));
}

void GridFourier::proposeUpdatedTensors(){
    wrapper = OneDimensionalWrapper(updated_tensors.getMaxIndex(), rule_fourier, 0.0, 0.0);

    MultiIndexManipulations::computeActiveTensorsWeights(updated_tensors, updated_active_tensors, updated_active_w);

    needed = MultiIndexManipulations::generateNestedPoints(updated_tensors, [&](int l) -> int{ return wrapper.getNumPoints(l); })
             - points;
}

void GridFourier::acceptUpdatedTensors(){
    if (points.empty()){
        clearGpuNodes(); // the points and needed will change, clear the cache
        points = std::move(needed);
        needed = MultiIndexSet();
    }else if (!needed.empty()){
        points += needed;
        needed = MultiIndexSet();

        tensors = std::move(updated_tensors);
        updated_tensors = MultiIndexSet();

        active_tensors = std::move(updated_active_tensors);
        updated_active_tensors = MultiIndexSet();

        active_w = std::move(updated_active_w);
        updated_active_w = std::vector<int>();

        max_levels = MultiIndexManipulations::getMaxIndexes(tensors);
    }
}

void GridFourier::loadNeededValues(const double *vals){
    clearGpuCoefficients(); // changing values and Fourier coefficients, clear the cache
    if (points.empty() || needed.empty()){
        values.setValues(vals);
    }else{
        values.addValues(points, needed, vals);
    }

    acceptUpdatedTensors();
    calculateFourierCoefficients();

    max_power = MultiIndexManipulations::getMaxIndexes(points);
}

void GridFourier::getLoadedPoints(double *x) const{
    MultiIndexManipulations::indexesToNodes(points, wrapper, x);
}
void GridFourier::getNeededPoints(double *x) const{
    MultiIndexManipulations::indexesToNodes(needed, wrapper, x);
}
void GridFourier::getPoints(double *x) const{
    if (points.empty()){ getNeededPoints(x); }else{ getLoadedPoints(x); };
}

std::vector<std::vector<int>> GridFourier::generateIndexingMap() const{
    // The internal point-indexing of Tasmanian goes 0, 1/3, 2/3, 1/9, 2/9, 4/9 ....
    // Fourier transform (and coefficients) need spacial order 0, 1/9, 2/9, 3/9=1/3, ...
    // Create a map, where at level 0: 0 -> 0, level 1: 0 1 2 -> 0 1 2, level 2: 0 1 2 3 4 5 6 7 8 -> 0 3 4 1 5 6 2 7 8
    // The map takes a point from previous map and adds two more points ...
    // Thus, a spacial point i on level l is Tasmanian point index_map[l][i]
    int maxl = 1 + active_tensors.getMaxIndex();
    std::vector<std::vector<int>> index_map(maxl);
    index_map[0].resize(1, 0);
    int c = 1;
    for(int l=1; l<maxl; l++){
        index_map[l].resize(3*c); // next level is 3 times the size of the previous
        auto im = index_map[l].begin();
        for(auto i: index_map[l-1]){
            *im++ = i; // point from old level
            *im++ = c++; // two new points
            *im++ = c++;
        }
    }
    return index_map;
}

void GridFourier::calculateFourierCoefficients(){
    // There are three indexing schemes needed here
    // First, is the way nodes are indexed and stored in the "work" IndexSet (same as all other nested grids)
    //       the order is contiguous in level, meaning that node indexed by i is always the same node regardless of level
    //       and all nodes on level l are indexed from 0 till 3^l
    // Second, is the spacial order needed by the Fourier transform
    //       nodes have to appear left to right in order for the transform to work so reindexing has to be done
    //       see generateIndexingMap() for index detail
    //       reindexing is done when all data to for the tensor is put into one data structure
    // Third, the exponents of the basis functions have to be indexed and some functions will have negative exponents
    //       following the same idea as the first indexing, the exponents are best used contiguously
    //       reorder the exponents so that 0, 1, 2, 3, 4 ... map to exponents 0, -1, 1, -2, 2, -3, 3 ...
    //       the formula is exponent = (point + 1) / 2, if point is odd, make exponent negative
    // The (point + 1) / 2 maps First indexing to Third indexing, generateIndexingMap() takes care of First -> Second
    // The Second -> Third indexing is done per-tensor, where the non-negative exponents are associated with coefficients
    //     going from left to right (in the order of the Fourier coefficients), while the negative coefficients go
    //     in reverse right to left order "int rj = (p[j] % 2 == 0) ? (p[j]+1) / 2 : num_oned_points[j] - (p[j]+1) / 2;"
    int num_points = getNumPoints();

    MultiIndexSet &work = (points.empty()) ? needed : points;
    std::vector<std::vector<int>> index_map = generateIndexingMap();

    fourier_coefs = Data2D<double>(num_outputs, 2 * num_points);

    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        const int* levels = active_tensors.getIndex(n);
        int num_tensor_points = 1;
        std::vector<int> num_oned_points(num_dimensions);
        std::vector<int> cnum_oned_points(num_dimensions, 1); // cumulative number of points
        for(int j=0; j<num_dimensions; j++){
            num_oned_points[j] = wrapper.getNumPoints(levels[j]);
            num_tensor_points *= num_oned_points[j];
        }
        for(int j=num_dimensions-2; j>=0; j--) cnum_oned_points[j] = num_oned_points[j+1] * cnum_oned_points[j+1];

        std::vector<std::vector<std::complex<double>>> tensor_data(num_tensor_points);
        std::vector<int> p(num_dimensions);
        for(int i=0; i<num_tensor_points; i++){
            // We interpret this "i" as running through the spatial indexing; convert to internal
            int t=i;
            for(int j=num_dimensions-1; j>=0; j--){
                // here p[] is the Tasmanian index of index from real space i (t % num_oned_points[j])
                p[j] = index_map[levels[j]][t % num_oned_points[j]];
                t /= num_oned_points[j];
            }
            //refs[i] = ; // refs[i] is the index of Tasmanian (index set) corresponding to real index "i"
            double const *v = values.getValues(work.getSlot(p));
            tensor_data[i] = std::vector<std::complex<double>>(v, v + num_outputs);
        }

        TasmanianFourierTransform::fast_fourier_transform(tensor_data, num_oned_points);

        double tensorw = ((double) active_w[n]) / ((double) num_tensor_points);
        for(int i=0; i<num_tensor_points; i++){
            int t = i;
            int r = 0; // holds the real index corresponding to the power
            for(int j=num_dimensions-1; j>=0; j--){
                // here rj is the real multi-index corresponding to "i"
                p[j] = t % num_oned_points[j];
                int rj = (p[j] % 2 == 0) ? (p[j]+1) / 2 : num_oned_points[j] - (p[j]+1) / 2; // +/- index
                r += rj * cnum_oned_points[j];
                t /= num_oned_points[j];
            }
            t = work.getSlot(p); // holds the Tasmanian index corresponding to real index p

            // Combine with tensor weights
            double *fc_real = fourier_coefs.getStrip(t);
            double *fc_imag = fourier_coefs.getStrip(t + num_points);

             for(auto d : tensor_data[r]){
                 *fc_real++ += tensorw * d.real();
                 *fc_imag++ += tensorw * d.imag();
             }
        }
    }
}

void GridFourier::getInterpolationWeights(const double x[], double weights[]) const {
    // if Fourier coefficient are c, Data from the target function is f, and values of the basis functions are b
    // then we have c = A * f (where A is both the Fourier transform and the reindexing)
    // and the value of the interpolant is result = <c, b> = <A f, b> = <f, A^* b>, where A^* is conjugate transpose
    // However, we consider only the real values (the complex ones add-up to zero), thus we really need A^T (regular transpose)
    // The Fourier transform is symmetric with respect to regular transpose, which leaves only the indexing
    // Take the basis functions, reindex and reorder to a data strucutre, take FFT, reindex and reorder into the weights
    //
    //
    // This code uses a O(N) algorithm instead of an O(N log N) FFT. On a 1D tensor with N=3^l points, we must compute the
    // Fourier transform of
    //      {x_n} = e^(2 \pi i x * {0,1,2, ..., (N-1)/2, -(N-1)/2, ..., -2, -1}).           (componentwise operations)
    // The FFT is
    //      X[m] = \sum_{j=0}^{N-1} e^{-2 \pi i j m / N} x_n
    //           = ( \sum_{j=0}^{(N-1)/2} e^{-2 \pi i j m / N} e^{2 \pi i x j} )
    //                + ( \sum_{j=1}^{(N-1)/2} e^{2 \pi i j m / N} e^{-2 \pi i x j} )
    //           = 2 * Re[ \sum_{j=0}^{(N-1)/2} e^{-2 \pi i j m / N} e^{2 \pi i x j} ] - 1
    //           = 2 * Re[ \sum_{j=0}^{(N-1)/2} (e^{2 \pi i (x-m/N)})^j ] - 1
    //           = 2 * Re[ \frac{1 - e^{2 \pi i (x-m/N) (N+1)/2}}{1 - e^{2 \pi i (x-m/N)}} ] - 1
    // The cost is mainly driven by evaluating the complex exponentials, so we compute what we can at the beginning and
    // park it in a cache.

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    std::vector<std::vector<int>> index_map = generateIndexingMap();

    std::fill_n(weights, work.getNumIndexes(), 0.0);

    // compute what we need for e^{-2 \pi i m / N}
    int maxl = active_tensors.getMaxIndex() + 1;
    std::vector<std::vector<std::complex<double>>> expcache(maxl);
    for(int i=0; i<maxl; i++){
        int num_oned_points = wrapper.getNumPoints(i);
        expcache[i].resize(num_oned_points);
        expcache[i][0] = std::complex<double>(1.0, 0.0);
        double theta = -2.0 * Maths::pi / ((double) num_oned_points);       // step angle
        std::complex<double> step(std::cos(theta), std::sin(theta));
        for(int j=1; j<num_oned_points; j++) expcache[i][j] = expcache[i][j-1] * step;
    }

    // compute what we need for e^{2 \pi i x (N+1)/2}
    std::vector<std::vector<std::complex<double>>> numerator_cache(num_dimensions);
    for(int k=0; k<num_dimensions; k++){
        numerator_cache[k].resize(max_levels[k]+1);
        double theta = 2.0 * Maths::pi * x[k];
        numerator_cache[k][0] = std::complex<double>(std::cos(theta), std::sin(theta));
        for(int j=1; j<max_levels[k]+1; j++){
            numerator_cache[k][j] = numerator_cache[k][j-1];
            for(int i=0; i<wrapper.getNumPoints(j-1); i++) numerator_cache[k][j] *= numerator_cache[k][0];
        }
    }

    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        const int *levels = active_tensors.getIndex(n);
        int num_tensor_points = 1;
        std::vector<int> num_oned_points(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            num_oned_points[j] = wrapper.getNumPoints(levels[j]);
            num_tensor_points *= num_oned_points[j];
        }
        std::vector<int> p(num_dimensions);

        double tensorw = ((double) active_w[n]) / ((double) num_tensor_points);
        for(int i=0; i<num_tensor_points; i++){
            // We interpret this "i" as running through the spatial indexing; convert to internal
            int t=i;
            double fftprod = 1.0;
            for(int j=num_dimensions-1; j>=0; j--){ // here p is the index of the spacial point in Tasmanian indexing
                int r = t % num_oned_points[j];
                int offset = (r*(num_oned_points[j]+1)/2) % num_oned_points[j];     // in order to fetch reduced form of (N+1)*r/(2*N)

                if (std::abs(1.0 - (numerator_cache[j][0] * expcache[levels[j]][r]).real()) <  Maths::num_tol){
                    // we're evaluating the basis functions at a node; take care of zero-divide
                    fftprod *= num_oned_points[j];
                }else{
                    fftprod *= 2.0 * ( (1.0 - numerator_cache[j][levels[j]] * expcache[levels[j]][offset])
                                / (1.0 - numerator_cache[j][0] * expcache[levels[j]][r]) ).real() - 1.0;
                }

                p[j] = index_map[levels[j]][r];
                t /= num_oned_points[j];
            }
            weights[work.getSlot(p)] += (tensorw * fftprod);
        }
    }
}

void GridFourier::getQuadratureWeights(double weights[]) const{
    // When integrating the Fourier series on a tensored grid, all the
    // nonzero modes vanish, and we're left with the normalized Fourier
    // coeff for e^0 (sum of the data divided by number of points)

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    std::fill_n(weights, work.getNumIndexes(), 0.0);

    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        const int *levels = active_tensors.getIndex(n);
        int num_tensor_points = 1;
        for(int j=0; j<num_dimensions; j++){
            num_tensor_points *= wrapper.getNumPoints(levels[j]);
        }
        std::vector<int> refs = MultiIndexManipulations::referencePoints<true>(levels, wrapper, work);

        double tensorw = ((double) active_w[n]) / ((double) num_tensor_points);
        for(int i=0; i<num_tensor_points; i++){
            weights[refs[i]] += tensorw;
        }
    }
}

void GridFourier::getDifferentiationWeights(const double x[], double weights[]) const {
    /* Our goal is to differentiate the function A^* b(x) where A is the DFT and b(x) are the basis functions evaluated at x.
     * The mathematics below is based on the derivation in getInterpolationWeights().
     *
     * For the 1D-tensor case, if x = m/N for some -(N-1)/2 <= m <= (N-1)/2, then X[m] = N and dX[m]/dx = 0. Otherwise, we have
     * X[l] = 2.0 * ℜ[u(x)] - 1.0, where
     *
     *   u(x) = {1 - exp[I * a(x)]} / {1 - exp[I * b(x)])},
     *   a(x) = 2 * π * (x - m / N) * (N + 1) /2,
     *   b(x) = 2 * π * (x - m / N),
     *
     * and I^2 = -1. Note that
     *
     *   |1 - exp[I * b(x)]|^2 * u(x) = {1 - exp[I * a(x)]} * {1 - exp[-I * b(x)])}
     *                                = 1 - exp[I * a(x)] - exp[-I * b(x)] + exp[I * {a(x) - b(x)}].
     *
     * Since |1 - exp[I * b(x)]|^2 == 2 - 2 * cos[b(x)], we have the expressions
     *
     *   ℜ[u(x)] = (1 - cos[a(x)] - cos[b(x)] + cos[a(x) - b(x)]) * (2 - cos[b(x)]) ^ {-1},
     *
     *   d{ℜ[u(x)]}/dx = (  {a'(x) * sin[a(x)] + b'(x) * sin[b(x)] - [a'(x) - b'(x)] * sin[a(x) - b(x)]} * {2 - 2 * cos[b(x)]} -
     *                      {1 - cos[a(x)] - cos[b(x)] + cos[a(x) - b(x)]} * {2 * b'(x) * sin[b(x)]}
     *                   ) * (2 - 2 * cos[b(x)]) ^ {-2},
     *
     * which then gives us dX[l]/dx = 2.0 * d{ℜ[u(x)]}/dx. To save computational effort, we apply the same caching strategy as
     * in getInterpolationWeights().
     */
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    std::vector<std::vector<int>> index_map = generateIndexingMap();
    std::fill_n(weights, work.getNumIndexes(), 0.0);

    // Cache exp(-2 * π * I * m / 3^{l}) for every (m, l) where N = 3^l.
    int maxl = active_tensors.getMaxIndex() + 1;
    std::vector<std::vector<std::complex<double>>> shift_cache(maxl);
    for(int i=0; i<maxl; i++){
        int num_oned_points = wrapper.getNumPoints(i);
        shift_cache[i].resize(num_oned_points);
        shift_cache[i][0] = std::complex<double>(1.0, 0.0);
        double theta = -2.0 * Maths::pi / ((double) num_oned_points);
        std::complex<double> step(std::cos(theta), std::sin(theta));
        for(int j=1; j<num_oned_points; j++) shift_cache[i][j] = shift_cache[i][j-1] * step;
    }

    // Cache exp(-2 * π * x[k] * [3^{l} + 1] / 2) for every (k, l) where N = 3^l.
    std::vector<std::vector<std::complex<double>>> slope_cache(num_dimensions);
    for(int k=0; k<num_dimensions; k++){
        slope_cache[k].resize(max_levels[k]+1);
        double theta = 2.0 * Maths::pi * x[k];
        slope_cache[k][0] = std::complex<double>(std::cos(theta), std::sin(theta));
        for(int j=1; j<max_levels[k]+1; j++){
            slope_cache[k][j] = slope_cache[k][j-1];
            for(int i=0; i<wrapper.getNumPoints(j-1); i++) slope_cache[k][j] *= slope_cache[k][0];
        }
    }

    // This lambda returns the FFT value for the interpolant in dimension d, level l, # of points N, and local
    // offset (within the level) r. Based on getInterpolationWeights().
    auto get_fft_val = [&](const int d, const int l, int N, int r) {
        std::complex<double> exp_Ibx = slope_cache[d][0] * shift_cache[l][r];
        if (std::abs(1.0 - exp_Ibx.real()) <= Maths::num_tol)
            return (double) N;
        int offset = (r * (N + 1) / 2) % N;
        std::complex<double> exp_Iax = slope_cache[d][l] * shift_cache[l][offset];
        return 2.0 * ((1.0 - exp_Iax) / (1.0 - exp_Ibx)).real() - 1.0;
     };

    // This lambda is the same as get_fft_val(), but returns the FFT derivative.
    auto get_fft_diff_val = [&](const int d, const int l, int N, int r) {
        std::complex<double> exp_Ibx = slope_cache[d][0] * shift_cache[l][r];
        if (std::abs(1.0 - exp_Ibx.real()) <= Maths::num_tol)
            return 0.0;
        double N_dbl = (double) N;
        double slope_b = 2.0 * Maths::pi;
        double slope_a = Maths::pi * (N_dbl + 1.0);
        int offset = (r * (N + 1) / 2) % N;
        std::complex<double> exp_Iax = slope_cache[d][l] * shift_cache[l][offset];
        std::complex<double> exp_Ia_sub_Ib = exp_Iax * std::conj(exp_Ibx);
        // q1 below is equivalent to [2.0 - 2.0 * cos(b)], but is more computationally stable since it combines
        // both real and imaginary parts.
        double q0 = std::abs(1.0 - exp_Ibx);
        double q1 = q0 * q0;
        double s1 = (slope_a * exp_Iax.imag() + slope_b * exp_Ibx.imag() - (slope_a - slope_b) * exp_Ia_sub_Ib.imag()) * q1;
        double s2 = (1.0 - exp_Iax.real() - exp_Ibx.real() + exp_Ia_sub_Ib.real()) * 2.0 * slope_b * exp_Ibx.imag();
        return 2.0 * (s1 - s2) / (q1 * q1);
   };

    // Compute the weights for each tensor.
    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        const int *levels = active_tensors.getIndex(n);
        int num_tensor_points = 1;
        std::vector<int> num_oned_points(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            num_oned_points[j] = wrapper.getNumPoints(levels[j]);
            num_tensor_points *= num_oned_points[j];
        }
        std::vector<int> p(num_dimensions);
        double tensorw = ((double) active_w[n]) / ((double) num_tensor_points);
        for(int i=0; i<num_tensor_points; i++){
            int t = i;
            std::vector<double> fft_vals(num_dimensions);
            std::vector<double> fft_diff_vals(num_dimensions);
            for(int j=num_dimensions-1; j>=0; j--) {
                int r = t % num_oned_points[j];
                fft_vals[j] = get_fft_val(j, levels[j], num_oned_points[j], r);
                fft_diff_vals[j] = get_fft_diff_val(j, levels[j], num_oned_points[j], r);
                p[j] = index_map[levels[j]][r];
                t /= num_oned_points[j];
            }
            for(int k=0; k<num_dimensions; k++) {
                double fftprod = 1.0;
                for(int j=0; j<k; j++) fftprod *= fft_vals[j];
                fftprod *= fft_diff_vals[k];
                for(int j=k+1; j<num_dimensions; j++) fftprod *= fft_vals[j];
                weights[work.getSlot(p) * num_dimensions + k] += tensorw * fftprod;
            }
        }
    }
}

void GridFourier::evaluate(const double x[], double y[]) const{
    int num_points = points.getNumIndexes();
    std::fill_n(y, num_outputs, 0.0);
    std::vector<double> wreal(num_points);
    std::vector<double> wimag(num_points);
    computeBasis<double, false>(points, x, wreal.data(), wimag.data());
    for(int i=0; i<num_points; i++){
        const double *fcreal = fourier_coefs.getStrip(i);
        const double *fcimag = fourier_coefs.getStrip(i + num_points);
        double wr = wreal[i];
        double wi = wimag[i];
        for(int k=0; k<num_outputs; k++) y[k] += wr * fcreal[k] - wi * fcimag[k];
    }
}
void GridFourier::evaluateBatch(const double x[], int num_x, double y[]) const{
    switch(acceleration->mode){
        case accel_gpu_magma:
        case accel_gpu_cuda: {
            acceleration->setDevice();
            GpuVector<double> gpu_x(acceleration, num_dimensions, num_x, x), gpu_y(acceleration, num_outputs, num_x);
            evaluateBatchGPU(gpu_x.data(), num_x, gpu_y.data());
            gpu_y.unload(acceleration, y);
            break;
        }
        case accel_gpu_cublas: {
            acceleration->setDevice();
            loadGpuCoefficients<double>();
            Data2D<double> wreal;
            Data2D<double> wimag;
            evaluateHierarchicalFunctionsInternal(x, num_x, wreal, wimag);

            int num_points = points.getNumIndexes();
            GpuVector<double> gpu_real(acceleration, wreal.begin(), wreal.end()),
                              gpu_imag(acceleration, wimag.begin(), wimag.end()),
                              gpu_y(acceleration, num_outputs, num_x);
            TasGpu::denseMultiply(acceleration, num_outputs, num_x, num_points,  1.0, gpu_cache->real, gpu_real, 0.0, gpu_y.data());
            TasGpu::denseMultiply(acceleration, num_outputs, num_x, num_points, -1.0, gpu_cache->imag, gpu_imag, 1.0, gpu_y.data());
            gpu_y.unload(acceleration, y);
            break;
        }
        case accel_cpu_blas: {
            int num_points = points.getNumIndexes();
            Data2D<double> wreal;
            Data2D<double> wimag;
            if (num_x > 1){
                evaluateHierarchicalFunctionsInternal(x, num_x, wreal, wimag);
            }else{ // work-around small OpenMP penalty
                wreal = Data2D<double>(num_points, 1);
                wimag = Data2D<double>(num_points, 1);
                computeBasis<double, false>(points, x, wreal.data(), wimag.data());
            }
            TasBLAS::denseMultiply(num_outputs, num_x, num_points, 1.0, fourier_coefs.getStrip(0), wreal.data(), 0.0, y);
            TasBLAS::denseMultiply(num_outputs, num_x, num_points, -1.0, fourier_coefs.getStrip(num_points), wimag.data(), 1.0, y);
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

template<typename T> void GridFourier::evaluateBatchGPUtempl(const T gpu_x[], int cpu_num_x, T gpu_y[]) const{
    loadGpuCoefficients<T>();

    GpuVector<T> gpu_real, gpu_imag;
    evaluateHierarchicalFunctionsInternalGPU(gpu_x, cpu_num_x, gpu_real, gpu_imag);

    int num_points = points.getNumIndexes();
    auto& ccache = getGpuCache<T>();
    TasGpu::denseMultiply(acceleration, num_outputs, cpu_num_x, num_points,  1.0, ccache->real, gpu_real, 0.0, gpu_y);
    TasGpu::denseMultiply(acceleration, num_outputs, cpu_num_x, num_points, -1.0, ccache->imag, gpu_imag, 1.0, gpu_y);
}
void GridFourier::evaluateBatchGPU(const double gpu_x[], int cpu_num_x, double gpu_y[]) const{
    evaluateBatchGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
void GridFourier::evaluateBatchGPU(const float gpu_x[], int cpu_num_x, float gpu_y[]) const{
    evaluateBatchGPUtempl(gpu_x, cpu_num_x, gpu_y);
}
void GridFourier::evaluateHierarchicalFunctionsGPU(const double gpu_x[], int num_x, double gpu_y[]) const{
    loadGpuNodes<double>();
    TasGpu::devalfor(acceleration, num_dimensions, num_x, max_levels, gpu_x, gpu_cache->num_nodes, gpu_cache->points, gpu_y, nullptr);
}
void GridFourier::evaluateHierarchicalFunctionsGPU(const float gpu_x[], int num_x, float gpu_y[]) const{
    loadGpuNodes<float>();
    TasGpu::devalfor(acceleration, num_dimensions, num_x, max_levels, gpu_x, gpu_cachef->num_nodes, gpu_cachef->points, gpu_y, nullptr);
}
template<typename T>
void GridFourier::evaluateHierarchicalFunctionsInternalGPU(const T gpu_x[], int num_x, GpuVector<T> &wreal, GpuVector<T> &wimag) const{
    size_t num_weights = ((size_t) points.getNumIndexes()) * ((size_t) num_x);
    if (wreal.size() != num_weights) wreal.resize(acceleration, num_weights);
    if (wimag.size() != num_weights) wimag.resize(acceleration, num_weights);
    loadGpuNodes<T>();
    auto& ccache = getGpuCache<T>();
    TasGpu::devalfor(acceleration, num_dimensions, num_x, max_levels, gpu_x, ccache->num_nodes, ccache->points, wreal.data(), wimag.data());
}
template<typename T> void GridFourier::loadGpuNodes() const{
    auto& ccache = getGpuCache<T>();
    if (!ccache) ccache = CudaFourierData<T>{};
    if (!ccache->num_nodes.empty()) return;

    std::vector<int> num_nodes(num_dimensions);
    std::transform(max_levels.begin(), max_levels.end(), num_nodes.begin(), [](int l)->int{ return OneDimensionalMeta::getNumPoints(l, rule_fourier); });
    ccache->num_nodes.load(acceleration, num_nodes);

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();
    Data2D<int> transpoints(work.getNumIndexes(), num_dimensions);
    for(int i=0; i<num_points; i++)
        for(int j=0; j<num_dimensions; j++)
            transpoints.getStrip(j)[i] = work.getIndex(i)[j];
    ccache->points.load(acceleration, transpoints.totalSize(), transpoints.data());
}
void GridFourier::clearGpuNodes() const{
    if (gpu_cache){
        gpu_cache->num_nodes.clear();
        gpu_cache->points.clear();
    }
    if (gpu_cachef){
        gpu_cachef->num_nodes.clear();
        gpu_cachef->points.clear();
    }
}
template<typename T> void GridFourier::loadGpuCoefficients() const{
    auto& ccache = getGpuCache<T>();
    if (!ccache) ccache = CudaFourierData<T>{};
    if (!ccache->real.empty()) return;
    int num_points = points.getNumIndexes();
    size_t num_coeff = Utils::size_mult(num_outputs, num_points);
    ccache->real.load(acceleration, num_coeff, fourier_coefs.getStrip(0));
    ccache->imag.load(acceleration, num_coeff, fourier_coefs.getStrip(num_points));
}
void GridFourier::clearGpuCoefficients() const{
    if (gpu_cache){
        gpu_cache->real.clear();
        gpu_cache->imag.clear();
    }
    if (gpu_cachef){
        gpu_cachef->real.clear();
        gpu_cachef->imag.clear();
    }
}

void GridFourier::integrate(double q[], double *conformal_correction) const{
    if (conformal_correction == 0){
        // everything vanishes except the Fourier coeff of e^0
        std::copy_n(fourier_coefs.getStrip(0), num_outputs, q);
    }else{
        // Do the expensive computation if we have a conformal map
        std::fill_n(q, num_outputs, 0.0);
        std::vector<double> w(getNumPoints());
        getQuadratureWeights(w.data());
        for(int i=0; i<points.getNumIndexes(); i++){
            w[i] *= conformal_correction[i];
            const double *v = values.getValues(i);
            for(int k=0; k<num_outputs; k++){
                q[k] += w[i] * v[k];
            }
        }
    }
}

void GridFourier::differentiate(const double x[], double jacobian[]) const {
    /* From GridFourier::evaluate(), the contribution of a particular point/tensor to the surrogate model at a point x is
     *
     *     p(x) := Re[w] * Re[exp(c*x*I)] - Im[w] * Im[exp(c*x*I)],
     *
     * where w is a complex scalar, c is a real scalar, and I^2 = -1. Hence, the derivative is
     *
     *     p'(x) := -c * ( Re[w] * Im[exp(c*x*I)] + Im[w] * Re[exp(c*x*I)] ).
     */
    int num_points = points.getNumIndexes();
    std::fill_n(jacobian, num_outputs * num_dimensions, 0.0);
    std::vector<double> wreal(num_points), wimag(num_points);
    computeBasis<double, false>(points, x, wreal.data(), wimag.data());
    for(int i=0; i<num_points; i++) {
        const int *p = points.getIndex(i);
        const double *fcreal = fourier_coefs.getStrip(i);
        const double *fcimag = fourier_coefs.getStrip(i + num_points);
        for(int k=0; k<num_outputs; k++) {
            for(int j=0; j<num_dimensions; j++) {
                double phase = Maths::pi * (double) (p[j] % 2 == 0 ? p[j] : -p[j]-1);
                jacobian[k * num_dimensions + j] += -phase * (wimag[i] * fcreal[k] + wreal[i] * fcimag[k]);
            }
        }
    }
}

void GridFourier::evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const{
    // y must be of size 2*num_x*num_points
    int num_points = getNumPoints();
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(2*num_points, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++){
        computeBasis<double, true>(((points.empty()) ? needed : points), xwrap.getStrip(i), ywrap.getStrip(i), 0);
    }
}
void GridFourier::evaluateHierarchicalFunctionsInternal(const double x[], int num_x, Data2D<double> &wreal, Data2D<double> &wimag) const{
    // when performing internal evaluations, split the matrix into real and complex components
    // thus only two real gemm() operations can be used (as opposed to one complex gemm)
    int num_points = getNumPoints();
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    wreal = Data2D<double>(num_points, num_x);
    wimag = Data2D<double>(num_points, num_x);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++){
        computeBasis<double, false>(((points.empty()) ? needed : points), xwrap.getStrip(i), wreal.getStrip(i), wimag.getStrip(i));
    }
}

void GridFourier::setHierarchicalCoefficients(const double c[]){
    // takes c to be length 2*num_outputs*num_points
    // first num_points*num_outputs are the real part; second num_points*num_outputs are the imaginary part
    clearGpuNodes();
    clearGpuCoefficients();
    if (points.empty()){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        clearRefinement();
    }
    auto num_points = points.getNumIndexes();
    fourier_coefs = Data2D<double>(num_outputs, 2 * num_points, std::vector<double>(c, c + Utils::size_mult(num_outputs, 2 * num_points)));

    std::vector<double> x(Utils::size_mult(num_dimensions, num_points));
    std::vector<double> y(Utils::size_mult(num_outputs,    num_points));

    getPoints(x.data());
    evaluateBatch(x.data(), points.getNumIndexes(), y.data()); // speed this up later

    values = StorageSet(num_outputs, num_points, std::move(y));

}
void GridFourier::integrateHierarchicalFunctions(double integrals[]) const{
    integrals[0] = 1.0;
    std::fill(integrals + 1, integrals + getNumPoints(), 0.0);
}

#ifdef Tasmanian_ENABLE_GPU
void GridFourier::updateAccelerationData(AccelerationContext::ChangeType change) const{
    if (change == AccelerationContext::change_gpu_device){
        gpu_cache.reset();
        gpu_cachef.reset();
    }
}
#else
void GridFourier::updateAccelerationData(AccelerationContext::ChangeType) const{}
#endif

void GridFourier::estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const{
    double tol = 1000.0 * Maths::num_tol;
    int num_points = points.getNumIndexes();
    std::vector<double> max_fcoef(num_points);

    if (output == -1){
        // normalize in case of outputs with hugely different scaling
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
            const double *fcreal = fourier_coefs.getStrip(i);
            const double *fcimag = fourier_coefs.getStrip(i + num_points);
            double fcmax = 0.0;
            for(int k=0; k<num_outputs; k++){
                double v = std::sqrt(fcreal[k] * fcreal[k] + fcimag[k] * fcimag[k]) / nrm[k];
                if (fcmax < v) fcmax = v;
            }
            max_fcoef[i] = fcmax;
        }
    }else{
        int i = 0;
        for(auto &m : max_fcoef){
            const double *fcreal = fourier_coefs.getStrip(i);
            const double *fcimag = fourier_coefs.getStrip(i++ + num_points);
            m = std::sqrt(fcreal[output] * fcreal[output] + fcimag[output] * fcimag[output]);
        }
    }

    weights = MultiIndexManipulations::inferAnisotropicWeights(acceleration, rule_fourier, type, points, max_fcoef, tol);
}

void GridFourier::setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits){
    clearRefinement();
    std::vector<int> weights;
    estimateAnisotropicCoefficients(type, output, weights);

    int level = 0;
    do{
        updateGrid(++level, type, weights, level_limits);
    }while(getNumNeeded() < min_growth);
}

void GridFourier::clearRefinement(){
    needed = MultiIndexSet();
    updated_tensors = MultiIndexSet();
    updated_active_tensors = MultiIndexSet();
    updated_active_w = std::vector<int>();
}

void GridFourier::mergeRefinement(){
    if (needed.empty()) return; // nothing to do
    int num_all_points = getNumLoaded() + getNumNeeded();
    values.setValues(std::vector<double>(Utils::size_mult(num_outputs, num_all_points), 0.0));
    acceptUpdatedTensors();
}

void GridFourier::beginConstruction(){
    dynamic_values = std::make_unique<DynamicConstructorDataGlobal>(num_dimensions, num_outputs);
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
void GridFourier::writeConstructionData(std::ostream &os, bool iomode) const{
    if (iomode == mode_ascii) dynamic_values->write<mode_ascii>(os); else dynamic_values->write<mode_binary>(os);
}
void GridFourier::readConstructionData(std::istream &is, bool iomode){
    if (iomode == mode_ascii)
        dynamic_values = std::make_unique<DynamicConstructorDataGlobal>(is, num_dimensions, num_outputs, IO::mode_ascii_type());
    else
        dynamic_values = std::make_unique<DynamicConstructorDataGlobal>(is, num_dimensions, num_outputs, IO::mode_binary_type());
    int max_level = dynamic_values->getMaxTensor();
    if (max_level + 1 > wrapper.getNumLevels())
        wrapper = OneDimensionalWrapper(max_level, rule_fourier, 0.0, 0.0);
    dynamic_values->reloadPoints([&](int l)->int{ return wrapper.getNumPoints(l); });
}
std::vector<double> GridFourier::getCandidateConstructionPoints(TypeDepth type, int output, const std::vector<int> &level_limits){
    std::vector<int> weights;
    if ((type == type_iptotal) || (type == type_ipcurved) || (type == type_qptotal) || (type == type_qpcurved)){
        int min_needed_points = ((type == type_ipcurved) || (type == type_qpcurved)) ? 4 * num_dimensions : 2 * num_dimensions;
        if (points.getNumIndexes() > min_needed_points) // if there are enough points to estimate coefficients
            estimateAnisotropicCoefficients(type, output, weights);
    }
    return getCandidateConstructionPoints(type, weights, level_limits);
}
std::vector<double> GridFourier::getCandidateConstructionPoints(TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
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
                for(size_t i=0; i<num; i++) effective_exactness[i] = OneDimensionalMeta::getIExact((int) i, rule_fourier);
            }else{ // must be quadrature
                for(size_t i=0; i<num; i++) effective_exactness[i] = OneDimensionalMeta::getQExact((int) i, rule_fourier);
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
std::vector<double> GridFourier::getCandidateConstructionPoints(std::function<double(const int *)> getTensorWeight, const std::vector<int> &level_limits){
    dynamic_values->clearTesnors(); // clear old tensors
    MultiIndexSet init_tensors = dynamic_values->getInitialTensors(); // get the initial tensors (created with make grid)

    MultiIndexSet new_tensors = (level_limits.empty()) ?
        MultiIndexManipulations::addExclusiveChildren<false>(tensors, init_tensors, level_limits) :
        MultiIndexManipulations::addExclusiveChildren<true>(tensors, init_tensors, level_limits);

    if (!new_tensors.empty()){
        int max_level = new_tensors.getMaxIndex();
        if (max_level+1 > wrapper.getNumLevels())
            wrapper = OneDimensionalWrapper(max_level, rule_fourier, 0.0, 0.0);
    }

    std::vector<double> tweights(new_tensors.getNumIndexes());
    for(int i=0; i<new_tensors.getNumIndexes(); i++)
        tweights[i] = (double) getTensorWeight(new_tensors.getIndex(i));

    for(int i=0; i<new_tensors.getNumIndexes(); i++)
        dynamic_values->addTensor(new_tensors.getIndex(i), [&](int l)->int{ return wrapper.getNumPoints(l); }, tweights[i]);

    return MultiIndexManipulations::getIndexesToNodes(dynamic_values->getNodesIndexes(), wrapper);
}
std::vector<int> GridFourier::getMultiIndex(const double x[]){
    std::vector<int> p(num_dimensions);
    for(int j=0; j<num_dimensions; j++){
        int i = 0;
        while(std::abs(wrapper.getNode(i) - x[j]) > Maths::num_tol){
            i++; // convert canonical node to index
            if (i == wrapper.getNumNodes())
                wrapper = OneDimensionalWrapper(wrapper.getNumLevels(), rule_fourier, 0.0, 0.0);
        }
        p[j] = i;
    }
    return p;
}
void GridFourier::loadConstructedPoint(const double x[], const std::vector<double> &y){
    std::vector<int> idx = getMultiIndex(x);
    auto result = dynamic_values->addNewNode(idx, y);
    if (result == DynamicConstructorDataGlobal::AddPointResult::tensor_complete){ // if a new tensor is complete
        loadConstructedTensors();
    }else if (result == DynamicConstructorDataGlobal::AddPointResult::tensor_missing){
        dynamic_values->addTensor(wrapper.getLevels(idx).data(), [&](int l)->int{ return wrapper.getNumPoints(l); },
                                  dynamic_values->getMaxTensorWeight() + 1.0);
    }
}
void GridFourier::loadConstructedPoint(const double x[], int numx, const double y[]){
    Utils::Wrapper2D<const double> wrapx(num_dimensions, x);
    Utils::Wrapper2D<const double> wrapy(num_outputs, y);
    for(int i=0; i<numx; i++){
        std::vector<int> idx = getMultiIndex(wrapx.getStrip(i));
        if (dynamic_values->addNewNode(idx, std::vector<double>(wrapy.getStrip(i), wrapy.getStrip(i) + num_outputs))
            == DynamicConstructorDataGlobal::AddPointResult::tensor_missing){
            dynamic_values->addTensor(wrapper.getLevels(idx).data(), [&](int l)->int{ return wrapper.getNumPoints(l); },
                                      dynamic_values->getMaxTensorWeight() + 1.0);
        }
    }
    loadConstructedTensors();
}
void GridFourier::loadConstructedTensors(){
    clearGpuNodes();
    clearGpuCoefficients();
    MultiIndexSet new_tensors, new_points;
    StorageSet new_values;
    dynamic_values->ejectCompleteTensor(tensors, new_tensors, new_points, new_values);
    if (new_tensors.empty()) return; // nothing to do

    if (points.empty()){ // no loaded points yet
        values = std::move(new_values);
        points = std::move(new_points);
    }else{
        values.addValues(points, new_points, new_values.getValues(0));
        points += new_points;
    }

    tensors += new_tensors;
    MultiIndexManipulations::computeActiveTensorsWeights(tensors, active_tensors, active_w);

    max_levels = MultiIndexManipulations::getMaxIndexes(active_tensors);
    max_power  = MultiIndexManipulations::getMaxIndexes(points);

    calculateFourierCoefficients();
}
void GridFourier::finishConstruction(){
    dynamic_values = std::unique_ptr<DynamicConstructorDataGlobal>();
}


const double* GridFourier::getFourierCoefs() const{
    return fourier_coefs.getStrip(0);
}

} // end TasGrid

#endif
