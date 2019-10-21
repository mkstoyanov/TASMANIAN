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
#include "tsgHiddenExternals.hpp"

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
        if (fourier_coefs.getNumStrips() != 0) IO::writeVector<iomode, IO::pad_line>(fourier_coefs.getVector(), os);
    }

    IO::writeFlag<iomode, IO::pad_line>(!updated_tensors.empty(), os);
    if (!updated_tensors.empty()){
        updated_tensors.write<iomode>(os);
        updated_active_tensors.write<iomode>(os);
        IO::writeVector<iomode, IO::pad_line>(updated_active_w, os);
    }
}
template<bool iomode> void GridFourier::read(std::istream &is){
    reset();
    num_dimensions = IO::readNumber<iomode, int>(is);
    num_outputs = IO::readNumber<iomode, int>(is);

    tensors.read<iomode>(is);
    active_tensors.read<iomode>(is);
    active_w.resize((size_t) active_tensors.getNumIndexes());
    IO::readVector<iomode>(is, active_w);

    if (IO::readFlag<iomode>(is)) points.read<iomode>(is);
    if (IO::readFlag<iomode>(is)) needed.read<iomode>(is);

    max_levels.resize((size_t) num_dimensions);
    IO::readVector<iomode>(is, max_levels);

    if (num_outputs > 0){
        values.read<iomode>(is);
        if (IO::readFlag<iomode>(is))
            fourier_coefs = IO::readData2D<iomode, double>(is, num_outputs, 2 * points.getNumIndexes());
    }

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

    wrapper.load(oned_max_level, rule_fourier, 0.0, 0.0);

    max_power = MultiIndexManipulations::getMaxIndexes(((points.empty()) ? needed : points));
}

template void GridFourier::write<mode_ascii>(std::ostream &) const;
template void GridFourier::write<mode_binary>(std::ostream &) const;
template void GridFourier::read<mode_ascii>(std::istream &);
template void GridFourier::read<mode_binary>(std::istream &);

void GridFourier::reset(){
    clearAccelerationData();
    wrapper = OneDimensionalWrapper();
    tensors = MultiIndexSet();
    active_tensors = MultiIndexSet();
    points = MultiIndexSet();
    needed = MultiIndexSet();
    values = StorageSet();
    active_w.clear();
    fourier_coefs.clear();
    num_dimensions = 0;
    num_outputs = 0;
}

void GridFourier::makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    setTensors(selectTensors((size_t) cnum_dimensions, depth, type, anisotropic_weights, level_limits), cnum_outputs);
}

void GridFourier::copyGrid(const GridFourier *fourier, int ibegin, int iend){
    num_dimensions = fourier->num_dimensions;
    num_outputs    = iend - ibegin;
    points = fourier->points;
    needed = fourier->needed;

    wrapper = fourier->wrapper;

    tensors        = fourier->tensors;
    active_tensors = fourier->active_tensors;
    active_w       = fourier->active_w;

    updated_tensors        = fourier->updated_tensors;
    updated_active_tensors = fourier->updated_active_tensors;
    updated_active_w       = fourier->updated_active_w;

    max_levels    = fourier->max_levels;
    fourier_coefs = (num_outputs == fourier->num_outputs) ? fourier->fourier_coefs : fourier->fourier_coefs.splitData(ibegin, iend);
    values        = (num_outputs == fourier->num_outputs) ? fourier->values : fourier->values.splitValues(ibegin, iend);

    max_power = fourier->max_power;
}

void GridFourier::updateGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    if ((num_outputs == 0) || points.empty()){
        makeGrid(num_dimensions, num_outputs, depth, type, anisotropic_weights, level_limits);
    }else{
        clearRefinement();

        updated_tensors = selectTensors((size_t) num_dimensions, depth, type, anisotropic_weights, level_limits);

        MultiIndexSet new_tensors = updated_tensors.diffSets(tensors);

        if (!new_tensors.empty()){
            updated_tensors.addMultiIndexSet(tensors);
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
    reset();
    tensors = std::move(tset);

    num_dimensions = (int) tensors.getNumDimensions();
    num_outputs = cnum_outputs;

    max_levels = MultiIndexManipulations::getMaxIndexes(tensors);

    wrapper.load(*std::max_element(max_levels.begin(), max_levels.end()), rule_fourier, 0.0, 0.0);

    std::vector<int> tensors_w = MultiIndexManipulations::computeTensorWeights(threaded, tensors);
    active_tensors = MultiIndexManipulations::createActiveTensors(tensors, tensors_w);

    int nz_weights = active_tensors.getNumIndexes();

    active_w.reserve(nz_weights);
    for(auto w : tensors_w) if (w != 0) active_w.push_back(w);

    needed = MultiIndexManipulations::generateNestedPoints(threaded, tensors, [&](int l) -> int{ return wrapper.getNumPoints(l); });

    if (num_outputs == 0){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        values.resize(num_outputs, needed.getNumIndexes());
    }

    max_power = MultiIndexManipulations::getMaxIndexes(((points.empty()) ? needed : points));
}

void GridFourier::proposeUpdatedTensors(){
    wrapper.load(updated_tensors.getMaxIndex(), rule_fourier, 0.0, 0.0);

    std::vector<int> updates_tensor_w = MultiIndexManipulations::computeTensorWeights(threaded, updated_tensors);
    updated_active_tensors = MultiIndexManipulations::createActiveTensors(updated_tensors, updates_tensor_w);

    updated_active_w.reserve(updated_active_tensors.getNumIndexes());
    for(auto w : updates_tensor_w) if (w != 0) updated_active_w.push_back(w);

    MultiIndexSet new_points = MultiIndexManipulations::generateNestedPoints(threaded, updated_tensors,
                                [&](int l) -> int{ return wrapper.getNumPoints(l); });

    needed = new_points.diffSets(points);
}

void GridFourier::acceptUpdatedTensors(){
    if (points.empty()){
        #ifdef Tasmanian_ENABLE_CUDA
        clearCudaNodes(); // the points and needed will change, clear the cache
        #endif
        points = std::move(needed);
        needed = MultiIndexSet();
    }else if (!needed.empty()){
        points.addMultiIndexSet(needed);
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

void GridFourier::loadNeededPoints(const double *vals){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaCoefficients(); // changing values and Fourier coefficients, clear the cache
    #endif
    if (points.empty() || needed.empty()){
        values.setValues(vals);
    }else{
        values.addValues(points, needed, vals);
    }

    acceptUpdatedTensors();
    calculateFourierCoefficients();

    max_power = MultiIndexManipulations::getMaxIndexes(points);
}

void GridFourier::mapIndexesToNodes(const std::vector<int> &indexes, double *x) const{
    std::transform(indexes.begin(), indexes.end(), x, [&](int i)->double{ return wrapper.getNode(i); });
}

void GridFourier::getLoadedPoints(double *x) const{
    mapIndexesToNodes(points.getVector(), x);
}
void GridFourier::getNeededPoints(double *x) const{
    mapIndexesToNodes(needed.getVector(), x);
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

    fourier_coefs.resize(num_outputs, 2 * num_points);
    fourier_coefs.fill(0.0);

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
            const double *v = values.getValues(work.getSlot(p));
            tensor_data[i].resize(num_outputs);
            std::copy(v, v + num_outputs, tensor_data[i].data());
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

    std::fill(weights, weights + getNumPoints(), 0.0);

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
    int num_points = work.getNumIndexes();
    std::fill(weights, weights + num_points, 0.0);

    for(int n=0; n<active_tensors.getNumIndexes(); n++){
        const int *levels = active_tensors.getIndex(n);
        int num_tensor_points = 1;
        for(int j=0; j<num_dimensions; j++){
            num_tensor_points *= wrapper.getNumPoints(levels[j]);
        }
        std::vector<int> refs;
        MultiIndexManipulations::referencePoints<true>(levels, wrapper, work, refs);

        double tensorw = ((double) active_w[n]) / ((double) num_tensor_points);
        for(int i=0; i<num_tensor_points; i++){
            weights[refs[i]] += tensorw;
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
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_outputs, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++)
        evaluate(xwrap.getStrip(i), ywrap.getStrip(i));
}

#ifdef Tasmanian_ENABLE_BLAS
void GridFourier::evaluateBlas(const double x[], int num_x, double y[]) const{
    int num_points = points.getNumIndexes();
    Data2D<double> wreal;
    Data2D<double> wimag;
    if (num_x > 1){
        evaluateHierarchicalFunctionsInternal(x, num_x, wreal, wimag);
    }else{ // work-around small OpenMP penalty
        wreal.resize(num_points, 1);
        wimag.resize(num_points, 1);
        computeBasis<double, false>(points, x, wreal.getStrip(0), wimag.getStrip(0));
    }
    TasBLAS::denseMultiply(num_outputs, num_x, num_points, 1.0, fourier_coefs.getStrip(0), wreal.getStrip(0), 0.0, y);
    TasBLAS::denseMultiply(num_outputs, num_x, num_points, -1.0, fourier_coefs.getStrip(num_points), wimag.getStrip(0), 1.0, y);
}
#endif

#ifdef Tasmanian_ENABLE_CUDA
void GridFourier::loadNeededPointsCuda(CudaEngine *, const double *vals){
    loadNeededPoints(vals);
}
void GridFourier::evaluateCudaMixed(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    loadCudaCoefficients<double>();
    Data2D<double> wreal;
    Data2D<double> wimag;
    evaluateHierarchicalFunctionsInternal(x, num_x, wreal, wimag);

    int num_points = points.getNumIndexes();
    CudaVector<double> gpu_real(wreal.getVector()), gpu_imag(wimag.getVector()), gpu_y(num_outputs, num_x);
    engine->denseMultiply(num_outputs, num_x, num_points,  1.0, cuda_cache->real, gpu_real, 0.0, gpu_y.data());
    engine->denseMultiply(num_outputs, num_x, num_points, -1.0, cuda_cache->imag, gpu_imag, 1.0, gpu_y.data());
    gpu_y.unload(y);
}
void GridFourier::evaluateCuda(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    CudaVector<double> gpu_x(num_dimensions, num_x, x), gpu_y(num_outputs, num_x);
    evaluateBatchGPU(engine, gpu_x.data(), num_x, gpu_y.data());
    gpu_y.unload(y);
}
template<typename T> void GridFourier::evaluateBatchGPUtempl(CudaEngine *engine, const T gpu_x[], int cpu_num_x, T gpu_y[]) const{
    loadCudaCoefficients<T>();

    CudaVector<T> gpu_real, gpu_imag;
    evaluateHierarchicalFunctionsInternalGPU(gpu_x, cpu_num_x, gpu_real, gpu_imag);

    int num_points = points.getNumIndexes();
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    engine->denseMultiply(num_outputs, cpu_num_x, num_points,  1.0, ccache->real, gpu_real, 0.0, gpu_y);
    engine->denseMultiply(num_outputs, cpu_num_x, num_points, -1.0, ccache->imag, gpu_imag, 1.0, gpu_y);
}
void GridFourier::evaluateBatchGPU(CudaEngine *engine, const double gpu_x[], int cpu_num_x, double gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
void GridFourier::evaluateBatchGPU(CudaEngine *engine, const float gpu_x[], int cpu_num_x, float gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
void GridFourier::evaluateHierarchicalFunctionsGPU(const double gpu_x[], int num_x, double gpu_y[]) const{
    loadCudaNodes<double>();
    TasCUDA::devalfor(num_dimensions, num_x, max_levels, gpu_x, cuda_cache->num_nodes, cuda_cache->points, gpu_y, nullptr);
}
void GridFourier::evaluateHierarchicalFunctionsGPU(const float gpu_x[], int num_x, float gpu_y[]) const{
    loadCudaNodes<float>();
    TasCUDA::devalfor(num_dimensions, num_x, max_levels, gpu_x, cuda_cachef->num_nodes, cuda_cachef->points, gpu_y, nullptr);
}
template<typename T>
void GridFourier::evaluateHierarchicalFunctionsInternalGPU(const T gpu_x[], int num_x, CudaVector<T> &wreal, CudaVector<T> &wimag) const{
    size_t num_weights = ((size_t) points.getNumIndexes()) * ((size_t) num_x);
    if (wreal.size() != num_weights) wreal.resize(num_weights);
    if (wimag.size() != num_weights) wimag.resize(num_weights);
    loadCudaNodes<T>();
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    TasCUDA::devalfor(num_dimensions, num_x, max_levels, gpu_x, ccache->num_nodes, ccache->points, wreal.data(), wimag.data());
}
template<typename T> void GridFourier::loadCudaNodes() const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaFourierData<T>>();
    if (!ccache->num_nodes.empty()) return;

    std::vector<int> num_nodes(num_dimensions);
    std::transform(max_levels.begin(), max_levels.end(), num_nodes.begin(), [](int l)->int{ return OneDimensionalMeta::getNumPoints(l, rule_fourier); });
    ccache->num_nodes.load(num_nodes);

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();
    Data2D<int> transpoints(work.getNumIndexes(), num_dimensions);
    for(int i=0; i<num_points; i++)
        for(int j=0; j<num_dimensions; j++)
            transpoints.getStrip(j)[i] = work.getIndex(i)[j];
    ccache->points.load(transpoints.getVector());
}
void GridFourier::clearCudaNodes(){
    if (cuda_cache){
        cuda_cache->num_nodes.clear();
        cuda_cache->points.clear();
    }
    if (cuda_cachef){
        cuda_cachef->num_nodes.clear();
        cuda_cachef->points.clear();
    }
}
template<typename T> void GridFourier::loadCudaCoefficients() const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaFourierData<T>>();
    if (!ccache->real.empty()) return;
    int num_points = points.getNumIndexes();
    size_t num_coeff = ((size_t) num_outputs) * ((size_t) num_points);
    ccache->real.load(num_coeff, fourier_coefs.getStrip(0));
    ccache->imag.load(num_coeff, fourier_coefs.getStrip(num_points));
}
void GridFourier::clearCudaCoefficients(){
    if (cuda_cache){
        cuda_cache->real.clear();
        cuda_cache->imag.clear();
    }
    if (cuda_cachef){
        cuda_cachef->real.clear();
        cuda_cachef->imag.clear();
    }
}
#endif

void GridFourier::integrate(double q[], double *conformal_correction) const{
    std::fill(q, q+num_outputs, 0.0);
    if (conformal_correction == 0){
        // everything vanishes except the Fourier coeff of e^0
        std::copy(fourier_coefs.getStrip(0), fourier_coefs.getStrip(0) + num_outputs, q);
    }else{
        // Do the expensive computation if we have a conformal map
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
    wreal.resize(num_points, num_x);
    wimag.resize(num_points, num_x);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++){
        computeBasis<double, false>(((points.empty()) ? needed : points), xwrap.getStrip(i), wreal.getStrip(i), wimag.getStrip(i));
    }
}

void GridFourier::setHierarchicalCoefficients(const double c[], TypeAcceleration){
    // takes c to be length 2*num_outputs*num_points
    // first num_points*num_outputs are the real part; second num_points*num_outputs are the imaginary part

    clearAccelerationData();
    if (points.empty()){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        clearRefinement();
    }
    fourier_coefs.resize(num_outputs, 2 * getNumPoints());
    std::copy_n(c, 2 * ((size_t) num_outputs) * ((size_t) getNumPoints()), fourier_coefs.getStrip(0));
}
void GridFourier::integrateHierarchicalFunctions(double integrals[]) const{
    integrals[0] = 1.0;
    std::fill(integrals + 1, integrals + getNumPoints(), 0.0);
}

void GridFourier::clearAccelerationData(){
    #ifdef Tasmanian_ENABLE_CUDA
    cuda_cache.reset();
    cuda_cachef.reset();
    #endif
}

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

    int n = 0;
    int m = num_dimensions + 1;
    for(int i=0; i<num_points; i++) n += (max_fcoef[i] > tol) ? 1 : 0;

    Data2D<double> A(n,m);
    std::vector<double> b(n);

    int count = 0;
    bool ishyperbolic = (OneDimensionalMeta::getControurType(type) == type_hyperbolic);
    for(int c=0; c<num_points; c++){
        const int *indx = points.getIndex(c);
        if (max_fcoef[c] > tol){
            for(int j=0; j<num_dimensions; j++) A.getStrip(j)[count] = (ishyperbolic) ? log((double) ((indx[j]+1)/2 + 1)) : ((double) ((indx[j]+1)/2));
            A.getStrip(num_dimensions)[count] = 1.0;
            b[count++] = -log(max_fcoef[c]);
        }
    }

    std::vector<double> x(m);
    TasmanianDenseSolver::solveLeastSquares(n, m, A.getStrip(0), b.data(), 1.E-5, x.data());

    weights.resize(--m);
    for(int j=0; j<m; j++) weights[j] = (int)(x[j] * 1000.0 + 0.5);

    int min_weight = *std::max_element(weights.begin(), weights.end()); // start with max weight (placeholder name)
    if (min_weight < 0){ // all directions are diverging, default to isotropic total degree
        for(int j=0; j<num_dimensions; j++) weights[j] = 1;
    }else{
        for(int j=0; j<num_dimensions; j++) if ((weights[j] > 0) && (weights[j] < min_weight)) min_weight = weights[j]; // find the smallest positive element now
        for(int j=0; j<num_dimensions; j++) if (weights[j] <= 0) weights[j] = min_weight;
    }
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
void GridFourier::writeConstructionData(std::ostream &os, bool iomode) const{
    if (iomode == mode_ascii) dynamic_values->write<mode_ascii>(os); else dynamic_values->write<mode_binary>(os);
}
void GridFourier::readConstructionData(std::istream &is, bool iomode){
    dynamic_values = std::unique_ptr<DynamicConstructorDataGlobal>(new DynamicConstructorDataGlobal((size_t) num_dimensions, (size_t) num_outputs));
    if (iomode == mode_ascii)
        dynamic_values->read<mode_ascii>(is);
    else
        dynamic_values->read<mode_binary>(is);
    int max_level = dynamic_values->getMaxTensor();
    if (max_level + 1 > wrapper.getNumLevels())
        wrapper.load(max_level, rule_fourier, 0.0, 0.0);
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
        auto max_indexes = MultiIndexManipulations::getMaxIndexes(new_tensors);
        int max_level = *std::max_element(max_indexes.begin(), max_indexes.end());
        if (max_level+1 > wrapper.getNumLevels())
            wrapper.load(max_level, rule_fourier, 0.0, 0.0);
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
std::vector<int> GridFourier::getMultiIndex(const double x[]){
    std::vector<int> p(num_dimensions);
    for(int j=0; j<num_dimensions; j++){
        int i = 0;
        while(std::abs(wrapper.getNode(i) - x[j]) > Maths::num_tol){
            i++; // convert canonical node to index
            if (i == wrapper.getNumNodes())
                wrapper.load(wrapper.getNumLevels(), rule_fourier, 0.0, 0.0);
        }
        p[j] = i;
    }
    return p;
}
void GridFourier::loadConstructedPoint(const double x[], const std::vector<double> &y){
    if (dynamic_values->addNewNode(getMultiIndex(x), y)) // if a new tensor is complete
        loadConstructedTensors();
}
void GridFourier::loadConstructedPoint(const double x[], int numx, const double y[]){
    Utils::Wrapper2D<const double> wrapx(num_dimensions, x);
    Utils::Wrapper2D<const double> wrapy(num_outputs, y);
    for(int i=0; i<numx; i++)
        dynamic_values->addNewNode(getMultiIndex(wrapx.getStrip(i)), std::vector<double>(wrapy.getStrip(i), wrapy.getStrip(i) + num_outputs));
    loadConstructedTensors();
}
void GridFourier::loadConstructedTensors(){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaNodes();
    clearCudaCoefficients();
    #endif
    MultiIndexSet new_tensors, new_points;
    StorageSet new_values;
    dynamic_values->ejectCompleteTensor(tensors, new_tensors, new_points, new_values);
    if (new_tensors.empty()) return; // nothing to do

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
