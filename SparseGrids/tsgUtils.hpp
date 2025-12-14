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

#ifndef __TASMANIAN_SPARSE_GRID_UTILS_HPP
#define __TASMANIAN_SPARSE_GRID_UTILS_HPP

/*!
 * \internal
 * \file tsgUtils.hpp
 * \brief Miscellaneous utility templates used thoughout the code.
 * \author Miroslav Stoyanov
 * \ingroup TasmanianUtils
 *
 * Templates uses throughout the internal algorithms of Tasmanian.
 * The header will be kept private.
 * \endinternal
 */

#include "tsgMathUtils.hpp"

/*!
 * \internal
 * \ingroup TasmanianSG
 * \addtogroup TasmanianUtils Miscellaneous utility templates
 *
 * \endinternal
 */

namespace TasGrid{

/*!
 * \ingroup TasmanianUtils
 * \brief Miscellaneous utility templates (internal use).
 */
namespace Utils{

//! \brief Constructs the transpose of an M by N matrix A in column major format, result is stored in B (implemented in linear solvers).
template<typename scalar_type>
void transpose(long long M, long long N, scalar_type const A[], scalar_type B[]);
//! \brief Constructs the transpose of an M by N matrix A in column major format, result is returned in a vector.
template<typename scalar_type>
std::vector<scalar_type> transpose(long long M, long long N, scalar_type const A[]){
    std::vector<scalar_type> B(M * N);
    transpose(M, N, A, B.data());
    return B;
}

/*!
 * \internal
 * \brief Converts two integer-like variables to \b size_t and returns the product.
 * \ingroup TasmanianUtils
 * \endinternal
 */
template<typename IntA, typename IntB>
inline size_t size_mult(IntA a, IntB b){ return static_cast<size_t>(a) * static_cast<size_t>(b); }

/*!
 * \internal
 * \ingroup TasmanianUtils
 * \brief Copies an array into a vector, returns empty vector if the input is nullpntr.
 *
 * \endinternal
 */
template<typename T, typename I>
std::vector<typename std::remove_const<T>::type> copyArray(T* x, I size){
    return (x == nullptr) ? std::vector<typename std::remove_const<T>::type>() :
        std::vector<typename std::remove_const<T>::type>(x, x + static_cast<size_t>(size));
}

/*!
 * \internal
 * \ingroup TasmanianUtils
 * \brief Takes a vector of vectors and returns a single contiguous vector.
 *
 * \endinternal
 */
template<typename T>
std::vector<T> mergeVectors(std::vector<std::vector<T>> const &vec){
    size_t total_size = 0;
    for(auto const &v : vec) total_size += v.size();
    std::vector<T> result;
    result.reserve(total_size);
    for(auto const &v : vec) result.insert(result.end(), v.begin(), v.end());
    return result;
}

/*!
 * \internal
 * \brief Wraps around a C-style of an array and mimics 2D data-structure.
 * \ingroup TasmanianUtils
 *
 * The Tasmanian external API accepts C-style arrays, which simplifies
 * interfacing with other languages such as C, Python and Fortran.
 * The arrays represent 2D data in strips with specific stride,
 * the Wrapper2D() takes such an array and logically divides it
 * so strips can be accessed without clumsy double-index notation.
 * \endinternal
 */
template<typename T>
class Wrapper2D{
public:
    //! \brief Wrap around \b raw_data with the given \b stride_size.
    Wrapper2D(int stride_size, T *raw_data) : stride(static_cast<size_t>(stride_size)), data(raw_data){}
    //! \brief Default destructor, \b raw_data has to be deleted elsewhere.
    ~Wrapper2D(){}

    //! \brief Return a pointer to the i-th strip.
    T* getStrip(int i){ return &(data[size_mult(i, stride)]); }

private:
    size_t stride;
    T *data;
};

/*!
 * \ingroup TasmanianUtils
 * \brief Equivalent to C++14 make_unique.
 */
template<typename T>
constexpr bool has_grid(T &&g) {
    return not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>;
}

}

}

#endif
