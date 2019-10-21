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

#include <thread>
#include <mutex>

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
 * \internal
 * \ingroup TasmanianUtils
 * \brief Miscellaneous utility templates.
 */
namespace Utils{

/*!
 * \internal
 * \brief Converts two integer-like variables to \b size_t and returns the product.
 * \ingroup TasmanianUtils
 * \endinternal
 */
template<typename IntA, typename IntB>
inline size_t size_mult(IntA a, IntB b){ return ((size_t) a) * ((size_t) b); }

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
        std::vector<typename std::remove_const<T>::type>(x, x + ((size_t) size));
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

struct SyncParallel{};
struct SyncStatic{};

/*!
 * \internal
 * \ingroup TasmanianUtils
 * \brief Compute entries in parallel.
 *
 * \endinternal
 */
template<typename IndexType, typename SyncType = SyncStatic>
inline void doParallel(IndexType num_threads, IndexType num_runs, std::function<void(IndexType i)> work){

    IndexType num_blocks = ((std::is_same<SyncType, SyncParallel>::value) ? 16 : 1) * num_threads;
    IndexType whole_jumps = num_runs / num_blocks;
    IndexType rem_jumps = num_runs % num_blocks;
    auto get_offset = [&](IndexType j)->IndexType{ return static_cast<IndexType>(j * whole_jumps + std::min(j, rem_jumps)); };

    if (std::is_same<SyncType, SyncParallel>::value){

        std::vector<bool> checked_out(num_blocks, false);
        std::mutex checked_out_lock;

        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for(IndexType thread_id=0; thread_id<num_threads; thread_id++){
            workers.emplace_back( // create a new worker thread
                    [&, thread_id](void)->void{
                        IndexType sample = 0;

                        do{
                            { // find the next sample
                                std::lock_guard<std::mutex> lock(checked_out_lock);
                                while ((sample < num_blocks) && checked_out[sample]) sample++;
                                if (sample < num_blocks) checked_out[sample] = true;
                            }

                            if (sample < num_blocks)
                                for(IndexType i=get_offset(sample); i<get_offset(sample + 1); i++) work(i);
                        }while(sample < num_blocks);
                    });
        }

        for(auto &w : workers) w.join();
    }else if (std::is_same<SyncType, SyncStatic>::value){

        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for(IndexType thread_id=0; thread_id<num_threads; thread_id++){
            workers.emplace_back( // create a new worker thread
                    [&, thread_id](void)->void{
                        for(IndexType i=get_offset(thread_id); i<get_offset(thread_id + 1); i++) work(i);
                    });
        }

        for(auto &w : workers) w.join();
    }
}

}

}

#endif
