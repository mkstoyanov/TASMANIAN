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

#ifndef __TASMANIAN_LINEAR_SOLVERS_CPP
#define __TASMANIAN_LINEAR_SOLVERS_CPP

//#include <xmmintrin.h>
#include <immintrin.h>

#include "tsgLinearSolvers.hpp"

namespace TasGrid{

void TasmanianDenseSolver::solveLeastSquares(int n, int m, const double A[], const double b[], double reg, double *x){
    // form Ar = A' * A
    // form x = A' * b
    std::vector<double> Ar(m * m);
    #pragma omp parallel for
    for(int i=0; i<m; i++){
        for(int j=0; j<m; j++){
            double sum = 0.0;
            for(int k=0; k<n; k++){
                sum += A[i*n + k] * A[j*n + k];
            }
            Ar[i*m + j] = sum;
        }
        x[i] = 0.0;
        for(int k=0; k<n; k++){
            x[i] += A[i*n+k] * b[k];
        }
    }

    // add regularization
    for(int i=0; i<m; i++){  Ar[i *m + i] += reg;  };

    // factorize Ar
    for(int i=0; i<m; i++){
        Ar[i*m + i] = std::sqrt(Ar[i*m + i]);

        for(int j=i+1; j<m; j++){
            Ar[i*m + j] /= Ar[i*m + i];
        }

        for(int k=i+1; k<m; k++){
            for(int j=i+1; j <= k; j++){
                Ar[j*m + k] -= Ar[i*m + j] * Ar[i*m + k];
            }
        }
    }

    // solve L^(-1) x
    for(int i=0; i<m; i++){
        x[i] /= Ar[i*m + i];
        for(int j=i+1; j<m; j++){
            x[j] -= x[i] * Ar[i*m + j];
        }
    }

    // solve L^(-T) x
    for(int i=m-1; i>=0; i--){
        for(int j=i+1; j<m; j++){
            x[i] -= x[j] * Ar[i*m + j];
        }
        x[i] /= Ar[i*m + i];
    }
}

void TasmanianTridiagonalSolver::decompose(int n, std::vector<double> &d, std::vector<double> &e, std::vector<double> &z){
    const double tol = Maths::num_tol;
    if (n == 1){ z[0] = z[0]*z[0]; return; }

    for(int l=0; l<n-1; l++){
        int m = l;
        while((m < n-1) && (std::abs(e[m]) > tol)) m++;

        while (m != l){
            double p = d[l];
            double g = (d[l+1] - p) / (2.0 * e[l]);
            double r = std::sqrt(g*g + 1.0);

            g = d[m] - p + e[l] / (g + Maths::sign(g) * r); // sign function here may be unstable

            double s = 1.0;
            double c = 1.0;
            p = 0.0;

            for(int i=m-1; i>=l; i--){
                double f = s * e[i];
                double b = c * e[i];

                if (std::abs(f) >= std::abs(g)){
                    c = g / f;
                    r = std::sqrt(c*c + 1.0);
                    e[i+1] = f*r;
                    s = 1.0 / r;
                    c *= s;
                }else{
                    s = f / g;
                    r =  std::sqrt(s*s + 1.0);
                    e[i+1] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }

                g = d[i+1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i+1] = g + p;
                g = c * r - b;
                f = z[i+1];
                z[i+1] = s * z[i] + c * f;
                z[i] = c * z[i] - s * f;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;

            m = l;
            while((m < n-1) && (std::abs(e[m]) > tol)) m++;
        }
    }

    for(int i=1; i<n; i++){
        for(int j=0; j<n-1; j++){
            if (d[j] > d[j+1]){
                double p = d[j];
                d[j] = d[j+1];
                d[j+1] = p;
                p = z[j];
                z[j] = z[j+1];
                z[j+1] = p;
            }
        }
    }
    for(int i=0; i<n; i++){
           z[i] = z[i]*z[i];
    }
}

void TasmanianFourierTransform::fast_fourier_transform(std::vector<std::vector<std::complex<double>>> &data, std::vector<int> &num_points){
    int num_dimensions = (int) num_points.size();
    int num_total = 1;
    for(auto n: num_points) num_total *= n;
    std::vector<int> cumulative_points(num_dimensions);
    for(int k=0; k<num_dimensions; k++){
        // split the data into vectors of 1-D transforms
        std::vector<std::vector<int>> maps1d(num_total / num_points[k]); // total number of 1-D transforms
        for(auto &v: maps1d) v.reserve(num_points[k]); // each 1-D transform will have size num_points[k]

        std::fill(cumulative_points.begin(), cumulative_points.end(), 1);
        for(int j=num_dimensions-2; j>=0; j--) cumulative_points[j] = cumulative_points[j+1] * ((j+1 != k) ? num_points[j+1] : 1);

        // convert i to tuple, i.e., i -> p0, p1, p2 .. pd, where d = num_dimensions
        // all tuples that differ only in the k-th entry will belong to the same 1-D transform
        // the index of the 1D transform is i1d = \sum_{j \neq k} pj * cumulative_points[j]
        for(int i=0; i<num_total; i++){
            int t = i; // used to construct the tensor index
            int i1d = 0; // index of the corresponding 1D transform
            for(int j=num_dimensions-1; j>=0; j--){
                int pj = t % num_points[j]; // tensor index j
                if (j != k) i1d += cumulative_points[j] * pj;
                t /= num_points[j];
            }
            maps1d[i1d].push_back(i);
        }

        #pragma omp parallel for // perform the 1D transforms
        for(int i=0; i<(int) maps1d.size(); i++){
            fast_fourier_transform1D(data, maps1d[i]);
        }
    }
}

struct dcomplex{ // two doubles stored in the sse 128-bit registers
    #ifdef __AVX__
    typedef double mem_type __attribute__((__vector_size__(32)));
    inline __m256d mm_complex_mul(__m256d const &a, __m256d const &b) const{
        //std::cout << "using AVX" << std::endl;
        return _mm256_addsub_pd( _mm256_permute_pd(a, 0) * b, _mm256_permute_pd(a, 15) * _mm256_permute_pd(b, 5) );
    }
    inline __m256d mm_setzero() const{ return _mm256_setzero_pd(); }
    inline __m256d mm_set(double r, double i) const{ return _mm256_setr_pd(r, i, r, i); }
    inline __m256d mm_load(std::complex<double> const *a) const{ return _mm256_loadu_pd(reinterpret_cast<double const*>(a)); }
    inline void mm_store(std::complex<double> *a, __m256d const &d) const{ _mm256_storeu_pd(reinterpret_cast<double*>(a), d); }
    static constexpr int stride = 2;
    #elif defined(__SSE3__)
    typedef double mem_type __attribute__((__vector_size__(16)));
    inline __m128d mm_complex_mul(__m128d const &a, __m128d const &b) const{
        //std::cout << "using SSE3" << std::endl;
        __m128d direct_mult = _mm_mul_pd(a, b);
        __m128d shuffle_b = _mm_shuffle_pd(b, b, 1); // shuffle_b = flip two entries in b
        __m128d cross_mult = _mm_mul_pd(a, shuffle_b);
        return _mm_addsub_pd( _mm_shuffle_pd(direct_mult, cross_mult, 0), _mm_shuffle_pd(direct_mult, cross_mult, 3) );
    }
    inline __m128d mm_setzero() const{ return _mm_setzero_pd(); }
    inline __m128d mm_set(double r, double i) const{ return _mm_setr_pd(r, i); }
    inline __m128d mm_load(std::complex<double> const *a) const{ return _mm_loadu_pd(reinterpret_cast<double const*>(a)); }
    inline void mm_store(std::complex<double> *a, __m128d const &d) const{ _mm_storeu_pd(reinterpret_cast<double*>(a), d); }
    static constexpr int stride = 1;
    #else
    typedef std::complex<double> mem_type;
    inline std::complex<double> mm_complex_mul(std::complex<double> const &a, std::complex<double> const &b) const{
        //std::cout << "using no-vectorization" << std::endl;
        return a * b;
    }
    inline std::complex<double> mm_setzero() const{ return std::complex<double>(0.0, 0.0); }
    inline std::complex<double> mm_set(double r, double i) const{ return std::complex<double>(r, i); }
    inline std::complex<double> mm_load(std::complex<double> const *a) const{ return *a; }
    inline void mm_store(std::complex<double> *a, std::complex<double> const &b) const{ *a = b; }
    static constexpr int stride = 1;
    #endif
    mem_type data;

    dcomplex() : data(mm_setzero()) {}
    dcomplex(mem_type const &a) : data(a){}
    dcomplex(double r, double i) : data(mm_set(r, i)){}

    dcomplex(dcomplex const &) = default;
    dcomplex(dcomplex &&) = default;
    dcomplex& operator = (dcomplex const &) = default;
    dcomplex& operator = (dcomplex &&) = default;

    void load(std::complex<double> const *a){ data = mm_load(a); }
    void load(std::complex<double> const *a, int num){
        std::vector<std::complex<double>> full(stride, {0.0, 0.0});
        std::copy_n(a, num, full.begin());
        data = mm_load(full.data());
    }
    void store(std::complex<double> *a) const{ mm_store(a, data); }
    void store(std::complex<double> *a, int num) const{
        std::vector<std::complex<double>> full(stride);
        mm_store(full.data(), data);
        std::copy_n(full.begin(), num, a);
    }

    dcomplex operator + (dcomplex const &other) const{ return data + other.data; }
    dcomplex operator - (dcomplex const &other) const{ return data - other.data; }

    dcomplex operator * (dcomplex const &other) const{
        return mm_complex_mul(data, other.data);
    }

    dcomplex operator *= (dcomplex const &other){
        data = mm_complex_mul(data, other.data);
        return *this;
    }
};

template <class T>
class aligned_allocator{
public:
    //! \brief Type of the allocated entries.
    using value_type = T;
    //! \brief Type of the pointers.
    using pointer = T*;
    //! \brief Type of the const pointers.
    using const_pointer = T const*;
    //! \brief Type of the index size.
    using size_type = std::size_t;

    //! \brief Do not propagate on copy, simply create a new allocator.
    using propagate_on_container_copy_assignment = std::false_type;
    //! \brief Move the allocator on container move, preserves the state.
    using propagate_on_container_move_assignment = std::true_type;
    //! \brief Move the allocator on container swap, preserves the state.
    using propagate_on_container_swap            = std::true_type;

    //! \brief The beginning of an aligned block with within a \b stride of any array, matches \b hala::vex:mem_slice::stride.
    static constexpr size_t extra_elements = (size_t) dcomplex::stride;
    //! \brief Number of bytes to use for the alignment.
    static constexpr size_t alignment_byte = extra_elements * sizeof(T);

    //! \brief Default constructor, do not use the offsets.
    aligned_allocator() : offset1(alignment_byte + 1), offset2(alignment_byte + 1){}

    //! \brief Returns an aligned pointer to allocated block.
    pointer allocate(size_type n, const_pointer hint = 0){
        void *p = std::allocator<T>().allocate(n + extra_elements, hint);
        if (p == nullptr) throw std::bad_alloc();

        size_t pval = reinterpret_cast<size_t>(p);
        offset2 = (alignment_byte - pval % alignment_byte) % alignment_byte;
        if (offset1 > alignment_byte) offset1 = offset2;

        p = reinterpret_cast<void*>(pval + offset2);

        return static_cast<pointer>(p);
    }

    //! \brief Deletes the entire block associated with \b p.
    void deallocate(pointer p, size_type n){
        pointer padjusted = reinterpret_cast<pointer>( reinterpret_cast<size_t>(p) - offset1 );
        std::allocator<T>().deallocate(padjusted, n);
        offset1 = offset2;
    }

    //! \brief Shrink the number of template parameters.
    template<class U> struct rebind{
        //! \brief Defines the reduced parameter template parameter alias, per standard requirements.
        using other = aligned_allocator<U>;
    };

    //! \brief Do not propagate on copy, this constructor cannot deallocate memory from another allocator.
    aligned_allocator<T> select_on_container_copy_construction() const{
        return aligned_allocator<T>();
    }

    //! \brief Allocators for regtype none have no effective state, all other allocators are different.
    bool operator == (aligned_allocator<T> const &) noexcept{
        return (extra_elements == 1); // the none allocator has no real state
    }

    //! \brief Allocators for regtype none have no effective state, all other allocators are different.
    bool operator != (aligned_allocator<T> const &) noexcept{
        return (extra_elements != 1); // the none allocator has no real state
    }

private:
    size_t offset1, offset2;
};


void TasmanianFourierTransform::fast_fourier_transform1D(std::vector<std::vector<std::complex<double>>> &data, std::vector<int> &indexes){
    //
    // Given vector x_n with size N, the Fourier transform F_k is defined as: F_k = \sum_{n=0}^{N-1} \exp(- 2 \pi k n / N) x_n
    // Assuming that N = 3^l for some l, we can sub-divide the transform into strips of 3
    // let j = 0, 1, 2; let k = 0, .., N/3; let x_{0,m} = x_{3m}; let x_{1,m} = x_{3m+1}; and let x_{2,m} = x_{3m+2}
    // F_{k + j N / 3} =                                         \sum_{m=0}^{N/3 - 1} x_{0,m} \exp(-2 \pi k m / (N / 3))
    //                   + \exp(-2 \pi k / N) \exp(-2 \pi j / 3) \sum_{m=0}^{N/3 - 1} x_{1,m} \exp(-2 \pi k m / (N / 3))
    //                   + \exp(-4 \pi k / N) \exp(-4 \pi j / 3) \sum_{m=0}^{N/3 - 1} x_{2,m} \exp(-2 \pi k m / (N / 3))
    // The three sums are the Fourier coefficients of x_{0, m}, x_{1, m}, and x_{2, m}
    // The terms \exp(-2 \pi k / N) \exp(-2 \pi j / 3), and \exp(-4 \pi k / N) \exp(-4 \pi j / 3) are the twiddle factors
    // The procedure is recursive splitting the transform into small sets, all the way to size 3
    //
    constexpr int vlenght = dcomplex::stride;
    int num_outputs = (int) data[0].size(); // get the problem dimensions, num outputs and num entries for the 1D transform
    int num_outputs_short     = vlenght * (num_outputs / vlenght);
    int num_outputs_remainder = num_outputs - num_outputs_short;
    if (num_outputs_remainder > 0) num_outputs += vlenght - num_outputs_remainder;
    num_outputs /= vlenght;

    int num_entries = (int) indexes.size(); // the size of the 1D problem, i.e., N
    if (num_entries == 1) return; // nothing to do for size 1
    // a copy of the data is needed to swap back and forth, thus we make two copies and swap between them
    std::vector<std::vector<dcomplex, aligned_allocator<dcomplex>>> V(num_entries);
    auto v = V.begin();
    for(auto i: indexes){
        (*v).resize(num_outputs);
        // copy from the data only the indexes needed for the 1D transform
        for(int k=0; k<num_outputs_short; k+=vlenght)
            (*v)[k / vlenght].load(&data[i][k]);
        if (num_outputs_remainder > 0)
            (*v)[num_outputs - 1].load(&data[i][num_outputs_short], num_outputs_remainder);
        *v++;
    }
    std::vector<std::vector<dcomplex, aligned_allocator<dcomplex>>> W(num_entries);
    for(auto &w : W) w.resize(num_outputs); // allocate storage for the second data set

    // the radix-3 FFT algorithm uses two common twiddle factors from known angles +/- 2 pi/3
    dcomplex twidlep(-0.5, -std::sqrt(3.0) / 2.0); // angle of -2 pi/3
    dcomplex twidlem(-0.5,  std::sqrt(3.0) / 2.0); // angle of  2 pi/3 = -4 pi/3

    int stride = num_entries / 3; // the jump between entries, e.g., in one level of split stride is 3, split again and stride is 9 ... up to N / 3
    int length = 3;               // the number of entries in the sub-sequences, i.e., how large k can be (see above), smallest sub-sequence uses length 3

    for(int i=0; i<stride; i++){ // do the 3 transform, multiply by 3 by 3 matrix
        auto x1 = V[i].begin();
        auto x2 = V[i + stride].begin();
        auto x3 = V[i + 2 * stride].begin(); // x1, x2, x3 are the three entries of a sub-sequence

        auto y1 = W[i].begin();
        auto y2 = W[i + stride].begin();
        auto y3 = W[i + 2 * stride].begin(); // y1, y2, y3 are the resulting Fourier coefficients

        for(int k=0; k<num_outputs; k++){
            *y1++ = *x1 + *x2 + *x3;
            *y2++ = *x1 + twidlep * *x2 + twidlem * *x3;
            *y3++ = *x1 + twidlem * *x2 + twidlep * *x3;
            x1++; x2++; x3++;
        }
    }

    std::swap(V, W); // swap, now V contains the computed transform of the sub-sequences with size 3, W will be used for scratch space

    // merge smaller sequences, do the recursion
    while(stride / 3 > 0){ // when the stride that we just computed is equal to 1, then stop the recursion
        int biglength = 3 * length; // big sequence, i.e., F_k has this total length
        int bigstride = stride / 3;

        double theta = -2.0 * Maths::pi / ((double) biglength);
        dcomplex expstep(std::cos(theta), std::sin(theta)); // initialize the twiddle factors common for this level of sub-sequences
        dcomplex expstep2 = expstep * expstep;

        // merge sets of 3 sub-sequences (coefficients of x_{i,m}) into 3 pieces of one sequence F_{k + j N / 3}
        for(int i=0; i<bigstride; i++){ // total number of triples of sequences is bigstride
            dcomplex t01(1.0, 0.0);
            dcomplex t02(1.0, 0.0);

            dcomplex t11 = twidlep;
            dcomplex t12 = twidlem;

            dcomplex t21 = twidlem;
            dcomplex t22 = twidlep; // the twiddle factors form a 3 by 3 matrix [1, 1, 1; 1, t11, t12; 1, t21, t22;]

            for(int k=0; k<length; k++){ // number of entries in the sub-sequences
                auto x1 = V[i + k * stride].begin();
                auto x2 = V[i + k * stride + bigstride].begin();
                auto x3 = V[i + k * stride + 2 * bigstride].begin(); // x1, x2, x3 are the next entries of the sub-sequence (i.e., the sums)

                auto y1 = W[i + k * bigstride].begin();
                auto y2 = W[i + (k + length) * bigstride].begin();
                auto y3 = W[i + (k + 2 * length) * bigstride].begin(); // y1, y2, y3 are the F_{k + j N / 3}

                for(int o=0; o<num_outputs; o++){ // traverse through all the outputs
                    *y1++ = *x1 + t01 * *x2 + t02 * *x3;
                    *y2++ = *x1 + t11 * *x2 + t12 * *x3;
                    *y3++ = *x1 + t21 * *x2 + t22 * *x3;
                    x1++; x2++; x3++;
                }

                // update the twiddle factors for the next index k
                t01 *= expstep;
                t11 *= expstep;
                t21 *= expstep;
                t02 *= expstep2;
                t12 *= expstep2;
                t22 *= expstep2;
            }
        }

        std::swap(V, W); // swap the data, V holds the current set of indexes and W is the next set

        stride = bigstride;
        length = biglength;
    }

    // copy back the solution into the data structure
    v = V.begin();
    for(auto i : indexes){
        for(int k=0; k<num_outputs_short; k+=vlenght)
            (*v)[k / vlenght].store(&data[i][k]);
        if (num_outputs_remainder > 0)
            (*v)[num_outputs - 1].store(&data[i][num_outputs_short], num_outputs_remainder);
        *v++;
    }
}

// void TasmanianFourierTransform::fast_fourier_transform1D(std::vector<std::vector<std::complex<double>>> &data, std::vector<int> &indexes){
//     //
//     // Given vector x_n with size N, the Fourier transform F_k is defined as: F_k = \sum_{n=0}^{N-1} \exp(- 2 \pi k n / N) x_n
//     // Assuming that N = 3^l for some l, we can sub-divide the transform into strips of 3
//     // let j = 0, 1, 2; let k = 0, .., N/3; let x_{0,m} = x_{3m}; let x_{1,m} = x_{3m+1}; and let x_{2,m} = x_{3m+2}
//     // F_{k + j N / 3} =                                         \sum_{m=0}^{N/3 - 1} x_{0,m} \exp(-2 \pi k m / (N / 3))
//     //                   + \exp(-2 \pi k / N) \exp(-2 \pi j / 3) \sum_{m=0}^{N/3 - 1} x_{1,m} \exp(-2 \pi k m / (N / 3))
//     //                   + \exp(-4 \pi k / N) \exp(-4 \pi j / 3) \sum_{m=0}^{N/3 - 1} x_{2,m} \exp(-2 \pi k m / (N / 3))
//     // The three sums are the Fourier coefficients of x_{0, m}, x_{1, m}, and x_{2, m}
//     // The terms \exp(-2 \pi k / N) \exp(-2 \pi j / 3), and \exp(-4 \pi k / N) \exp(-4 \pi j / 3) are the twiddle factors
//     // The procedure is recursive splitting the transform into small sets, all the way to size 3
//     //
//     int num_outputs = (int) data[0].size(); // get the problem dimensions, num outputs and num entries for the 1D transform
//     int num_entries = (int) indexes.size(); // the size of the 1D problem, i.e., N
//     if (num_entries == 1) return; // nothing to do for size 1
//     // a copy of the data is needed to swap back and forth, thus we make two copies and swap between them
//     std::vector<std::vector<std::complex<double>>> V(num_entries);
//     auto v = V.begin();
//     for(auto i: indexes) *v++ = data[i]; // copy from the data only the indexes needed for the 1D transform
//     std::vector<std::vector<std::complex<double>>> W(num_entries);
//     for(auto &w : W) w.resize(num_outputs); // allocate storage for the second data set
//
//     // the radix-3 FFT algorithm uses two common twiddle factors from known angles +/- 2 pi/3
//     std::complex<double> twidlep(-0.5, -std::sqrt(3.0) / 2.0); // angle of -2 pi/3
//     std::complex<double> twidlem(-0.5,  std::sqrt(3.0) / 2.0); // angle of  2 pi/3 = -4 pi/3
//
//     int stride = num_entries / 3; // the jump between entries, e.g., in one level of split stride is 3, split again and stride is 9 ... up to N / 3
//     int length = 3;               // the number of entries in the sub-sequences, i.e., how large k can be (see above), smallest sub-sequence uses length 3
//
//     for(int i=0; i<stride; i++){ // do the 3 transform, multiply by 3 by 3 matrix
//         auto x1 = V[i].begin();
//         auto x2 = V[i + stride].begin();
//         auto x3 = V[i + 2 * stride].begin(); // x1, x2, x3 are the three entries of a sub-sequence
//
//         auto y1 = W[i].begin();
//         auto y2 = W[i + stride].begin();
//         auto y3 = W[i + 2 * stride].begin(); // y1, y2, y3 are the resulting Fourier coefficients
//
//         for(int k=0; k<num_outputs; k++){
//             *y1++ = *x1 + *x2 + *x3;
//             *y2++ = *x1 + twidlep * *x2 + twidlem * *x3;
//             *y3++ = *x1 + twidlem * *x2 + twidlep * *x3;
//             x1++; x2++; x3++;
//         }
//     }
//
//     std::swap(V, W); // swap, now V contains the computed transform of the sub-sequences with size 3, W will be used for scratch space
//
//     // merge smaller sequences, do the recursion
//     while(stride / 3 > 0){ // when the stride that we just computed is equal to 1, then stop the recursion
//         int biglength = 3 * length; // big sequence, i.e., F_k has this total length
//         int bigstride = stride / 3;
//
//         double theta = -2.0 * Maths::pi / ((double) biglength);
//         std::complex<double> expstep(std::cos(theta), std::sin(theta)); // initialize the twiddle factors common for this level of sub-sequences
//         std::complex<double> expstep2 = expstep * expstep;
//
//         // merge sets of 3 sub-sequences (coefficients of x_{i,m}) into 3 pieces of one sequence F_{k + j N / 3}
//         for(int i=0; i<bigstride; i++){ // total number of triples of sequences is bigstride
//             std::complex<double> t01(1.0, 0.0);
//             std::complex<double> t02(1.0, 0.0);
//
//             std::complex<double> t11 = twidlep;
//             std::complex<double> t12 = twidlem;
//
//             std::complex<double> t21 = twidlem;
//             std::complex<double> t22 = twidlep; // the twiddle factors form a 3 by 3 matrix [1, 1, 1; 1, t11, t12; 1, t21, t22;]
//
//             for(int k=0; k<length; k++){ // number of entries in the sub-sequences
//                 auto x1 = V[i + k * stride].begin();
//                 auto x2 = V[i + k * stride + bigstride].begin();
//                 auto x3 = V[i + k * stride + 2 * bigstride].begin(); // x1, x2, x3 are the next entries of the sub-sequence (i.e., the sums)
//
//                 auto y1 = W[i + k * bigstride].begin();
//                 auto y2 = W[i + (k + length) * bigstride].begin();
//                 auto y3 = W[i + (k + 2 * length) * bigstride].begin(); // y1, y2, y3 are the F_{k + j N / 3}
//
//                 for(int o=0; o<num_outputs; o++){ // traverse through all the outputs
//                     *y1++ = *x1 + t01 * *x2 + t02 * *x3;
//                     *y2++ = *x1 + t11 * *x2 + t12 * *x3;
//                     *y3++ = *x1 + t21 * *x2 + t22 * *x3;
//                     x1++; x2++; x3++;
//                 }
//
//                 // update the twiddle factors for the next index k
//                 t01 *= expstep;
//                 t11 *= expstep;
//                 t21 *= expstep;
//                 t02 *= expstep2;
//                 t12 *= expstep2;
//                 t22 *= expstep2;
//             }
//         }
//
//         std::swap(V, W); // swap the data, V holds the current set of indexes and W is the next set
//
//         stride = bigstride;
//         length = biglength;
//     }
//
//     // copy back the solution into the data structure
//     v = V.begin();
//     for(auto i : indexes) data[i] = *v++;
// }

namespace TasSparse{

SparseMatrix::SparseMatrix() : tol(Maths::num_tol), num_rows(0){}
SparseMatrix::~SparseMatrix(){}

void SparseMatrix::load(const std::vector<int> &lpntr, const std::vector<std::vector<int>> &lindx, const std::vector<std::vector<double>> &lvals){
    num_rows = (int) lpntr.size();

    pntr.resize(num_rows+1);
    pntr[0] = 0;
    for(int i=0; i<num_rows; i++)
        pntr[i+1] = pntr[i] + lpntr[i];

    int num_nz = pntr[num_rows];
    indx.resize(num_nz);
    vals.resize(num_nz);

    int j = 0;
    for(const auto &idx : lindx) for(auto i : idx) indx[j++] = i;
    j = 0;
    for(const auto &vls : lvals) for(auto v : vls) vals[j++] = v;

    computeILU();
}

int SparseMatrix::getNumRows() const{ return num_rows; }

void SparseMatrix::computeILU(){
    indxD.resize(num_rows);
    ilu.resize(pntr[num_rows]);
    for(int i=0; i<num_rows; i++){
        int j = pntr[i];
        while(indx[j] < i){ j++; };
        indxD[i] = j;
    }

    ilu = vals;

    for(int i=0; i<num_rows-1; i++){
        double u = ilu[indxD[i]];
        #pragma omp parallel for
        for(int j=i+1; j<num_rows; j++){ // update the rest of the matrix, each row can be done in parallel
            int jc = pntr[j];
            while(indx[jc] < i){ jc++; }
            if (indx[jc] == i){
                ilu[jc] /= u;
                double l = ilu[jc];
                int ik = indxD[i]+1;
                int jk = jc+1;
                while((ik<pntr[i+1]) && (jk<pntr[j+1])){
                    if (indx[ik] == indx[jk]){
                        ilu[jk] -= l * ilu[ik];
                        ik++; jk++;
                    }else if (indx[ik] < indx[jk]){
                        ik++;
                    }else{
                        jk++;
                    }
                }
            }
        }
    }
}

void SparseMatrix::solve(const double b[], double x[], bool transposed) const{ // ustd::sing GMRES
    int max_inner = 30;
    int max_outer = 80;
    std::vector<double> W((max_inner+1) * num_rows); // Krylov basis

    std::vector<double> H(max_inner * (max_inner+1)); // holds the transformation for the normalized basis
    std::vector<double> S(max_inner); // std::sin and std::cos of the Givens rotations
    std::vector<double> C(max_inner+1);
    std::vector<double> Z(max_inner); // holds the coefficients of the solution

    double alpha, h_k; // temp variables

    double outer_res = tol + 1.0; // outer and inner residual
    int outer_itr = 0; // counts the inner and outer iterations

    std::vector<double> pb(num_rows);
    if (!transposed){
        std::copy(b, b + num_rows, pb.data());

        // action of the preconditioner
        for(int i=1; i<num_rows; i++){
            for(int j=pntr[i]; j<indxD[i]; j++){
                pb[i] -= ilu[j] * pb[indx[j]];
            }
        }
        for(int i=num_rows-1; i>=0; i--){
            for(int j=indxD[i]+1; j<pntr[i+1]; j++){
                pb[i] -= ilu[j] * pb[indx[j]];
            }
            pb[i] /= ilu[indxD[i]];
        }
    }

    std::fill(x, x + num_rows, 0.0); // zero initial guess, I wonder if we can improve this

    while ((outer_res > tol) && (outer_itr < max_outer)){
        for(int i=0; i<num_rows; i++) W[i] = 0.0;
        if (transposed){
            std::copy(x, x + num_rows, pb.data());
            for(int i=0; i<num_rows; i++){
                pb[i] /= ilu[indxD[i]];
                for(int j=indxD[i]+1; j<pntr[i+1]; j++){
                    pb[indx[j]] -= ilu[j] * pb[i];
                }
            }
            for(int i=num_rows-2; i>=0; i--){
                for(int j=pntr[i]; j<indxD[i]; j++){
                    pb[indx[j]] -= ilu[j] * pb[i];
                }
            }
            for(int i=0; i<num_rows; i++){
                for(int j=pntr[i]; j<pntr[i+1]; j++){
                    W[indx[j]] += vals[j] * pb[i];
                }
            }
            for(int i=0; i<num_rows; i++){
                W[i] = b[i] - W[i];
            }
        }else{
            for(int i=0; i<num_rows; i++){
                for(int j=pntr[i]; j<pntr[i+1]; j++){
                    W[i] += vals[j] * x[indx[j]];
                }
            }
            for(int i=1; i<num_rows; i++){
                for(int j=pntr[i]; j<indxD[i]; j++){
                    W[i] -= ilu[j] * W[indx[j]];
                }
            }
            for(int i=num_rows-1; i>=0; i--){
                for(int j=indxD[i]+1; j<pntr[i+1]; j++){
                    W[i] -= ilu[j] * W[indx[j]];
                }
                W[i] /= ilu[indxD[i]];
            }

            for(int i=0; i<num_rows; i++){
                W[i] = pb[i] - W[i];
            }
        }

        Z[0] = 0.0;  for(int i=0; i<num_rows; i++){  Z[0] += W[i]*W[i];  };  Z[0] = std::sqrt(Z[0]);
        for(int i=0; i<num_rows; i++){  W[i] /= Z[0]; };

        double inner_res = Z[0]; // first residual
        //std::cout << Z[0] << std::endl;
        int inner_itr = 0; // counts the size of the basis

        while ((inner_res > tol) && (inner_itr < max_inner-1)){
            inner_itr++;

            std::fill(&(W[inner_itr*num_rows]), &(W[inner_itr*num_rows]) + num_rows, 0.0);
            if (transposed){
                std::copy(&(W[num_rows*(inner_itr-1)]), &(W[num_rows*(inner_itr-1)]) + num_rows, pb.data());
                for(int i=0; i<num_rows; i++){
                    pb[i] /= ilu[indxD[i]];
                    for(int j=indxD[i]+1; j<pntr[i+1]; j++){
                        pb[indx[j]] -= ilu[j] * pb[i];
                    }
                }
                for(int i=num_rows-2; i>=0; i--){
                    for(int j=pntr[i]; j<indxD[i]; j++){
                        pb[indx[j]] -= ilu[j] * pb[i];
                    }
                }
                for(int i=0; i<num_rows; i++){
                    for(int j=pntr[i]; j<pntr[i+1]; j++){
                        W[inner_itr*num_rows + indx[j]] += vals[j] * pb[i];
                    }
                }
            }else{
                for(int i=0; i<num_rows; i++){
                    for(int j=pntr[i]; j<pntr[i+1]; j++){
                        W[inner_itr*num_rows + i] += vals[j] * W[num_rows*(inner_itr-1) + indx[j]];
                    }
                }
                for(int i=1; i<num_rows; i++){
                    for(int j=pntr[i]; j<indxD[i]; j++){
                        W[inner_itr*num_rows + i] -= ilu[j] * W[inner_itr*num_rows + indx[j]];
                    }
                }
                for(int i=num_rows-1; i>=0; i--){
                    for(int j=indxD[i]+1; j<pntr[i+1]; j++){
                        W[inner_itr*num_rows + i] -= ilu[j] * W[inner_itr*num_rows + indx[j]];
                    }
                    W[inner_itr*num_rows + i] /= ilu[indxD[i]];
                }
            }

            #pragma omp parallel for
            for(int i=0; i<inner_itr; i++){
                H[i*max_inner + inner_itr-1] = 0.0; for(int j=0; j<num_rows; j++){  H[i*max_inner + inner_itr-1] += W[inner_itr*num_rows+j] * W[i*num_rows+j];  };
            }

            #pragma omp parallel for
            for(int j=0; j<num_rows; j++){
                for(int i=0; i<inner_itr; i++){
                    W[inner_itr*num_rows+j] -= H[i*max_inner + inner_itr-1] * W[i*num_rows+j];
                }
            };

            h_k = 0.0;  for(int i=0; i<num_rows; i++){  h_k += W[inner_itr*num_rows+i]*W[inner_itr*num_rows+i];  }; h_k = std::sqrt(h_k); //std::cout << "h_k = " << h_k << "  itr = " << inner_itr << std::endl;
            if (h_k > 0.0) for(int i=0; i<num_rows; i++){ W[inner_itr*num_rows+i] /= h_k; };

            for (int i=0; i<inner_itr-1; i++){ // form the next row of the transformation
                alpha = H[i*max_inner + inner_itr-1];
                H[   i*max_inner + inner_itr-1] = C[i] * alpha + S[i] * H[(i+1)*max_inner + inner_itr-1];
                H[(i+1)*max_inner + inner_itr-1] = S[i] * alpha - C[i] * H[(i+1)*max_inner + inner_itr-1];
            };

            alpha = std::sqrt(h_k * h_k  +  H[(inner_itr-1)*max_inner + inner_itr-1] * H[(inner_itr-1)*max_inner + inner_itr-1]);

            // set the next set of Givens rotations
            S[inner_itr-1] = h_k / alpha;
            C[inner_itr-1] = H[(inner_itr-1)*max_inner + inner_itr-1] / alpha;

            H[(inner_itr-1) * max_inner + inner_itr-1] = alpha;

            // Z is used to reconstruct the solution in the end
            Z[inner_itr] = S[inner_itr-1]*Z[inner_itr-1];
            Z[inner_itr-1] = C[inner_itr-1]*Z[inner_itr-1]; // apply it on z

            inner_res = std::abs(Z[inner_itr]);
        }

        inner_itr--;

        if (inner_itr > -1){ // if the first guess was not within TOL of the true solution
            Z[inner_itr] /= H[inner_itr * max_inner + inner_itr];
            for(int i=inner_itr-1; i>-1; i--){
                h_k = 0.0;
                for(int j=i+1; j<=inner_itr; j++){
                    h_k += H[i*max_inner + j] * Z[j];
                };
                Z[i] = (Z[i] - h_k) / H[i * max_inner + i];
            }

            for(int i=0; i<=inner_itr; i++){
                for(int j=0; j<num_rows; j++){
                    x[j] += Z[i] * W[i*num_rows+j];
                }
            }

            if (transposed){
                for(int i=0; i<num_rows; i++){
                    x[i] /= ilu[indxD[i]];
                    for(int j=indxD[i]+1; j<pntr[i+1]; j++){
                        x[indx[j]] -= ilu[j] * x[i];
                    }
                }
                for(int i=num_rows-2; i>=0; i--){
                    for(int j=pntr[i]; j<indxD[i]; j++){
                        x[indx[j]] -= ilu[j] * x[i];
                    }
                }
            }

        }

        outer_res = inner_res;
        outer_itr++;
    }
}

} /* namespace TasSparse */

}
#endif
