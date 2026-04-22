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

#ifndef __TASMANIAN_SPARSE_GRID_ENUMERATES_HPP
#define __TASMANIAN_SPARSE_GRID_ENUMERATES_HPP

// system headers used in many/many places
#ifdef _MSC_VER
// without this MSVC++ does not accepts keywords such as "not" and "or"
#include <iso646.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>


#include "TasmanianConfig.hpp" // contains build options passed down from CMake
#include "tsgUtils.hpp" // contains small utilities

/*!
 * \internal
 * \file tsgEnumerates.hpp
 * \brief Omnipresent enumerate types.
 * \author Miroslav Stoyanov
 * \ingroup TasmanianEnumerates
 *
 * Contains the enumerate types that are used throughout Tasmanian.
 * This header is included directly or transitively in every other
 * file of the project including \b TasmanianSparseGrids.hpp
 * \endinternal
 */

/*!
 * \ingroup TasmanianSG
 * \addtogroup SGEnumerates Enumerated types
 *
 * \par Enumerated types
 * Enumerations are used as input to many Tasmanian functions
 * in both internal and external API.
 */

namespace TasGrid{

    /*!
     * \internal
     * \ingroup TasmanianEnumerates
     * \brief Defines the integer used by the LAPACK methods, usually int but DPC++ uses int64_t.
     *
     * \endinternal
     */
#ifdef Tasmanian_ENABLE_DPCPP
    using int_gpu_lapack = std::int64_t;
#else
    using int_gpu_lapack = int;
#endif

//! \internal
//! \brief Describes the relation between two multi-indexes when compared during sorting.
//! \ingroup SGEnumerates

//! The standard C++11 algorithms std::merge and std::binary_search use bools as return
//! types when comparing entries, thus in the case of many repeated entries, two
//! comparisons have to be performed. Repeated entries are encountered in many sparse
//! grids algorithms, especially when dealing with nested one dimensional rules.
//! Thus, for performance reasons, Tasmanian implements merging and searching
//! methods that compare multi-indexes and produce three outcomes in a single comparison.
//!
//! Specifically, see \b SetManipulations::push_merge_map() and \b MultiIndexSet::getSlot()
//! and \b MultiIndexSet::addSortedInsexes()
enum TypeIndexRelation{ // internal for IndexSets, unambiguous set comparison
    //! \brief Indicates that the first multi-index in the call to compare comes before the second.
    type_abeforeb,
    //! \brief Indicates that the second multi-index in the call to compare after the second.
    type_bbeforea,
    //! \brief Indicates that the two multi-indexes are the same.
    type_asameb
};

/*!
 * \brief Strong type for the number of dimensions
 * \ingroup SGEnumerates
 */
enum class NumDimensions : int {};

/*!
 * \brief Strong type for the number of outputs
 * \ingroup SGEnumerates
 */
enum class NumOutputs : int {};

/*!
 * \brief Strong type for the depth (e.g., level/polynomial-order) of the grid
 * \ingroup SGEnumerates
 */
enum class NumDepth : int {};

/*!
 * \brief Strong type for the maximum polynomial order of the grid
 * \ingroup SGEnumerates
 */
enum class NumOrder : int {};

/*!
 * \brief Used by Global Sequence and Fourier grids, indicates the selection criteria.
 * \ingroup SGEnumerates
 *
 * \par Approximation Error
 * The approximation error when interpolating or integrating a target model with
 * a \b Global sparse grid,
 * is bound by a constant multiplied by the best approximation of the model in the
 * set of polynomial exactness. The exactness set consists of all polynomials that
 * can be \a captured exactly (approximated to round-off error) by the sparse grid.
 * The constant is the
 * norm of the sparse grid operator (a.k.a., the Lebesgue constant)
 * and the exactness set is the range.
 *
 * \par
 * The operator norm depends primarily on the underlying one dimensional rule used
 * to construct the grid, and all rules implemented in Tasmanian have reasonable norms.
 * Of primary focus is the target model and constructing a grid with range suitable
 * for either integrating or interpolating the model.
 *
 * \par Categories
 * The types of \b Global grids fall into four categories:
 * - \b Level selection uses the order of the one dimensional rules as criteria
 *   and pays no regard to the polynomial exactness. \b Level types are strongly influenced
 *   by the growth of points in the one dimensional rule and are seldom optimal,
 *   although such grid can be a good initial guess for an iterative refinement process.
 * - \b Total degree polynomial space is good for smooth models, i.e., analytic functions
 *   with convergent Taylor series for every point in the domain.
 * - \b Curved polynomial space is essentially a \b Total degree space with an
 *   added logarithmic correction term. The correction can account for rules with slightly
 *   higher norm growth (e.g., R-Leja rules) or for models with heavily constrained
 *   region of analytic extension, i.e., slightly less smooth.
 * - \b Hyperbolic cross section space is suitable for models with more irregular behavior, e.g.,
 *   models with only a finite number of derivatives.
 *
 * \par
 * The four main types can be tuned for either interpolation or quadrature (integration);
 * the corresponding enumerate will have either \b i or \b q in the name.
 * In all cases, Tasmanian will construct the grid with minimum number of points
 * (or fewest basis functions), that will \a capture the corresponding polynomial space.
 *
 * \par Anisotropic Weights
 * All types can be combined with anisotropic weights that adjust the density
 * of the grid in different directions. Below, we denote the anisotropic weights
 * with \f$ \xi_1, \xi_2, \cdots, \xi_d \f$ and \f$ \eta_1, \eta_2, \cdots, \eta_d \f$,
 * where \f$ d \f$ is the number of dimensions in the grid. Note that only curved
 * types use the \f$ \eta \f$ weights.
 *
 * \par
 * The \b TypeDepth also uses the \b depth parameter to define a cut-off point, the
 * \b depth is normalized by multiplying it by the smallest of the \f$ \xi \f$ weights.
 * In the formulas below, \b L stands for the normalized offset.
 *
 * Details regarding sparse grids construction and polynomial spaces can be found in\n
 * M. Stoyanov, C. G. Webster, <a style="font-weight:bold" href="https://arxiv.org/pdf/1508.01125.pdf">
 * A Dynamically Adaptive Sparse Grid Method for Quasi-Optimal
 * Interpolation of Multidimensional Analytic Functions</a>, arXiv:1508.01125.\n
 * also published in <a style="font-weight:bold" href="https://www.sciencedirect.com/science/article/pii/S0898122116000031">
 * Computers and Mathematics with Applications 71 (2016) 2449–2465.</a>
 *
 * \par Full Tensor Types
 * Finally, in addition to the aforementioned types, Tasmanian includes the types for
 * dense tensor grids. Only in 2-dimensions Gauss quadrature rules with full tensors
 * are more efficient than a general sparse grid. Performing interpolation or
 * working with more than 2 dimensions, sparse grids will always result in better
 * approximation error per number of nodes, i.e., model evaluations; nevertheless,
 * dense tensor grids naturally fall in the framework and can be useful for testing
 * purposes.
 *
 * \par Fourier Grids
 * The Fourier grids are very similar to Global grids, with the exception that
 * polynomial powers are replaced by frequencies of trigonometric functions.
 * The different selection strategies can be used with frequency exactness in
 * place of polynomial power.
 *
 * The structure of the Fourier spaces, the convergence estimates, and
 * the refinement strategies are described in\n
 * Z. Morrow, M. Stoyanov, <a style="font-weight:bold" href="https://arxiv.org/abs/1908.10672">A Method for
 * Dimensionally Adaptive Sparse Trigonometric Interpolation of Periodic Functions</a>,
 * arXiv:1908.10672.
 */
enum TypeDepth{
    //! \brief Null type, should \b never be used for input, indicates an error of some sort.
    type_none,

    //! \brief Ignoring the polynomial space, use rules with index
    //! \f$ \left\{ (i_1, i_2, \cdots, i_d) : \sum_{k=1}^d \xi_k i_k \leq L \right\} \f$
    type_level,

    //! \brief Ignoring the polynomial space, use rules with index \n
    //! \f$ \left\{ (i_1, i_2, \cdots, i_d) : \sum_{k=1}^d \xi_k i_k + \sum_{k=1}^d \eta_k \log(i_k + 1) \leq L \right\} \f$
    type_curved,

    //! \brief Ignoring the polynomial space, use rules with index
    //! \f$ \left\{ (i_1, i_2, \cdots, i_d) : \prod_{k=1}^d (i_k + 1)^{\xi_k} \leq L \right\} \f$
    type_hyperbolic,

    //! \brief Total degree polynomial space for interpolation, i.e., the span of \n
    //! \f$ \left\{ \prod_{k=1}^d x^{i_k} : \sum_{k=1}^d \xi_k i_k \leq L \right\} \f$
    type_iptotal,

    //! \brief Total degree polynomial space for quadrature/integration.
    type_qptotal,

    //! \brief Curved polynomial space for interpolation, i.e., the span of \n
    //! \f$ \left\{ \prod_{k=1}^d x^{i_k} : \sum_{k=1}^d \xi_k i_k +  \sum_{k=1}^d \eta_k \log(i_k + 1) \leq L \right\} \f$
    type_ipcurved,

    //! \brief Curved polynomial space for quadrature/integration.
    type_qpcurved,

    //! \brief Hyperbolic cross section polynomial space for interpolation, i.e., the span of \n
    //! \f$ \left\{ \prod_{k=1}^d x^{i_k} : \prod_{k=1}^d (i_k + 1)^{\xi_k} \leq L \right\} \f$
    type_iphyperbolic,

    //! \brief Hyperbolic cross section polynomial space for quadrature/integration.
    type_qphyperbolic,

    //! \brief Make a dense tensor grid with rules indexed by \f$ \xi_1, \xi_2, \cdots, \xi_d \f$
    type_tensor,

    //! \brief Make a dense tensor grid with interpolation range that includes
    //! \f$ \left\{ \prod_{k=1}^d x^{i_k} : i_k \leq \xi_k \mbox{ for all } k \right\} \f$
    type_iptensor,

    //! \brief Make a dense tensor grid with quadrature/integration range that includes
    //! \f$ \left\{ \prod_{k=1}^d x^{i_k} : i_k \leq \xi_k \mbox{ for all } k \right\} \f$
    type_qptensor
};

//! \brief Used to specify the one dimensional family of rules that induces the sparse grid.
//! \ingroup SGEnumerates

//! \par One Dimensional Rules
//! A sparse grid is a superposition of tensors of one dimensional rules with varying precision
//! in the different directions. In this context, a \b TypeOneDRule defines a family of one
//! dimensional rules with corresponding abscissas (nodes), quadrature weights, and
//! basis functions.
//!
//! \par Categories
//! The rules fall into 5 categories, corresponding to the 5 main classes of grids.
//! - \b Global using Lagrange polynomials for interpolation and (sometimes) specialized
//!   quadrature weights.
//! - \b Sequence rules are a subset of the \b Global rules, \b Sequence rules are always
//!   nested and each level adds one node to the previous level.
//! - \b Local \b Polynomial rules use uniformly spaced nodes and polynomial basis with
//!   localized support.
//! - \b Wavelet and \b Fourier rules also use uniformly spaced nodes, but with
//!   wavelet and trigonometric basis functions.
//!
//! \par Special Quadrature
//! All rules can be used to approximate integrals of the form \f$ \int_\Gamma f(x) \rho(x) dx \f$
//! where \f$ \Gamma \f$ is the (transformed) domain and \f$ f(x) \f$ is the model.
//! The majority of rules assume that the weight is constant 1, but in some cases
//! more exotic weights are used. Assume that constant weight 1 is used, unless otherwise
//! specified.
//!
//! \par Domain Transformation
//! Most rules are defined on a canonical interval [-1, 1], which can be transformed to an
//! arbitrary [a, b] with \b TasmanianSparseGrid::setDomainTransform(), so long as \b a is less than \b b.
//! Fourier grids use canonical [0, 1] and can be similarly transformed to [a, b]. Gauss-Laguerre rule
//! is set to \f$ [0, \infty) \f$ transformed to \f$ [a, \infty) \f$ with scale parameter \b b (see below).
//! Gauss-Hermite is set to \f$ (-\infty, \infty) \f$ with scale and shift parameters.
enum TypeOneDRule{
    //! \brief Null rule, should \b never be used as input (default rule for an empty grid).
    rule_none,
    //! \brief Classic nested rule using Chebyshev nodes with very low Lebesgue constant.
    rule_clenshawcurtis,
    //! \brief Same as \b rule_clenshawcurtis but with modified basis that assumes the model is zero at the boundary.
    rule_clenshawcurtis0,
    //! \brief Similar to \b rule_clenshawcurtis but with nodes strictly in the interior.
    rule_fejer2,
    //! \brief Using Chebyshev nodes with very low Lebesgue constant and slow node growth, but non-nested.
    rule_chebyshev,
    //! \brief Same as \b rule_chebyshev but using only odd levels, partially mitigates the non-nested issues.
    rule_chebyshevodd,
    //! \brief Classic \b sequence rule, moderate Lebesgue constant growth (empirical result only).
    rule_leja,
    //! \brief Same as \b rule_leja but using only odd levels, quadrature is more stable.
    rule_lejaodd,
    //! \brief Classic \b sequence rule based on complex analysis, moderate Lebesgue constant growth (theoretically proven).
    rule_rleja,
    //! \brief Using \b rule_rleja nodes but doubling the nodes every 2 levels, reduces the Lebesgue constant.
    rule_rlejadouble2,
    //! \brief Using \b rule_rleja nodes but doubling the nodes every 4 levels, reduces the Lebesgue constant.
    rule_rlejadouble4,
    //! \brief Same as \b rule_rleja but using only odd levels, quadrature is more stable.
    rule_rlejaodd,
    //! \brief Similar \b sequence to \b rule_rleja but with nodes strictly in the interior.
    rule_rlejashifted,
    //! \brief Same as \b rule_rlejashifted but using only even levels, quadrature is more stable.
    rule_rlejashiftedeven,
    //! \brief Same as \b rule_rlejashifted but doubling the number of nodes per level, which reduced the Lebesgue constant.
    rule_rlejashifteddouble,
    //! \brief A greedy \b sequence rule with nodes placed at the maximum of the Lebesgue function.
    rule_maxlebesgue,
    //! \brief Same as \b rule_maxlebesgue but using only odd levels, quadrature is more stable.
    rule_maxlebesgueodd,
    //! \brief A greedy \b sequence rule with nodes added to minimize the Lebesgue constant.
    rule_minlebesgue,
    //! \brief Same as \b rule_minlebesgue but using only odd levels, quadrature is more stable.
    rule_minlebesgueodd,
    //! \brief A greedy \b sequence rule with nodes added to minimize the norm of the surplus operator.
    rule_mindelta,
    //! \brief Same as \b rule_mindelta but using only odd levels, quadrature is more stable.
    rule_mindeltaodd,
    //! \brief Non-nested rule but optimized for integration.
    rule_gausslegendre,
    //! \brief Same as \b rule_gausslegendre but using only odd levels, partially mitigates the non-nested issues.
    rule_gausslegendreodd,
    //! \brief Nested rule that is optimized for integration, probably the best integration rule in more than 2 dimensions.
    rule_gausspatterson,
    //! \brief Non-nested rule optimized for integral of the form \f$ \int_{a}^{b} f(x) / \sqrt{(b - x)(x - a)} dx \f$.
    rule_gausschebyshev1,
    //! \brief Same as \b rule_gausschebyshev1 but using only odd levels, partially mitigates the non-nested issues.
    rule_gausschebyshev1odd,
    //! \brief Non-nested rule optimized for integral of the form \f$ \int_{a}^{b} f(x) \sqrt{(b - x)(x - a)} dx \f$.
    rule_gausschebyshev2,
    //! \brief Same as \b rule_gausschebyshev2 but using only odd levels, partially mitigates the non-nested issues.
    rule_gausschebyshev2odd,
    //! \brief Non-nested rule optimized for integral of the form \f$ \int_{a}^{b} f(x) (b - x)^\alpha (x - a)^\alpha dx \f$.
    rule_gaussgegenbauer,
    //! \brief Same as \b rule_gaussgegenbauer but using only odd levels, partially mitigates the non-nested issues.
    rule_gaussgegenbauerodd,
    //! \brief Non-nested rule optimized for integral of the form \f$ \int_{a}^{b} f(x) (b - x)^\alpha (x - a)^\beta dx \f$.
    rule_gaussjacobi,
    //! \brief Same as \b rule_gaussjacobi but using only odd levels, partially mitigates the non-nested issues.
    rule_gaussjacobiodd,
    //! \brief Non-nested rule optimized for integral of the form \f$ \int_{a}^{\infty} f(x) (x - a)^\alpha \exp \left(-b (x - a) \right) dx \f$.
    rule_gausslaguerre,
    //! \brief Same as \b rule_gausslaguerre but using only odd levels, partially mitigates the non-nested issues.
    rule_gausslaguerreodd,
    //! \brief Non-nested rule optimized for integral of the form \f$ \int_{-\infty}^{\infty} f(x) (x - a)^\alpha \exp \left(-b (x - a)^2 \right) dx \f$.
    rule_gausshermite,
    //! \brief Same as \b rule_gausshermite but using only odd levels, partially mitigates the non-nested issues.
    rule_gausshermiteodd,
    //! \brief User provided rule, nodes and weights must be provided with a separate file.
    rule_customtabulated,
    // Piece-Wise rules
    //! \brief Nested rule with a hierarchy of uniformly distributed nodes and functions with compact support.
    rule_localp,
    //! \brief Variation of \b rule_localp assuming the model is zero at the domain boundary.
    rule_localp0,
    //! \brief Variation of \b rule_localp using increased support in exchange for higher order basis (better for smoother functions).
    rule_semilocalp,
    //! \brief Variation of \b rule_localp focusing nodes on the boundary instead of the interior.
    rule_localpb,
    //! \brief Wavelet basis with uniformly distributed nodes (primarily for internal use).
    rule_wavelet,
    //! \brief Trigonometric basis with uniformly distributed nodes (primarily for internal use).
    rule_fourier
};

//! \brief Refinement strategy for local polynomial and wavelet grids.
//! \ingroup SGEnumerates

//! \par Local Hierarchical Approximation
//! The nodes and basis functions used in local polynomial and wavelet sparse grid form a complex multidimensional hierarchy,
//! where each node is associated with one or more parents and children in each direction. In both cases, the support of the
//! children is (almost always) smaller than the support of the patent and in the polynomial case, the support of the
//! children nodes is strictly contained within the support of the parent.
//!
//! \par
//! The interpolant is expressed as a linear combination of basis functions with coefficients computed so that the
//! interpolant matches the model values at the sparse gird nodes. The coefficients are thus indicators of the local
//! approximation error and (in the asymptotic limit) the coefficients decrease when going form a parent to a child.
//!
//! \par Convergence Criteria
//! Using the local error estimator, the sparse grid approximation has reached desired tolerance
//! when all nodes with missing children
//! have coefficients with absolute values smaller than the tolerance. This is a necessary and not a sufficient condition
//! an is heavily reliant on the assumption of monotonic decay of the coefficients; however, in practical situations this is a good
//! way to construct an adaptive grid that captures the behavior of the target model while minimizing the number of nodes.
//!
//! \par Adaptive Refinement
//! The idea of refinement is to start with a coarse spars grid and gradually add children of existing nodes, but only if the nodes
//! have a large hierarchical coefficient. The nodes are added until the coefficients of all nodes with with missing children
//! drop below user specified tolerance. The refinement types listed below differ in the way that the parents and children
//! are selected in the adaptive construction, and whether local anisotropy is considered.
//!
//! \par Refinement Types
//! Each nodes is associated with multiple parents corresponding
//! to different directions, even if each one dimensional node has only a single parent. Thus, over several iterations
//! and if the hierarchical coefficients do not decay monotonically, it is possible to have nodes with missing parents
//! for some directions, which can hinter convergence. Furthermore, the hierarchical coefficients do not carry any
//! information regarding potential local anisotropy and a second refinement criteria can be employed, where an interpolant
//! is constructed using only the data associated with the nodes that lay in a single direction. Children and/or parents of
//! a node are added to the grid only when both the isotropic and anisotropic error indicators exceed the user specified tolerance.
//!
//! \par
//! The \b classic refinement strategy adds only the children for the nodes with hierarchical coefficient that exceed the
//! user provided tolerance (local anisotropy and missing parents are ignored). The \b parents \b first and \b fds
//! strategies loop for missing parents and add the parents before adding children, which usually improves stability
//! when compared to adding only children. The \b direction \b selective and \b fds strategies consider local anisotropy
//! which can reduce the number of required model evaluations, but can also lead to decrease in stability.
//! The \b stable refinement is both isotropic and maintains lower-complete structures, i.e., no point has a missing
//! parent. While the most stable, the \b stable strategy can lead to significant oversampling.
//!
//! Details regarding adaptive hierarchical sparse grids construction and children-parent relations can be found in: \n
//! M. Stoyanov, <a style="font-weight:bold" href="https://link.springer.com/chapter/10.1007/978-3-319-75426-0_8">
//! Adaptive Sparse Grid Construction in a Context of Local Anisotropy and Multiple Hierarchical Parents</a>,\n
//! Sparse Grids and Applications - Miami 2016 pp 175-199.\n
//! Also in: <a style="font-weight:bold" href="https://info.ornl.gov/sites/publications/files/Pub45746.pdf">Tech Report ORNL/TM-2013/384.</a>
enum TypeRefinement{
    //! \brief Isotropic refinement using only the children and disregarding missing parents.
    refine_classic,
    //! \brief Isotropic refinement adding children only if the parents are already included.
    refine_parents_first,
    //! \brief Anisotropic refinement using only the children and disregarding missing parents.
    refine_direction_selective,
    //! \brief Anisotropic refinement adding children only if the parents are already included.
    refine_fds,
    //! \brief Isotropic refinement that ensures the points maintain lower-complete structures.
    refine_stable,
    //! \brief Null method, should \b never be used as input.
    refine_none
};

/*!
 * \ingroup SGEnumerates
 * \brief Modes of acceleration.
 *
 * \par Acceleration Context
 * Each Tasmanian object creates a Tasmanian acceleration context that operate on one of several modes
 * (depending on the external libraries being used) and the context will remain persistent on read/write,
 * make/clean, and move operation. The context cannot be copied from one object to another.
 * For example, an object is created and CUDA acceleration is enabled, then makeLocalPolynomial() is called,
 * then the Nvidia GPU device will be used to speed the different operation of the grid, if then read()
 * is called, a new grid will be loaded from a file and regardless of the type of grid the GPU will still
 * be in use by the object. If the grid is copied, both source and destination will retain their original
 * acceleration contexts and only the grid data (e.g., points and model values) will be copied.
 *
 * \par Utilizing Accelerated Libraries
 * Each Tasmanian operation that has a (potentially) significant computational cost is a candidate for acceleration.
 * Currently, OpenMP is uses throughout the Tasmanian code is essentially all potential places and
 * the evaluateBatch() methods associated with all types of grids can be accelerated with BLAS and CUDA.
 * The BLAS/LAPACK and CUDA acceleration is also available in other select methods, e.g.,
 * TasmanianSparseGrid::loadNeededPoints() for local polynomial and wavelet grid.
 *
 * \par OpenMP and Thread Count
 * If enabled, OpenMP is ubiquitous throughout Tasmanian in almost all computationally expensive operations.
 * The acceleration mode can be disabled only at compile time; however, the standard OpenMP options
 * are respected at runtime, e.g., the method omp_set_num_threads() and OMP_NUM_THREADS environment variable,
 * if either is set to 1, Tasmanian will use only one hardware thread.
 *
 * \par Acceleration Mode None
 * In this mode, Tasmanian will forego the use of third party libraries (other then OpenMP) and the algorithms
 * will run in the most memory conservative way. Due to improved data-locality and depending on the hardware architecture,
 * the TasGrid::accel_none mode can be the fastest mode for small grids with few points and few outputs.
 *
 * \par Memory Management and Persistent Data
 * The use of third party libraries requires that matrices are explicitly formed even when that can be avoided
 * in the no-acceleration mode. Furthermore, operations that are called repeatedly, i.e., evaluateBatch() used
 * in the context of DREAM sampling, will also cache matrices that can be reused in multiple calls.
 * The cache is particularly an issue for the various GPU modes since the memory on the accelerator devices is
 * sometimes limited. One-shot calls, such as loadNeededPoints(), will not use cache that persistent after
 * the call to the method.
 * The cache can always be cleared by either changing the grid (creating a new grid or loading more points
 * as part of iterative refinement), or by switching to mode TasGrid::accel_none and then back to one of the fast modes.
 *
 * \par Memory usage for Evaluate Batch
 * Global, Sequence, Fourier and Wavelet grids operate only with dense matrices and require one persistent
 * matrix of size \b getNumPoints() times \b getNumOutputs(), one temporary matrix of size \b getNumPoints()
 * times \b num_x, and user provided matrix to hold the result of size \b getNumOutputs() times \b num_x.
 * The batch size \b num_x can be adjusted to accommodate memory constraints, but better performance
 * is usually achieved when \b num_x is large due to the increased opportunities to parallelize.
 * Local polynomial grids (when using the default sparse mode) will use sparse representation of
 * the transient matrix and thus the memory usage strongly depends on the graph structure of the grid.
 * In addition, all grids use a relatively small amount of GPU cache which depends on the number of underlying
 * tensors and the specific graph structure of the grid.
 *
 * \par Defaults
 * If \b Tasmanian_ENABLE_BLAS is enabled in CMake, then TasGrid::accel_cpu_blas will be the default acceleration mode,
 * otherwise the default mode is TasGrid::accel_none.
 * The TasGrid::accel_gpu_default uses the first available in the list:
 * - TasGrid::accel_gpu_magma
 * - TasGrid::accel_gpu_cuda
 * - TasGrid::accel_cpu_blas
 * - TasGrid::accel_none
 *
 * \par
 * The defaults and fallback extend to all methods that could use accelerations. For example, if GPU acceleration
 * has been enabled, but the current method does not have a GPU implementation, the "next-best" BLAS implementation
 * will be used instead. Thus, the use of BLAS and LAPACK can be disabled either at compile time or with accel_none,
 * but not with accel_gpu_cuda.
 *
 * \par
 * Note that regardless of the acceleration mode, all evaluate methods will to generate output that
 * is identical up to rounding error. By design, the \b TasmanianSparseGrid::enableAcceleration() can be called
 * with any acceleration type regardless whether it has been enabled by CMake.
 * In order to facilitate code portability, Tasmanian implements sane fall-back
 * modes so that "the next best mode" will be automatically selected if the desired mode is not present. Specifically,
 * missing MAGMA will fallback to CUDA, missing CUDA will fallback to BLAS, and missing BLAS will fallback to none.
 *
 * \par Thread Safety and Const-correctness
 * Using mode TasGrid::accel_none, Tasmanian will not use any transient data-structures and the code is const-correct
 * in the strictest definition, i.e., any two const-methods can be called concurrently from any two threads
 * and will produce the correct result.
 * Other modes use transient cache (with the C++ keyword mutable), which will be generated after the first call
 * to the method and will be reused in follow on call. Thus, the first call is not thread-safe, while the follow on
 * calls are strictly const-correct in the same sense as TasGrid::accel_none.
 * Also note that some third party libraries can have additional restrictions, e.g., concurrent calls to OpenBLAS
 * are not thread-safe unless the library is build with a specific flag. The CUDA and MAGMA libraries appear
 * to not have such limitations.
 *
 * \par Library Handles
 * The GPU libraries use context handles to manage metadata such as active device and device pointer type.
 * By default, Tasmanian will automatically create the handles whenever needed and will destroy them whenever
 * the object is destroyed or the acceleration mode is updated.
 * However, Tasmanian also provides an API where the user can provide the corresponding handles:
 * \code
 *  // Nvidia CUDA framework
 *  grid.setCuBlasHandle(cublas_handle);
 *  grid.setCuSparseHandle(cusparse_handle);
 *  grid.setCuSolverHandle(cusolverdn_handle);
 *  // AMD ROCm framework
 *  grid.setRocBlasHandle(rocblas_handle);
 *  grid.setRocSparseHandle(rocsparse_handle);
 *  // Intel DPC++ framework
 *  grid.setSycleQueue(pointer_to_sycl_queue);
 * \endcode
 *  - Each handle must be a valid, i.e., already created by the corresponding API call and must be associated
 *    with the currently selected GPU device ID.
 *  - Tasmanian assumes that the pointer mode is "host pointer" (for Nvidia and AMD) and if a different mode is set in-between Tasmanian
 *    API calls, the mode must be reset before the next Tasmanian call.
 *  - The SYCL queue is accepted as a pointer, which is different then the usual SYCL approach but it is consistent within the Tasmanian API.
 *  - The methods interact directly with the TPL API and hence the methods do not have fallback modes
 *    if GPU has not been enabled.
 */
enum TypeAcceleration{
    //! \brief Usually the slowest mode, uses only OpenMP multi-threading, but optimized for memory and could be the fastest mode for small problems.
    accel_none,
    //! \brief Default (if available), uses both BLAS and LAPACK libraries.
    accel_cpu_blas,
    //! \brief Equivalent to the first available from MAGMA, CUDA, BLAS, or none.
    accel_gpu_default,
    //! \brief Mixed usage of the CPU (OpenMP) and GPU libraries.
    accel_gpu_cublas,
    //! \brief Similar to the cuBLAS option but also uses a set of Tasmanian custom GPU kernels.
    accel_gpu_cuda,
    //! \brief Same the CUDA option but uses the UTK MAGMA library for the linear algebra operations.
    accel_gpu_magma
};

/*!
 * \ingroup SGEnumerates
 * \brief At the front API, the HIP and CUDA options are equivalent, see TasGrid::TypeAcceleration.
 */
constexpr TypeAcceleration accel_gpu_hip = accel_gpu_cuda;
/*!
 * \ingroup SGEnumerates
 * \brief At the front API, the HIP and CUDA options are equivalent, see TasGrid::TypeAcceleration.
 */
constexpr TypeAcceleration accel_gpu_rocblas = accel_gpu_cublas;

}

#endif
