/*
 * Copyright (c) 2021, Miroslav Stoyanov & William Kong
 *
 * This file is part of
 * Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following
 * conditions are met:
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
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND
 * IMPLIED. THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE
 * OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL
 * ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE. THE USER ASSUMES
 * RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING
 * FROM OR ARISING OUT OF, IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
 */

#ifndef __TASMANIAN_ADDONS_EXOTICQUADRATURE_HPP
#define __TASMANIAN_ADDONS_EXOTICQUADRATURE_HPP

/*!
 * \internal
 * \file tsgExoticQuadrature.hpp
 * \brief Header to include the exotic quadrature templates.
 * \author Miroslav Stoyanov
 * \ingroup TasmanianAddonsExoticQuad
 *
 * \endinternal
 */

#include "tsgAddonsCommon.hpp"

/*!
 * \defgroup TasmanianAddonsExoticQuad Exotic Quadrature
 * \ingroup TasmanianAddons
 *
 * \par Exotic Quadrature
 * The Exotic Quadrature module of Tasmanian offers a collection of functions for generating one dimensional quadrature rules
 * with weight functions that are not strictly non-negative. The quadrature rules are loaded inside TasGrid::CustomTabulated
 * objects and be used in calls to TasmanianSparseGrid::makeGlobalGrid() to extend those to a multidimensional context.
 *
 * \par
 * The integral of interest has the form \f$ \int_{-1}^{1} f(x) \rho(x) dx \f$ and it differs from the classical Gaussian
 * rules by allowing the weight function \f$ \rho(x) \f$ to be negative in portions of the domain.
 * We still require that the weight is integrable and bounded from below, i.e., \f$ \int_{-1}^{1} | \rho(x) | dx < \infty \f$
 * and \f$ \rho(x) \geq c > -\infty \f$.
 *
 * \par
 * The quadrature consists of two components, one is a quadrature that uses classical Gaussian construction using
 * the root of polynomials that are orthogonal with respect to a shifted positive weight \f$ \rho(x) + c \geq 0 \f$
 * and a correction term using Gauss-Legendre rule scaled by -c.
 * While the asymptotic convergence rate is inferior to that of Gauss-Legendre, for small number of points
 * the accuracy of the combined quadrature rule can be significantly greater.
 * This advantage is further magnified in a sparse grid context where reaching the asymptotic regime for Gauss-Legendre
 * may require a prohibitive number of points due to the curse of dimensionality.
 *
 * \par
 * The weight can be defined either with a C++ lambda expression or a one-dimensional TasmanianSparseGrid object.
 * The lambda expression will be evaluated at a large number of Gauss-Legendre points that will be used for
 * a reference quadrature to build the orthogonal polynomial basis.
 * In the case of a sparse grid object, only the loaded points and quadrature weights will be used,
 * and no interpolation will be performed inbetween (i.e., we circumvent potential interpolation error).
 */

namespace TasGrid {

/*!
 * \internal
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Specifies which algorithm is used in computing the Gauss quadrature points and weights in getGaussNodesAndWeights().
 *
 * \endinternal
 */
static int gauss_quadrature_version = 1;

/*!
 * \internal
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Evaluates a polynomial with roots given by \b roots at the point \b x.
 *
 * The polynomial is defined by a simple product of (x - roots[i]) and will not be normalized.
 * \endinternal
 */
inline double poly_eval(const std::vector<double> &roots, double x) {
    double eval_value = 1.0;
    for (auto r : roots) eval_value *= (x - r);
    return eval_value;
}

/*!
 * \internal
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Evaluates a Lagrange basis polynomial at a point \b x.
 *
 * The Lagrange basis at idx is 1 when x = roots[idx] and 0 for all other roots.
 * \param idx is the index of the Lagrange polynomial
 * \param roots is the point set the defines the Lagrange polynomials
 * \param x is the point of interest
 * \endinternal
 */
inline double lagrange_eval(size_t idx, const std::vector<double> &roots, double x) {
    double eval_value = 1.0;
    for (size_t i=0; i<idx; i++) {
        eval_value *= (x - roots[i]) / (roots[idx] - roots[i]);
    }
    for (size_t i=idx+1; i<roots.size(); i++) {
        eval_value *= (x - roots[i]) / (roots[idx] - roots[i]);
    }
    return eval_value;
}

/*!
 * \internal
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Generates n levels of points and weights using orthogonal polynomials with respect to ref_points and ref_weights.
 *
 * Given the reference points and weights, finds the roots of the first n + 1 orthogonal polynomials.
 * The first polynomial is always constant, every following polynomial is chosen so that it is orthogonal
 * with respect to all previous polynomials and the orthogonality (i.e., inner product integral)
 * is computed with the reference quadrature points and weights.
 *
 * \tparam is_symmetric indicates whether the weight function is symmetric or not,
 *         the algorithm makes slight modifications  that improve stability in the symmetric case
 *
 * \param n is the number of quadrature rules to construct
 * \param ref_points are the points of the reference quadrature
 * \param ref_weights are the weights of the reference quadrature multiplied by the shifted weight function
 * \param points_cache on exit will be loaded so that points_cache[i] will be roots of the i+1-st orthogonal polynomial
 * \param weights_cache on exit will be loaded with the quadrature weights corresponding to the points_cache
 * \endinternal
 */
template <bool is_symmetric>
inline void getGaussNodesAndWeights(const int n, const std::vector<double> &ref_points, const std::vector<double> &ref_weights,
                                    std::vector<std::vector<double>> &points_cache, std::vector<std::vector<double>> &weights_cache) {
    #ifndef Tasmanian_ENABLE_BLAS
    // If BLAS is not enabled, we must use an implementation that does not require it.
    if (gauss_quadrature_version == 1) {
        gauss_quadrature_version = 2;
    }
    #endif
    // Compute the roots incrementally.
    assert(ref_points.size() == ref_weights.size());
    assert(ref_points.size() > 0);
    double mu0 = 0.0;
    for (auto &w : ref_weights) mu0 += w;
    std::vector<double> poly_m1_vals(ref_points.size(), 0.0), poly_vals(ref_points.size(), 1.0);
    std::vector<double> diag, offdiag;
    points_cache.resize(n);
    weights_cache.resize(n);
    for (int i=0; i<n; i++) {
        // Form the diagonal and offdiagonal vectors of the Jacobi matrix.
        if (is_symmetric) {
            diag.push_back(0.0);
        } else {
            double diag_numr = 0.0;
            double diag_denm = 0.0;
            for (size_t j=0; j<ref_points.size(); j++) {
            diag_numr += ref_points[j] * (poly_vals[j] * poly_vals[j]) * ref_weights[j];
            diag_denm += (poly_vals[j] * poly_vals[j]) * ref_weights[j];
            }
            diag.push_back(diag_numr / diag_denm);
        }
        if (i >= 1) {
            double sqr_offdiag_numr = 0.0;
            double sqr_offdiag_denm = 0.0;
            for (size_t j=0; j<ref_points.size(); j++) {
                sqr_offdiag_numr += (poly_vals[j] * poly_vals[j]) * ref_weights[j];
                sqr_offdiag_denm += (poly_m1_vals[j] * poly_m1_vals[j])  * ref_weights[j];
            }
            offdiag.push_back(std::sqrt(sqr_offdiag_numr / sqr_offdiag_denm));
        }
        // Compute the Gauss points and weights.
        if (gauss_quadrature_version == 1) {
            // Use LAPACK to obtain the Gauss points, and use the reference points and weights to obtain the Gauss weights.
            // Note that the matrix is always symmetric even if the weights function is not.
            points_cache[i] = TasmanianTridiagonalSolver::getSymmetricEigenvalues(i+1, diag, offdiag);
            if (points_cache[i].size() % 2 == 1 && is_symmetric) {
                points_cache[i][(points_cache[i].size() - 1) / 2] = 0.0;
            }
            weights_cache[i] = std::vector<double>(points_cache[i].size(), 0.0);
            for (size_t j=0; j<points_cache[i].size(); j++) {
                // Integrate the function l_j(x) according to the measure induced by the function [weight_fn(x) + shift].
                for (size_t k=0; k<ref_points.size(); k++) {
                    weights_cache[i][j] += lagrange_eval(j, points_cache[i], ref_points[k]) * ref_weights[k];
                }
            }
        } else if (gauss_quadrature_version == 2) {
            // Use TasmanianTridiagonalSolver::decompose().
            std::vector<double> dummy_diag = diag;
            std::vector<double> dummy_offdiag = offdiag;
            TasmanianTridiagonalSolver::decompose(dummy_diag, dummy_offdiag, mu0, points_cache[i], weights_cache[i]);
        } else {
            throw std::invalid_argument("ERROR: gauss_quadrature_version must be a valid number!");
        }
        // Update the values of the polynomials at ref_points.
        std::swap(poly_m1_vals, poly_vals);
        std::transform(ref_points.begin(), ref_points.end(), poly_vals.begin(),
                       [&points_cache, i](double x){return poly_eval(points_cache[i], x);});
    }
}

/*!
 * \internal
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Generate the exotic quadrature points and weights and load them into a TasGrid::CustomTabulated object.
 *
 * The method calls getGaussNodesAndWeights() and adds the Gauss-Legendre correction.
 *
 * \param n is the number of levels to compute
 * \param shift is the correction term applied to the weight function to enforce positivity
 * \param shifted_weights are the reference quadrature points multiplied by the values of the shifted weight function at the reference points
 * \param ref_points are the reference quadrature points
 * \param description is the human readable string to identify the quadrature rule
 * \param is_symmetric indicates whether we can use algorithm modifications that can improve stability
 * \endinternal
 */
inline TasGrid::CustomTabulated getShiftedExoticQuadrature(const int n, const double shift, const std::vector<double> &shifted_weights,
                                                           const std::vector<double> &ref_points, const char* description,
                                                           const bool is_symmetric = false) {
    // Create the set of points (roots) and weights for the first term.
    assert(shifted_weights.size() == ref_points.size());
    std::vector<std::vector<double>> points_cache, weights_cache;
    if (is_symmetric) {
        getGaussNodesAndWeights<true>(n, ref_points, shifted_weights, points_cache, weights_cache);
    } else {
        getGaussNodesAndWeights<false>(n, ref_points, shifted_weights, points_cache, weights_cache);
    }

    // Create and append the set of correction points and weights for the second term only if the shift is nonzero.
    if (shift != 0.0) {
        for (int i=0; i<n; i++) {
            const size_t init_size = points_cache[i].size();
            std::vector<double> correction_points, correction_weights;
            TasGrid::OneDimensionalNodes::getGaussLegendre(static_cast<int>(init_size), correction_weights, correction_points);
            for (auto &w : correction_weights) w *= -shift;
            if (correction_points.size() % 2 == 1 && is_symmetric) {
                // Zero out for stability.
                correction_points[(correction_points.size() - 1) / 2] = 0.0;
            }
            // Account for duplicates up to a certain tolerance.
            for (size_t j=0; j<correction_points.size(); j++) {
                long long nonunique_idx = -1;
                for (size_t k=0; k<init_size; k++) {
                    if (std::abs(correction_points[j] - points_cache[i][k]) <= Maths::num_tol) {
                        nonunique_idx = static_cast<long long>(k);
                        break;
                    }
                }
                if (nonunique_idx == -1) {
                    points_cache[i].push_back(correction_points[j]);
                    weights_cache[i].push_back(correction_weights[j]);
                } else {
                    weights_cache[i][nonunique_idx] += correction_weights[j];
                }
            }
        }
    }

    // Output to a CustomTabulated instance.
    std::vector<int> num_nodes(n), precision(n);
    for (int i=0; i<n; i++) {
        num_nodes[i] = static_cast<int>(points_cache[i].size());
        precision[i] = 2 * (i + 1) - 1;
    }
    return TasGrid::CustomTabulated(std::move(num_nodes), std::move(precision), std::move(points_cache), std::move(weights_cache),
                                    description);
}

/*!
 * \internal
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Multiplies the reference quadrature weights by the shifted values of the weight function.
 *
 * \param vals are the values of the exotic weight function at the reference quadrature points
 * \param shift is a real-valued scalar that ensures all \b vals[i] + \b shift are positive
 * \param ref_weights are the reference quadrature weights, will be overwritten with the result ref_weights[i] * (vals[i] + shift)
 *
 * \throws std::invalid_argument if \b vals[i] + \b shift is negative for some index i
 *
 * \endinternal
 */
inline void shiftReferenceWeights(std::vector<double> const &vals, double shift, std::vector<double> &ref_weights){
    assert(ref_weights.size() == vals.size());
    if (not std::all_of(vals.begin(), vals.end(), [shift](double x){return (x+shift)>=0.0;}))
        throw std::invalid_argument("ERROR: shift needs to be chosen so that weight_fn(x)+shift is nonnegative!");
    for (size_t k=0; k<ref_weights.size(); k++)
        ref_weights[k] *= (vals[k] + shift);
}

/*!
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Constructs an exotic quadrature from a sparse grid object.
 *
 * Constructs a set of one dimensional quadrature rules with respect to a 1D weight function approximated by \b grid.
 *
 * \param num_levels is the number of levels that should be computed
 * \param shift is a real-valued scalar that needs to be added to the weight function to make it positive
 * \param grid has dimension 1 and at least one loaded point, should use as many points as possible for better accuracy
 *        (up to some point where numerical stability is lost)
 * \param description is a human readable string that can be used to identify the quadrature rule (could be empty)
 * \param is_symmetric indicates whether to assume that the weight is symmetric, which allows for some stability improvements;
 *        symmetric means "even" on [-1, 1], which leads to all odd power polynomials integrating to zero;
 *        the same holds for shifted domains so long as the weight is "even" with respect to the mid-point of the domain.
 *
 * \returns TasGrid::CustomTabulated object holding the points and weights for the different levels of the quadrature
 *
 * \throws std::invalid_argument if \b grid doesn't have a single input and output or no loaded points,
 *         or if the value of the weight function plus the shift is not non-negative
 *
 * The \b grid object is used in two ways: first, it defines a reference quadrature that will be used to compute
 * the inner-product integrals that define orthogonal polynomials; second, it defines the values of
 * the weight function \f$ \rho(x) \f$ at the quadrature points.
 * Note that the values of the weight function will not be interpolated between the reference points.
 *
 * Examples of symmetric weight functions are \f$ \sin(x) \f$, \f$ \sin(x) / x \f$, constant weight, and
 * the Gauss-Chebyshev weights. On the other hand, \f$ \cos(x) \f$ is not symmetric in the context of
 * this algorithm due to the shift that must be applied.
 */
inline TasGrid::CustomTabulated getExoticQuadrature(const int num_levels, const double shift, const TasGrid::TasmanianSparseGrid &grid,
                                                    const char* description, const bool is_symmetric = false) {
    if (grid.getNumLoaded() == 0) {
        throw std::invalid_argument("ERROR: grid needs to contain a nonzero number of loaded points returned by grid.getLoadedPoints()!");
    }
    if (grid.getNumDimensions() != 1) {
        throw std::invalid_argument("ERROR: grid needs to be one dimensional!");
    }
    if (grid.getNumOutputs() != 1) {
        throw std::invalid_argument("ERROR: grid needs to have scalar outputs!");
    }
    std::vector<double> ref_points = grid.getLoadedPoints();
    std::vector<double> shifted_weights = grid.getQuadratureWeights();
    std::vector<double> weight_fn_vals = std::vector<double>(grid.getLoadedValues(), grid.getLoadedValues() + grid.getNumLoaded());
    shiftReferenceWeights(weight_fn_vals, shift, shifted_weights);
    return getShiftedExoticQuadrature(num_levels, shift, shifted_weights, ref_points, description, is_symmetric);
}

/*!
 * \ingroup TasmanianAddonsExoticQuad
 * \brief Similar to getExoticQuadrature() but the weight function is defined by a lambda expression.
 *
 * Constructs a CustomTabulated object holding the points and weights for the different levels of the quadrature
 * for the given parameter \b weight_fn. The \b num_ref_points parameter sets the number of reference points
 * from a Gauss-Legendre quadrature that will be used to compute the inner-product integrals
 * that define the orthogonal polynomials.
 */
inline TasGrid::CustomTabulated getExoticQuadrature(const int num_levels, const double shift, std::function<double(double)> weight_fn,
                                                    const int num_ref_points, const char* description, const bool is_symmetric = false) {
    std::vector<double> shifted_weights, ref_points, weight_fn_vals(num_ref_points);
    TasGrid::OneDimensionalNodes::getGaussLegendre(num_ref_points, shifted_weights, ref_points);
    std::transform(ref_points.begin(), ref_points.end(), weight_fn_vals.begin(), weight_fn);
    shiftReferenceWeights(weight_fn_vals, shift, shifted_weights);
    return getShiftedExoticQuadrature(num_levels, shift, shifted_weights, ref_points, description, is_symmetric);
}

inline TasGrid::CustomTabulated getNestedExoticQuadrature(
    const int num_levels, TypeOneDRule rule, const double shift, std::function<double(double)> weight_fn,
    const int num_ref_points, const char* description)
{
    if (OneDimensionalMeta::isNonNested(rule))
        throw std::runtime_error("cannot call getNestedExoticQuadrature() with a non-nested rule");
    std::vector<double> shifted_weights, ref_points, weight_fn_vals(num_ref_points);
    TasGrid::OneDimensionalNodes::getGaussLegendre(num_ref_points, shifted_weights, ref_points);
    std::transform(ref_points.begin(), ref_points.end(), weight_fn_vals.begin(), weight_fn);
    shiftReferenceWeights(weight_fn_vals, shift, shifted_weights);

    std::vector<int> num_nodes(num_levels);
    std::vector<int> precision(num_levels);
    std::vector<std::vector<double>> points_cache(num_levels);
    std::vector<std::vector<double>> weights_cache(num_levels);

    for (int l = 0; l < num_levels; l++) {
        // std::cout << " building level " << l << '\n';
        auto grid = makeGlobalGrid(1, 1, l, TypeDepth::type_level, rule);

        num_nodes[l] = grid.getNumPoints();
        precision[l] = num_nodes[l] - 1;

        // std::cout << " num-nodes and prec = " << num_nodes[l] << "   " << precision[l] << "\n";

        points_cache[l] = grid.getNeededPoints();
        weights_cache[l].resize(num_nodes[l]);

        std::vector<double> vals(num_nodes[l], 0.0);
        std::vector<double> neg_weights = grid.getQuadratureWeights();

        for (int i = 0; i < num_nodes[l]; i++) {
            // std::cout << " building points = " << i << '\n';
            vals[i] = 1.0;
            grid.loadNeededValues(vals);
            vals[i] = 0.0;

            std::vector<double> leg;
            grid.evaluateBatch(ref_points, leg);

            double sum = 0.0;
            for (size_t j = 0; j < ref_points.size(); j++) {
                sum += leg[j] * shifted_weights[j];
            }
            weights_cache[l][i] = sum - shift * neg_weights[i];
        }
        // std::cout << " ---------- done level = " << l << "\n";
    }

    // std::cout << " building rule \n";

    return TasGrid::CustomTabulated(std::move(num_nodes), std::move(precision), std::move(points_cache), std::move(weights_cache),
                                    description);

    //return getShiftedExoticQuadrature(num_levels, shift, shifted_weights, ref_points, description, is_symmetric);
}

} // namespace(TasGrid)
#endif
