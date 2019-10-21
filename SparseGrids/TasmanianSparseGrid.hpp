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

/*!
 * \internal
 * \file TasmanianSparseGrid.hpp
 * \brief Main header for the Sparse Grid module.
 * \author Miroslav Stoyanov
 * \ingroup TasmanianSG
 *
 * The header needed to access the Tasmanian sparse grids module,
 * the TasGrid::TasmanianSparseGrids class is defined here.
 * \endinternal
 */

#ifndef __TASMANIAN_SPARSE_GRID_HPP
#define __TASMANIAN_SPARSE_GRID_HPP

#include "tsgGridGlobal.hpp"
#include "tsgGridLocalPolynomial.hpp"
#include "tsgGridWavelet.hpp"
#include "tsgGridFourier.hpp"

/*!
 * \defgroup TasmanianSG Sparse Grids
 *
 * \par Sparse Grids
 * A family of algorithms for multidimensional integration and interpolation
 * where the operator is constructed as a linear combination of tensors
 * with different number of points in each direction.
 * The combination of tensors is chosen to include a specific function space
 * that is optimal for approximation of the target model.
 * For more details of the sparse grid construction, refer to the
 * .pdf manual available at [http://tasmanian.ornl.gov/](http://tasmanian.ornl.gov/)
 */

/*!
 * \ingroup TasmanianSG
 * \brief Encapsulates the Tasmanian Sparse Grid module.
 */
namespace TasGrid{

/*!
 * \ingroup TasmanianSG
 * \addtogroup TasmanianSGClass Sparse Grid Class
 *
 * The sparse grid capabilities of Tasmanian are encapsulated in a single master class
 * TasGrid::TasmanianSparseGrid.
 */

/*!
 * \ingroup TasmanianSGClass
 * \brief The master-class that represents an instance of a Tasmanian sparse grid.
 *
 * \par class TasmanianSparseGrid
 * An object of this class represents a sparse grid, all aspects of the grid
 * can be accessed through member methods.
 * Each object operates independently to the extend allowed by third-party
 * libraries, e.g., making concurrent calls to OpenBLAS will result in a race
 * condition unless OpenBLAS is compiled with a special flag.
 * The available member methods are split into groups that deal with
 * different aspects of the grid.
 * The object itself wraps around one of five different types of grids,
 * Global, Sequence, Local Polynomial, Wavelet, and Fourier,
 * each using different back-end data structures.
 * The class itself handles the domain transformations,
 * e.g., the translation between the canonical [-1, 1] and a user provided [a, b],
 * as well as mechanisms to select acceleration back-end, common meta I/O,
 * and error checking.
 *
 * \par Error Handling
 * \b TasmanianSparseGrid throws \b std::invalid_argument and \b std::runtime_error
 * exceptions with simple human-readable messages.
 * The \b std::invalid_argument exception indicates method call arguments of
 * insufficient or incompatible data, e.g., vectors have insufficient size
 * or negative number was given in place of a positive one.
 * The \b std::runtime_error indicates problems with missing files, incorrect
 * file format, or out-of-order function calls, e.g., adaptive refinement
 * strategies require model values in order to infer the optimal structure of
 * the grid, therefore calling a refinement method before model values are
 * provided will result in a \b std::runtime_error.
 * Similarly, \b std::runtime_error will be thrown if a method is called
 * with the wrong grid type, e.g., global anisotropic coefficients cannot be
 * computed for local polynomial grids.
 *
 * If CUDA acceleration is used, CUDA error may be encountered, e.g.,
 * incorrect driver initialization, out-of-memory, etc.
 * Tasmanian will translate CUDA error codes and messages into
 * \b std::runtime_error exceptions and such exceptions can be raised
 * by any method that uses acceleration.
 *
 * \par Constructors and Copy/Move assignment
 * TasGrid::TasmanianSparseGrid includes move and copy constructors
 * and assignment operator= overloads.
 * There is also a default constructor that creates an empty grid.
 * Note that move will preserve the enabled acceleration mode and the internal
 * cache data-structures, while copy will reset the acceleration to the default.
 * - TasmanianSparseGrid()
 * - copyGrid()
 *
 * \par Make Grid Methods
 * Each of the five grid types comes with a separate make command.
 * The first three parameters are always the number of inputs (dimensions),
 * the number of outputs, and the initial depth or level of the grid.
 * The remaining parameters are specific to the type of grid.
 * - makeGlobalGrid()
 * - makeSequenceGrid()
 * - makeLocalPolynomialGrid()
 * - makeWaveletGrid()
 * - makeFourierGrid()
 * - see also the factory methods, e.g., TasGrid::makeGlobalGrid()
 *
 * \par Domain Transformations
 * In most cases, sparse grids are constructed on a canonical domain [-1, 1].
 * The exceptions are the Fourier grids with domain [0, 1], and the
 * Gauss-Hermite and Gauss-Laguerre rules with unbounded domain, see
 * TasGrid::rule_gausshermite and TasGrid::rule_gausslaguerre for details.
 * Tasmanian can automatically apply a linear domain transformations shifting
 * the canonical interval to an arbitrary [a, b] or a non-linear (conformal)
 * transformation that may accelerate convergence for some models.
 * Each dimension of the grid can have a different linear and/or non-linear
 * transform.
 * - setDomainTransform(), getDomainTransform()
 * - isSetDomainTransfrom(), clearDomainTransform()
 * - setConformalTransformASIN(), getConformalTransformASIN()
 * - isSetConformalTransformASIN(), clearConformalTransform()
 *
 * \par Stream and File I/O
 * Sparse grid objects can be written to and from files and \b std::ostream objects.
 * Tasmanian supports both binary and ASCII formats; the binary format
 * has lower overhead in both file size and operations required to read/write,
 * the ASCII format is portable without considerations of endians and
 * for small grids the files are human-readable which helps debugging.
 * - read(), TasGrid::readGrid()
 * - write()
 *
 * \par Get Points and Weights
 * The points on the grid are labeled as "loaded" or "needed" based on
 * whether the associated model values are already provided by the user.
 * The grid points are also associated with quadrature and interpolation
 * weights, where the computed weights correspond to the loaded points
 * unless all points are labeled as needed. If the grid is set to work
 * with zero model outputs, then all points are labeled as "loaded".
 * - getNumLoaded(), getNumNeeded(), getNumPoints()
 * - getLoadedPoints(), getNeededPoints(), getPoints()
 * - getLoadedValues()
 * - getQuadratureWeights()
 * - getInterpolationWeights()
 *
 * \par Load Model Values or Hierarchical Coefficients
 * In order to construct an interpolant, Tasmanian needed to compute the
 * coefficients of the basis functions. The user can directly provide the
 * coefficients, e.g., computed externally with a regression method using
 * an unstructured set of samples, or the user can provide the model values
 * at the needed points and let Tasmanian do the computations.
 * - TasGrid::loadNeededPoints()
 * - loadNeededPoints(), getLoadedValues()
 * - setHierarchicalCoefficients()
 *
 * \par Update and Adaptive Refinement
 * The most efficient approximation schemes adapt the grid (and the associated
 * basis functions) to the target model. Tasmanian provides several adaptive
 * procedures tailored to different types of grids.
 * The refinement requires two-way communication between the model and
 * Tasmanian, i.e., Tasmanian provides an initial set of points, the model
 * provides the values, then Tasmanian provides an updated set of points, and so on.
 * This can be done either in batches of point or in dynamic construction setup
 * where values can be given one at a time in an arbitrary order (see next paragraph).
 * See also the papers referenced in TasGrid::TypeDepth and TasGrid::TypeRefinement.
 * - updateGlobalGrid(), updateSequenceGrid(), updateFourierGrid()
 * - setAnisotropicRefinement(), setSurplusRefinement()
 * - clearRefinement()
 * - getLevelLimits(), clearLevelLimits()
 * - removePointsByHierarchicalCoefficient()
 *
 * \par Dynamic Construction
 * The standard refinement procedure works in batches, where all model values
 * for the batch have to be computed externally and loaded together in the
 * correct order. The dynamic construction alleviates those restrictions,
 * but comes at an increase cost since Tasmanian has to do more work and
 * store extra data. The dynamic construction (and the internal data-structures)
 * are initiated with the beginConstruction() command and batch refinement
 * cannot be used until the procedure is concluded. See also TasGrid::constructSurrogate()
 * that can take a user defined model and fully automate the construction process.
 * - TasGrid::constructSurrogate()
 * - beginConstruction(), finishConstruction()
 * - getCandidateConstructionPoints()
 * - loadConstructedPoints()
 *
 * \par Using Unstructured Data
 * The standard sparse grid methods assume that model data is available at the very
 * specific grid points that are chosen according to optimal estimates.
 * However, in some cases, the model inputs cannot be controlled precisely
 * and only randomly samples model data is available.
 * Sparse grids can still produce accurate surrogates using such unstructured
 * data by effectively removing the points and working only with the underlying
 * basis (the basis is still optimal for the corresponding class of functions).
 * The hierarchical basis method allow direct access to the values of
 * the basis functions and the associated coefficients.
 * - evaluateHierarchicalFunctions(), evaluateSparseHierarchicalFunctions()
 * - evaluateHierarchicalFunctionsGPU(), evaluateSparseHierarchicalFunctionsGPU()
 * - setHierarchicalCoefficients(), getHierarchicalCoefficients()
 * - integrateHierarchicalFunctions()
 *
 * \par Evaluate Methods
 * The evaluate methods compute the sparse grid approximation to the model for
 * arbitrary point within the domain, i.e., for points that don't necessarily
 * align with the original grid.
 * The batch and fast evaluate method can leverage accelerated linear algebra
 * libraries such as BLAS, cuBLAS and MAGAMA.
 * See the documentation for TasGrid::TypeAcceleration for details.
 * - evaluate()
 * - evaluateBatch(), evaluateBatchGPU(), evaluateFast()
 * - integrate()
 *
 * \par Acceleration Back-end Selection
 * Allows specifying the acceleration used for evaluation methods and (in some
 * cases) for computing the basis coefficients during loadNeededPoints().
 * For example, methods are provided to check the number of available CUDA
 * devices and to select an active device on a multi-GPU system.
 * - enableAcceleration(), getAccelerationType(), favorSparseAcceleration()
 * - setGPUID(), getGPUID(), getNumGPUs()
 * - isAccelerationAvailable(), getGPUName(), getGPUMemory()
 *
 * \par Get Grid Meta-data
 * Various method that read the number of points, grid type, specifics about
 * the rule and domain transformations and others.
 * Some of the output is in human-readable format for debugging and sanity check.
 * - getNumDimensions(), getNumOutputs(), getRule()
 * - getCustomRuleDescription()
 * - getAlpha(), getBeta(), getOrder()
 * - isGlobal(), isSequence(), isLocalPolynomial(), isFourier(), isWavelet()
 * - isEmpty(), empty()
 * - printStats()
 *
 * \par Get Library Meta-data
 * Runtime API is provided for the library version and various compile
 * time options.
 * Those methods are \b static and are included in the class API solely for
 * consistency reasons.
 * - getVersion(), getVersionMajor(), getVersionMinor(), getLicense()
 * - getGitCommitHash(), getCmakeCxxFlags(), isOpenMPEnabled()
 */
class TasmanianSparseGrid{
public:
    //! \brief Default constructor, creates and empty grid.
    TasmanianSparseGrid();
    //! \brief Copy constructor, note that the selected acceleration mode is not copied, acceleration is reset to the default.
    TasmanianSparseGrid(const TasmanianSparseGrid &source);
    //! \brief Move constructor, the selected acceleration mode is also carried over.
    TasmanianSparseGrid(TasmanianSparseGrid &&source) = default;
    //! \brief Destructor, releases all resources.
    ~TasmanianSparseGrid();

    //! \brief Copy assignment, note that the selected acceleration mode is not copied, acceleration is reset to the default.
    TasmanianSparseGrid& operator=(TasmanianSparseGrid const &source);
    //! \brief Move assignment, the selected acceleration mode is also carried over.
    TasmanianSparseGrid& operator=(TasmanianSparseGrid &&source) = default;

    //! \brief Return a hard-coded character string with the version in format "Major.Minor".
    static const char* getVersion(); // human readable
    //! \brief Return the library major version.
    static int getVersionMajor();
    //! \brief Return the library minor version.
    static int getVersionMinor();
    //! \brief Return a hard-coded character string with a brief statement of the license.
    static const char* getLicense(); // human readable
    //! \brief Return the git hash string, will use a placeholder if the git command was not found on compile time or if building from an official release.
    static const char* getGitCommitHash();
    //! \brief Return the CMAKE_BUILD_TYPE and CMAKE_CXX_FLAGS used in the configuration.
    static const char* getCmakeCxxFlags();
    //! \brief Returns \b true if compiled with OpenMP support, e.g., Tasmanian_ENABLE_OPENMP=ON.
    static bool isOpenMPEnabled();

    //! \brief Write the grid to the given \b filename using either \b binary or ASCII format.
    void write(const char *filename, bool binary = mode_binary) const;
    //! \brief Read the grid from the given \b filename, automatically detect the format.
    void read(const char *filename); // auto-check if format is binary or ascii

    //! \brief Write the grid to the given stream \b ofs using either \b binary or ASCII format.
    void write(std::ostream &ofs, bool binary = mode_binary) const;
    //! \brief Read the grid from the given stream \b ifs using either \b binary or ASCII format.
    void read(std::istream &ifs, bool binary = mode_binary);

    /*!
     * \brief Make a Global Grid using Lagrange polynomials with support over the entire domain.
     *
     * Construct a sparse grid with the given set of parameters.
     * \param dimensions is a positive integer specifying the dimension or number of model inputs of the grid.
     * There is no hard restriction on how big the dimension can be; however, for large dimensions,
     * the number of points of the sparse grid grows fast and hence the grid may require a prohibitive amount of memory.
     * \param outputs is a non-negative integer specifying the number of outputs for the model.
     * If outputs is zero, then the grid can only generate quadrature and interpolation weights.
     * There is no hard restriction on how many outputs can be handled; however, Tasmanian requires at least
     * outputs times number of points in storage and the computational complexity of evaluate and I/O methods
     * increases linearly with the number of outputs.
     * \param depth is a non-negative integer that controls the density of grid points.
     * This is commonly referred to the "level" of the grid and corresponds to the L parameter
     * in the formulas for TasGrid::TypeDepth. There is no hard restriction on how big depth can be;
     * however, the number of points have a strong influence on performance and memory requirements.
     * \param type indicates the tensor selection strategy, see TasGrid::TypeDepth.
     * \param rule is one of the global rules, see TasGrid::TypeOneDRule.
     * \param anisotropic_weights indicates the relative "importance" of the inputs.
     * If an empty vector is provided, the isotropic selection is used, i.e., assuming all inputs
     * are equally important.
     * If non-empty vector is given, then the size must match the number of required weights
     * based on the selection \b type, i.e., the \b curved types use 2 x \b dimensions number of
     * weights while the non-curved types use only \b dimensions.
     * Integer values are used, but only the relative scale affects the selection,
     * e.g., {2, 1}, {4, 2}, {20, 10} and {200, 100} are equivalent.
     * The first \b dimensions entries correspond to the \f$ \xi_1, \xi_2, \cdots, \xi_d \f$ weights
     * and the second set corrections to the \f$ \eta_1, \eta_2, \cdots, \eta_d \f$ weights
     * in the formulas in TasGrid::TypeDepth.
     * \param alpha specifies the \f$ \alpha \f$ parameter for the integration weight \f$ \rho(x) \f$,
     * ignored when \b rule is doesn't have such parameter. See TasGrid::TypeOneDRule.
     * \param beta specifies the \f$ \beta \f$ parameter for the integration weight \f$ \rho(x) \f$,
     * ignored when \b rule is doesn't have such parameter. See TasGrid::TypeOneDRule.
     * \param custom_filename specifies the file containing the custom rule description,
     * used only when \b rule is TasGrid::rule_customtabulated.
     * \param level_limits is either empty or has size \b dimensions indicating the restriction
     * of the levels for each direction which in turn restricts the number of points.
     * For example, limit 1 when using TasGrid::rule_clenshawcurtis will restricts the grid to the
     * mid-point and end-points of the domain. Each dimension can have a different restriction
     * and negative numbers indicate no restriction for that direction.
     *
     * \throws std::invalid_argument with human readable messages when integers are out of range,
     * the \b rule is not a global rule, or the vectors have incorrect size.
     * \throws std::runtime_error if the \b custom_filename if missing, or it cannot be opened,
     * or the format is incorrect.
     */
    void makeGlobalGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule,
                        std::vector<int> const &anisotropic_weights, double alpha = 0.0, double beta = 0.0,
                        const char* custom_filename = nullptr, std::vector<int> const &level_limits = std::vector<int>());
    /*!
     * \brief Overload using raw-arrays.
     *
     * The inputs are the same as \b makeGlobalGrid() except the vectors are replaced by arrays.
     * Passing \b nullptr for an array is equivalent to passing an empty vector.
     * Throws the same exceptions, but it does not check if the arrays have the correct size.
     */
    void makeGlobalGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule,
                        const int *anisotropic_weights = nullptr, double alpha = 0.0, double beta = 0.0,
                        const char* custom_filename = nullptr, const int *level_limits = nullptr);

    /*!
     * \brief Make Sequence Grid using Newton polynomials with support over the entire domain.
     *
     * Mathematically the Sequence and Global grids do not differ in properties;
     * however, the implementation of the Sequence grids uses optimized internal data structures
     * which leads to massive increase in speed when calling evaluate methods
     * at the expense of doubled memory size and increased cost of \b loadNeededPoints().
     * The inputs are identical to \b makeGlobalGrid() with the restriction that \b rule
     * must be one of:
     * \code
     *   rule_rleja           rule_leja           rule_minlebesgue
     *   rule_rlejashifted    rule_maxlebesgue    rule_mindelta
     * \endcode
     *
     * \throws std::invalid_argument with human readable messages when integers are out of range,
     * the \b rule is not a sequence rule, or the vectors have incorrect size.
     */
    void makeSequenceGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule,
                          std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits = std::vector<int>());
    /*!
     * \brief Overload using raw-arrays.
     *
     * The inputs are the same as \b makeSequenceGrid() except the vectors are replaced by arrays.
     * Passing \b nullptr for an array is equivalent to passing an empty vector.
     * Throws the same exceptions, but it does not check if the arrays have the correct size.
     */
    void makeSequenceGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule,
                          const int *anisotropic_weights = nullptr, const int *level_limits = nullptr);

    /*!
     * \brief Make Local Polynomial Grid using piece-wise polynomials with decreasing (compact) support.
     *
     * Creates a grid based on one of the local hierarchical piece-wise polynomial rules.
     * Local grids can be used for integration, but in many cases the quadrature weights will be zero for some points.
     * \param dimensions is a positive integer specifying the dimension or number of model inputs of the grid.
     * See also makeGlobalGrid().
     * \param outputs is a non-negative integer specifying the number of outputs for the model.
     * Unlike makeGlobalGrid() the storage requirement is at least twice the number of points times the number of outputs.
     * \param depth is a non-negative integer that controls the density of grid points, i.e., the "level".
     * The initial construction of the local grids uses tensor selection equivalent to TasGrid::type_level
     * and depth is the \b L parameter in the formula.
     * \param order is an integer no smaller than -1 indicating the largest polynomial order of the basis functions,
     * i.e., 1 indicates the use of constant and linear functions only, while 2 would allow quadratics
     * (if enough points are present). Using -1 indicates using the largest possible order for each point.
     * See the papers referenced in the TasGrid::TypeRefinement description.
     * \param rule is one of the local polynomial rules, e.g.,
     * \code
     * rule_localp    rule_semilocalp    rule_localp0    rule_localpb
     * \endcode
     * Note that using order \b 0 with any rule will automatically switch to a special piece-wise constant hierarchy.
     * \param level_limits is identical to that for makeGlobalGrid().
     *
     * \throws std::invalid_argument with human readable messages when integers are out of range,
     * the \b rule is not a local polynomial rule, or the vector has incorrect size.
     */
    void makeLocalPolynomialGrid(int dimensions, int outputs, int depth, int order, TypeOneDRule rule, std::vector<int> const &level_limits);
    /*!
     * \brief Overload using raw-arrays.
     *
     * The inputs are the same as \b makeLocalPolynomialGrid() except the vector is replaced by an array.
     * Passing \b nullptr for an array is equivalent to passing an empty vector.
     * Throws the same exceptions, but this does not check if the array has the correct size.
     */
    void makeLocalPolynomialGrid(int dimensions, int outputs, int depth, int order = 1, TypeOneDRule rule = rule_localp, const int *level_limits = nullptr);

    /*!
     * \brief Make a Wavelet grid using local hierarchical wavelet basis.
     *
     * Wavelets are specialized functions that form a Riesz basis and can offer better convergence in an adaptive refinement process.
     * See the pdf manual for more details.
     * \param dimensions is a positive integer specifying the dimension or number of model inputs of the grid.
     * See also makeGlobalGrid().
     * \param outputs is a non-negative integer specifying the number of outputs for the model.
     * Unlike makeGlobalGrid() the storage requirement is at least twice the number of points times the number of outputs.
     * \param depth is a non-negative integer that controls the density of grid points, i.e., the "level".
     * The initial construction of the local grids uses tensor selection equivalent to TasGrid::type_level
     * and depth is the \b L parameter in the formula, but unlike the local polynomial hierarchies
     * the zero-th level has 3 or 5 points and hence the grid density is overall higher.
     * \param order is the wavelet approximation order, implemented orders are only 1 and 3 (wavelet order cannot be an even number).
     * \param level_limits is identical to that for makeGlobalGrid().
     *
     * \throws std::invalid_argument with human readable messages when integers are out of range, or the vector has incorrect size.
     */
    void makeWaveletGrid(int dimensions, int outputs, int depth, int order, std::vector<int> const &level_limits);
    /*!
     * \brief Overload using raw-arrays.
     *
     * The inputs are the same as \b makeWaveletGrid() except the vector is replaced by an array.
     * Passing \b nullptr for an array is equivalent to passing an empty vector.
     * Throws the same exceptions, but this does not check if the array has the correct size.
     */
    void makeWaveletGrid(int dimensions, int outputs, int depth, int order = 1, const int *level_limits = nullptr);

    /*!
     * \brief Make a Fourier grid using trigonometric basis with support over the entire domain.
     *
     * The trigonometric basis guarantees that the interpolant is periodic (in all derivatives) across the domain boundary.
     * This is the only grid that defaults to a canonical interval [0, 1] (as opposed to [-1, 1]).
     * The parameters and error messages are identical to those used in makeGlobalGrid(),
     * but this always uses TasGrid::rule_fourier and the polynomial order in the selection
     * is replaced by the frequency of the trigonometric basis.
     */
    void makeFourierGrid(int dimensions, int outputs, int depth, TypeDepth type,
                         std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits = std::vector<int>());
    /*!
     * \brief Overload using raw-arrays.
     *
     * The inputs are the same as \b makeFourierGrid() except the vectors are replaced by arrays.
     * Passing \b nullptr for an array is equivalent to passing an empty vector.
     * Throws the same exceptions, but it does not check if the arrays have the correct size.
     */
    void makeFourierGrid(int dimensions, int outputs, int depth, TypeDepth type, const int* anisotropic_weights = nullptr, const int* level_limits = nullptr);

    /*!
     * \brief Replace the grid with a copy of the \b source, does not copy the acceleration options.
     *
     * Using the default inputs is equivalent to the assignment operator,
     * otherwise this allows to copy all points and inputs of the grid but
     * only a sub-range of the outputs.
     *
     * \param source the grid to copy from.
     *
     * \param outputs_begin the first output of the new grid.
     * \param outputs_end is the first not-included output, i.e.,
     *        the logic is similar to std::copy().
     *        If the last parameter is outside of the range,
     *        then include all outputs from output_begin till
     *        the end of the outputs.
     */
    void copyGrid(const TasmanianSparseGrid *source, int outputs_begin = 0, int outputs_end = -1);
    /*!
     * \brief Overload using pass-by-reference as opposed to a pointer.
     *
     * The pointer version is supported for backwards compatibility,
     * this is the preferred way to use the copy command.
     */
    void copyGrid(TasmanianSparseGrid const &source, int outputs_begin = 0, int outputs_end = -1){ copyGrid(&source, outputs_begin, outputs_end); }

    /*!
     * \brief Construct a new grid and merge it with the current one.
     *
     * This method is used for refinement with user specified anisotropic weights.
     * As such, it can be called only for global grids with a nested rule.
     * If there are no loaded points (or if the number of outputs is zero) then this is equivalent to
     * calling makeGlobalGrid() (i.e., completely replacing the grid)
     * using the current \b dimensions, \b outputs, and \b rule parameters and the new values for \b depth, \b type,
     * anisotropic coefficients and level limits.
     * If there are loaded model values, then the new grid will be added to the existing one without removing any existing points.
     *
     * The parameters used and the thrown exceptions are identical to those in the call to makeGlobalGrid().
     * In addition, \b std::runtime_error is thrown if the current grid is not Global.
     */
    void updateGlobalGrid(int depth, TypeDepth type, std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits = std::vector<int>());
    /*!
     * \brief Overload using raw-arrays.
     *
     * Array dimensions are not checked, otherwise identical to updateGlobalGrid().
     */
    void updateGlobalGrid(int depth, TypeDepth type, const int *anisotropic_weights = nullptr, const int *level_limits = nullptr);

    /*!
     * \brief Construct a new grid and merge it with the current one.
     *
     * This method is used for refinement with user specified anisotropic weights.
     * If there are no loaded points (or if the number of outputs is zero) then this is equivalent to
     * calling makeSequenceGrid() (i.e., completely replacing the grid)
     * using the current \b dimensions, \b outputs, and \b rule parameters and the new values for \b depth, \b type,
     * anisotropic coefficients and level limits.
     * If there are loaded model values, then the new grid will be added to the existing one without removing any existing points.
     *
     * The parameters used and the thrown exceptions are identical to those in the call to makeSequenceGrid().
     * In addition, \b std::runtime_error is thrown if the current grid is not Sequence.
     */
    void updateSequenceGrid(int depth, TypeDepth type, std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits = std::vector<int>());
    /*!
     * \brief Overload using raw-arrays.
     *
     * Array dimensions are not checked, otherwise identical to updateSequenceGrid().
     */
    void updateSequenceGrid(int depth, TypeDepth type, const int *anisotropic_weights = nullptr, const int *level_limits = nullptr);

    /*!
     * \brief Construct a new grid and merge it with the current one.
     *
     * This method is used for refinement with user specified anisotropic weights.
     * If there are no loaded points (or if the number of outputs is zero) then this is equivalent to
     * calling makeFourierGrid() (i.e., completely replacing the grid)
     * using the current \b dimensions, \b outputs, and \b rule parameters and the new values for \b depth, \b type,
     * anisotropic coefficients and level limits.
     * If there are loaded model values, then the new grid will be added to the existing one without removing any existing points.
     *
     * The parameters used and the thrown exceptions are identical to those in the call to makeFourierGrid().
     * In addition, \b std::runtime_error is thrown if the current grid is not Fourier.
     */
    void updateFourierGrid(int depth, TypeDepth type, std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits = std::vector<int>());
    /*!
     * \brief Overload using raw-arrays.
     *
     * Array dimensions are not checked, otherwise identical to updateFourierGrid().
     */
    void updateFourierGrid(int depth, TypeDepth type, const int *anisotropic_weights = nullptr, const int *level_limits = nullptr);

    //! \brief Return the \b alpha parameter in the call to makeGlobalGrid(), or return 0 if the grid is not Global.
    double getAlpha() const{ return (isGlobal()) ? get<GridGlobal>()->getAlpha() : 0.0; }
    //! \brief Return the \b beta parameter in the call to makeGlobalGrid(), or return 0 if the grid is not Global.
    double getBeta() const{ return (isGlobal()) ? get<GridGlobal>()->getBeta() : 0.0; }
    //! \brief Return the \b order parameter in the call to makeLocalPolynomialGrid() or makeWaveletGrid(), or return -1 for all other grid types.
    int getOrder() const{ return (isLocalPolynomial()) ? get<GridLocalPolynomial>()->getOrder() : ((isWavelet()) ? get<GridWavelet>()->getOrder() : -1); }


    //! \brief Return the \b dimensions of the grid, i.e., number of model inputs.
    int getNumDimensions() const{ return (base) ? base->getNumDimensions() : 0; }
    //! \brief Return the \b outputs of the grid, i.e., number of model outputs.
    int getNumOutputs() const{ return (base) ? base->getNumOutputs() : 0; }
    //! \brief Return the underlying TasGrid::TypeOneDRule that gives the points and basis functions.
    TypeOneDRule getRule() const;
    /*!
     * \brief Return the character string that is the description of the user-provided tabulated rule.
     *
     * User-provided rules, i.e., Global grids with TasGrid::rule_customtabulated
     * have a description string that is returned by this method.
     * All other grids and rules will return an empty string.
     */
    const char* getCustomRuleDescription() const; // used only for Global Grids with rule_customtabulated

    //! \brief Return the number of points already associated with model values via loadNeededPoints().
    int getNumLoaded() const{ return (base) ? base->getNumLoaded() : 0; }
    //! \brief Return the number of points that should be provided to the next call of loadNeededPoints().
    int getNumNeeded() const{ return (base) ? base->getNumNeeded() : 0; }
    //! \brief Returns getNumLoaded() if positive, otherwise returns getNumNeeded(), see getPoints().
    int getNumPoints() const{ return (base) ? base->getNumPoints() : 0; }

    /*!
     * \brief Return the points already associated with model values.
     *
     * Returns a vector of size getNumLoaded() times getNumDimensions()
     * with the coordinates of the loaded points.
     * The vector is logically divided into strips of length getNumDimensions(),
     * each strip corresponds to a single point.
     */
    std::vector<double> getLoadedPoints() const{ std::vector<double> x; getLoadedPoints(x); return x; }
    /*!
     * \brief Overload that accepts the vector as a parameter.
     *
     * The vector is resized to getNumLoaded() times getNumDimensions() and overwritten with the loaded points.
     * See getLoadedPoints().
     */
    void getLoadedPoints(std::vector<double> &x) const{ x.resize(Utils::size_mult(getNumDimensions(), getNumLoaded())); getLoadedPoints(x.data()); }
    /*!
     * \brief Overload using raw-array, writes the loaded points to the first getNumLoaded() times getNumDimensions() entries of the array.
     *
     * Assumes the array size if at least getNumLoaded() times getNumDimensions(),
     * overwrites the entries with the loaded points in the same format as in getLoadedPoints().
     */
    void getLoadedPoints(double x[]) const;

    /*!
     * \brief Return the points that require model values.
     *
     * Returns a vector of size getNumNeeded() times getNumDimensions()
     * with the coordinates of the needed points.
     * The vector is logically divided into strips of length getNumDimensions(),
     * each strip corresponds to a single point.
     */
    std::vector<double> getNeededPoints() const{ std::vector<double> x; getNeededPoints(x); return x; }
    /*!
     * \brief Overload that accepts the vector as a parameter.
     *
     * The vector is resized to getNumNeeded() times getNumDimensions() and overwritten with the needed points.
     * See getNeededPoints().
     */
    void getNeededPoints(std::vector<double> &x) const{ x.resize(Utils::size_mult(getNumDimensions(), getNumNeeded())); getNeededPoints(x.data()); }
    /*!
     * \brief Overload using raw-array, writes the loaded points to the first getNumNeeded() times getNumDimensions() entries of the array.
     *
     * Assumes the array size if at least getNumNeeded() times getNumDimensions(),
     * overwrites the entries with the loaded points in the same format as in getNeededPoints().
     */
    void getNeededPoints(double *x) const;

    /*!
     * \brief Returns the loaded points if any, otherwise returns the needed points.
     *
     * Grid operations that do not require model values, e.g., getQuadratureWeights(),
     * operate with the loaded points, but in some cases all points may be needed,
     * e.g., right after a make method is called.
     * Thus, points is an alias to the loaded points unless there are no loaded points,
     * in which case points aliases to the needed points.
     * The overloads of this method have the same behavior as the corresponding overload
     * of the other get points methods.
     *
     * Note that if getNumPoints() is zero then the grid is empty().
     */
    std::vector<double> getPoints() const{ std::vector<double> x; getPoints(x); return x; }
    /*!
     * \brief Overload that accepts the vector as a parameter.
     *
     * See getNeededPoints().
     */
    void getPoints(std::vector<double> &x) const{ x.resize(Utils::size_mult(getNumDimensions(), getNumPoints())); getPoints(x.data()); }
    /*!
     * \brief Overload that accepts the raw array as an input.
     *
     * See getNeededPoints().
     */
    void getPoints(double x[]) const; // returns the loaded points unless no points are loaded, then returns the needed points

    /*!
     * \brief Returns a vector of size getNumPoints() of the quadrature weights of the grid.
     *
     * The quadrature is designed to work with weight of constant 1 unless the grid
     * \b rule is associated with a special weight. See TasGrid::TypeOneDRule for details.
     *
     * \return A vector of size getNumPoints() holding the quadrature weights;
     * the order of the weights matches the getPoints().
     */
    std::vector<double> getQuadratureWeights() const{ std::vector<double> w; getQuadratureWeights(w); return w; }
    /*!
     * \brief Overload that accepts the vector as a parameter.
     *
     * See getQuadratureWeights().
     */
    void getQuadratureWeights(std::vector<double> &weights) const{ weights.resize((size_t) getNumPoints()); getQuadratureWeights(weights.data()); }
    /*!
     * \brief Overload that accepts the raw array as an input.
     *
     * See getQuadratureWeights().
     */
    void getQuadratureWeights(double weights[]) const; // static memory, assumes that weights has size getNumPoints()

    /*!
     * \brief Returns the weights of the model outputs that combine to construct the approximation value at \b x.
     *
     * The sum of the model values times the weights will produce the approximation at \b x.
     * For problems where the model outputs can be represented by a vector,
     * it is recommended to use loadNeededPoints() and evaluate() methods
     * which have much better performance.
     * However, not all models can be easily represented as vector valued functions,
     * e.g., the discretization of the operators in a parametrized partial
     * differential equation can result in sparse matrices with very different fill.
     * Therefore, Tasmanian offers the option to compute these weights and leave to the
     * user to compute the corresponding weighted sum,
     * e.g., the matrix for each grid point is stored independently and
     * the action of the parametrized operator onto the vector is approximated
     * as a weighed linear combination of the individual matrix vector results.
     *
     * \param x is a vector of size getNumDimensions() with the coordinates of the point of interest
     * in the transformed domain.
     *
     * \returns A vector of size getNumPoints() with the interpolation weights,
     * the order of the weights matches the order of the getPoints().
     *
     * Note that using a vector \b x outside of the domain will result in undefined behavior,
     * but will not throw an exception.
     *
     * \throws std::runtime_error if \b x has an incorrect size.
     */
    std::vector<double> getInterpolationWeights(std::vector<double> const &x) const;
    /*!
     * \brief Overload that uses raw-array, does not check the array size.
     *
     * Identical to getInterpolationWeights() but does not throw if \b x is larger than getNumDimensions();
     * however, using shorter \b x is undefined behavior and will likely segfault.
     */
    std::vector<double> getInterpolationWeights(double const x[]) const{ std::vector<double> w((size_t) getNumPoints()); getInterpolationWeights(x, w.data()); return w; }
    /*!
     * \brief Overload that uses the vector as a parameter.
     *
     * Identical to getInterpolationWeights() but the \b weights vector will be resized
     * to size getNumPoints() and the entries will be overwritten with the output of getInterpolationWeights().
     */
    void getInterpolationWeights(const std::vector<double> &x, std::vector<double> &weights) const;
    /*!
     * \brief Overload that uses raw-array, does not check the array size.
     *
     * Identical to getInterpolationWeights() but does not throw if \b x is larger than getNumDimensions();
     * the length of \b x must be at least getNumDimensions() and the length of \b weighs must be at least getNumPoints().
     */
    void getInterpolationWeights(const double x[], double weights[]) const;

    /*!
     * \brief Provides the values of the model outputs at the needed points, or overwrites the currently loaded points.
     *
     * In order to construct an interpolant, Tasmanian needs the values of the model outputs at each grid point.
     * If there are needed points (i.e., getNumNeeded() is not zero), then \b vals must correspond to
     * the model outputs at the needed points, otherwise, it must correspond to the loaded points.
     *
     * \param vals A vector that is logically divided into strips of size getNumOutputs() each strip corresponding
     * to a single point. The order of the outputs must match the order of the points from either
     * getNeededPoints() or getLoadedPoints().
     * If getNumNeeded() is zero, the total size must be getNumOutputs() times getNumLoaded(),
     * otherwise, it must be getNumOutputs() times getNumNeeded().
     *
     * \throws std::runtime_error if \b vals has an incorrect size.
     *
     * \b Note: The needed points can always be cleared with clearRefinement()
     * and new needed points can be assigned with setAnisotropicRefinement() or setSurplusRefinement().
     */
    void loadNeededPoints(std::vector<double> const &vals); // checks if vals has size num_outputs X getNumNeeded()
    /*!
     * \brief Overload that uses a raw-array, does not check the array size.
     *
     * Identical to loadNeededPoints() but does not throw if \b vals has an incorrect size (but will segfault).
     */
    void loadNeededPoints(const double *vals);
    /*!
     * \brief Returns the model values that have been loaded in the gird.
     *
     * Returns a pointer to the internal data-structures, which \b must \b not be modified
     * and will be invalidated by any operation that affects the loaded points,
     * e.g., mergeRefinement() or loadNeededPoints().
     * The model values will follow the internal Tasmanian order, identical to getLoadedPoints().
     */
    const double* getLoadedValues() const{ return (empty()) ? nullptr : base->getLoadedValues(); }

    /*!
     * \brief Computes the value of the interpolant (or point-wise approximation) at the given point \b x.
     *
     * This is the reference implementation that does not use any acceleration mode even if one is set.
     * As a result, the calls to evaluate() are thread-safe but potentially slow.
     *
     * \param[in] x indicates the coordinate entries of a point within the domain of the sparse grid,
     * the size must be equal to getNumDimensions().
     * \param[out] y will contain the approximated model outputs corresponding to \b x,
     * the vector will be resized to getNumOutputs().
     *
     * \throws std::runtime_error if \b x has an incorrect size.
     *
     * \b Note: calling evaluate() for a point outside of the domain is Mathematically incorrect,
     * even though no exception will be generated.
     */
    void evaluate(std::vector<double> const &x, std::vector<double> &y) const;
    /*!
     * \brief Overload that uses raw-arrays, does not check the array size.
     *
     * Identical to evaluate() but does not throw, assumes \b x has size at least getNumDimensions()
     * and \b y is at least getNumOutputs(), will segfault if either is too short.
     */
    void evaluate(const double x[], double y[]) const;
    /*!
     * \brief Computes the value of the interpolant (or point-wise approximation) for a batch of points.
     *
     * Identical to calling evaluate() for each point defined by \b x, but uses the specified
     * acceleration mode and is potentially much faster, see TasGrid::TypeAcceleration for details.
     *
     * \tparam FloatType is either \b float or \b double indicating the precision to use.
     *
     * \param[in] x is logically divided into strips of size getNumDimensions() defining
     * the coordinates of points within the sparse grid domain, see also evaluate().
     *
     * \param[out] y is logically divided into \b x.size() / getNumDimensions() number of strips
     * each with length getNumOutputs(), provides the approximated model outputs for each
     * point defined by \b x. The vector will be resized, if the original size is incorrect.
     *
     * \throws std::runtime_error if instantiated with \b float and the current acceleration
     *      mode is neither CUDA nor MAGMA, see TasGrid::TypeAcceleration.
     *
     * \b Notes: this does not check if \b x.size() divides evenly.
     *
     * The batch call:
     * \code
     * grid.evaluateBatch(x, y);
     * \endcode
     * is Mathematically equivalent to:
     * \code
     * for(int i=0; i<x.size() / grid.getNumDimensions(); i++)
     *      grid.evaluate(&x[i * grid.getNumDimensions()],
     *                    &y[i * grid.getNumOutputs()]);
     * \endcode
     * However, depending on the acceleration mode, the performance can be significantly different.
     */
    template<typename FloatType> void evaluateBatch(std::vector<FloatType> const &x, std::vector<FloatType> &y) const;
    /*!
     * \brief Overload that uses raw-arrays.
     *
     * Raw-arrays do not provide size, thus the user must specify the number of points
     * and the arrays are assumed to have sufficient size (otherwise the call will segfault).
     * See also evaluate() and evaluateBatch().
     *
     * \param[in] x is logically organized into \b num_x strips of length getNumDimensions(),
     * each strip will hold the coordinates of a point within the sparse grid domain.
     *
     * \param[in] num_x is the total number of points stored in \b x.
     *
     * \param[out] y is logically organized into \b num_x strips of length getNumOutputs(),
     * each strip will be overwritten with the corresponding approximated model outputs.
     *
     * The following two calls are equivalent:
     * \code
     * grid.evaluateBatch(x, y); // using containers
     * \endcode
     * \code
     * grid.evaluateBatch(x.data(), x.size() / grid.getNumDimensions(), y.data()); // using raw arrays
     * \endcode
     */
    void evaluateBatch(const double x[], int num_x, double y[]) const;
    /*!
     * \brief Overload using single precision and GPU/CUDA acceleration.
     *
     * Works identical to the other raw-array overload but works only with CUDA and MAGMA acceleration modes.
     *
     * \param[in] x see the double precision array overload
     * \param[in] num_x see the double precision array overload
     * \param[in] y see the double precision array overload
     *
     * \throws std::runtime_error if the acceleration mode is not CUDA or MAGMA, see TasGrid::TypeAcceleration.
     * \throws std::runtime_error from failed calls to evaluateBatchGPU().
     */
    void evaluateBatch(const float x[], int num_x, float y[]) const;
    /*!
     * \brief Overload that uses GPU raw-arrays.
     *
     * Identical to the raw-array version of evaluateBatch(), but \b gpu_x and \b gpu_y must point
     * to memory allocated on the CUDA device matching getGPUID().
     * Requires that Tasmanian was compiled with CUDA support and CUDA
     * (or MAGAMA) acceleration mode has been enabled.
     *
     * \throws std::runtime_error if Tasmanian was not build with \b -DTasmanian_ENABLE_CUDA=ON.
     * \throws std::runtime_error if calling for a grid type that doesn't have appropriate CUDA kernels,
     *      e.g., local polynomial or wavelet grids with order more than 2.
     */
    template<typename FloatType> void evaluateBatchGPU(const FloatType gpu_x[], int cpu_num_x, FloatType gpu_y[]) const;
    /*!
     * \brief Equivalent to evaluate() with enabled acceleration or evaluateBatch() with a batch of one point.
     *
     * Some use cases still require performance but cannot batch multiple points together.
     * The evaluateFast() method will work the same as evaluate(), but will use
     * the specified acceleration mode.
     *
     * \b Note: in older versions of Tasmanian, this function used to call a
     * different set of algorithms optimized for single point evaluations;
     * currently, evaluateBatch() will automatically switch to the appropriate
     * mode depending on \b x.size() and \b num_x.
     * Thus, this method is now an alias to evaluateBatch().
     */
    template<typename FloatType> void evaluateFast(const FloatType x[], FloatType y[]) const{ evaluateBatch(x, 1, y); }
    /*!
     * \brief Alias to evaluateBatch().
     *
     * Provided for consistency and backwards compatibility.
     */
    template<typename FloatType> void evaluateFast(std::vector<FloatType> const &x, std::vector<FloatType> &y) const{ evaluateBatch(x, y); }
    /*!
     * \brief Computes the integral of each model output over the sparse grid domain.
     *
     * The integration weight is assumed 1 unless another weight is associated with
     * the underlying one dimensional rule, see TasGrid::TypeOneDRule for details.
     *
     * \param q will be resized to getNumOutputs() and overwritten with the approximated
     * integral of each model output.
     *
     * The output of integrate() is Mathematically equivalent to calling getQuadratureWeights()
     * and computing the sum of model values times weights.
     * However, when the model values have been loaded integrate() is faster and more convenient.
     */
    void integrate(std::vector<double> &q) const;
    /*!
     * \brief Overload that uses a raw-array.
     *
     * Equivalent to integrate() but \b q must have sufficient size to write the result.
     *
     * \param q must have size of at least getNumOutputs().
     */
    void integrate(double q[]) const;

    //! \brief Returns \b true if the grid is of type global, \b false otherwise.
    bool isGlobal() const{ return base && base->isGlobal(); }
    //! \brief Returns \b true if the grid is of type sequence, \b false otherwise.
    bool isSequence() const{ return base && base->isSequence(); }
    //! \brief Returns \b true if the grid is of type local polynomial, \b false otherwise.
    bool isLocalPolynomial() const{ return base && base->isLocalPolynomial(); }
    //! \brief Returns \b true if the grid is of type wavelet, \b false otherwise.
    bool isWavelet() const{ return base && base->isWavelet(); }
    //! \brief Returns \b true if the grid is of type Fourier, \b false otherwise.
    bool isFourier() const{ return base && base->isFourier(); }
    //! \brief Returns \b true if the grid is empty (no type), \b false otherwise.
    bool isEmpty() const{ return !base; }
    //! \brief Returns \b true if the grid is empty (no type), \b false otherwise.
    bool empty() const{ return !base; }

    /*!
     * \brief Set a linear domain transformation.
     *
     * By default integration and interpolation are performed on a canonical interval [-1, 1],
     * with the exception of Fourier grids using [0, 1] and some Gauss rules,
     * see TasGrid::TypeOneDRule.
     * The linear transformation will shift the interval to an arbitrary [a, b]
     * (or shift and scale for the unbounded case).
     * Different values can be specified for each dimension and the transformation
     * will be automatically applied to every operation that uses points, e.g.,
     * getPoints() and evaluate() will return/accept only transformed points.
     *
     * Setting or changing the transformation will change the points and weights,
     * therefore, the transformations should be set immediately after calling a make
     * command and before calling get-points or computing model values.
     * Changing the transformation will not throw, but will likely invalidate the loaded data
     * and should be used with extreme caution (the validity of the underlying Mathematics
     * is left to the user to analyze).
     *
     * \param a with size getNumDimensions() specifies the \b a transformation parameter
     * for each input
     * \param b with size getNumDimensions() specifies the \b b transformation parameter
     * for each input
     *
     * \throws std::runtime_error if the grid is empty.
     * \throws std::invalid_argument if either input has incorrect size.
     */
    void setDomainTransform(std::vector<double> const &a, std::vector<double> const &b);
    /*!
     * \brief Overload using raw-arrays.
     *
     * Identical to setDomainTransform() but does not check for the size of \b a and \b b,
     * although it still checks if the grid is empty.
     */
    void setDomainTransform(const double a[], const double b[]);
    /*!
     * \brief Returns \b true if a linear domain transformation has been set, \b false otherwise.
     *
     * Allows to check if there is a set transformation or if working on the canonical domain.
     */
    bool isSetDomainTransfrom() const;
    /*!
     * \brief Removes the domain transformation.
     *
     * Use with extreme caution, see setDomainTransform().
     */
    void clearDomainTransform();
    /*!
     * \brief Returns the two vectors used to call setDomainTransform().
     *
     * If the grid is empty or if no transformation has been set,
     * this will return empty vectors.
     *
     * \param[out] a will be resized to getNumDimensions() and overwritten with the a vector of setDomainTransform().
     * \param[out] b will be resized to getNumDimensions() and overwritten with the b vector of setDomainTransform().
     *
     * \b Note: this works the same regardless of which setDomainTransform() overload has been used.
     */
    void getDomainTransform(std::vector<double> &a, std::vector<double> &b) const;
    /*!
     * \brief Returns the values of the two vectors used to call setDomainTransform().
     *
     * Assuming that the inputs have sufficient size, will overwrite the getNumDimensions() entries
     * with the domain transformation.
     *
     * \throws std::runtime_error if the grid is empty or if the domain transformation has not been set.
     *
     * \b Note: this works the same regardless of which setDomainTransform() overload has been used.
     */
    void getDomainTransform(double a[], double b[]) const;

    /*!
     * \brief Set conformal transformation using truncated Maclaurin series of the arcsin() function.
     *
     * Conformal transformations apply a non-linear map to the points, weights and basis functions of a grid,
     * in an attempt to accelerate convergence for some types of functions.
     * The objective is to expand the area of analytic extension of the target function and thus accelerate convergence;
     * however, if applied to the wrong function, conformal transformations can deteriorate the accuracy.
     * Characterizing the functions that would most benefit from conformal maps is an active area of research.
     *
     * The truncated Maclaurin series of the arcsin() work well with functions that have a pole close
     * to the edges of the domain. See:\n
     * P. Jantsch, C. G. Webster, <a style="font-weight:bold" href="https://link.springer.com/chapter/10.1007/978-3-319-75426-0_6">
     * Sparse Grid Quadrature Rules Based on Conformal Mappings</a>,
     * Sparse Grids and Applications - Miami 2016 pp 117--134.
     *
     * \param truncation is a vector of size getNumDimensions() that indicates the number of terms
     *      to keep in each direction. The more terms, the more pronounced the transformation becomes;
     *      however, too many terms can in fact produce a pole even worse than the original.
     */
    void setConformalTransformASIN(std::vector<int> const &truncation);
    //! \brief Returns \b true if conformal transformation has been set.
    bool isSetConformalTransformASIN() const;
    //! \brief Removes any currently set transformation.
    void clearConformalTransform();
    //! \brief Fills the array with the values of the set transformation.
    std::vector<int> getConformalTransformASIN() const;

    /*!
     * \brief Removes the currently set level limits.
     *
     * Once level limits have been set (by either the make or set-refinement commands),
     * the limits will be used for all follow-on commands unless either overwritten with a new set of limits
     * of cleared with this command.
     */
    void clearLevelLimits(){ llimits.clear(); }
    /*!
     * \brief Return the currently set level limits.
     *
     * Returns the limits that have been set with the last command.
     * If no limits have been set, an empty vector will be returned.
     */
    std::vector<int> getLevelLimits() const{ return llimits; }

    /*!
     * \brief Set refinement using anisotropic estimate for the optimal points.
     *
     * Computes the anisotropic coefficients based on the current set of loaded points,
     * then adds more points to the grid by selecting the points with largest coefficients.
     * The new points are set to \b needed. See also estimateAnisotropicCoefficients()
     *
     * \param type indicates the type of estimate to use, e.g., total degree or curved;
     *      regardless of the specified rule the interpolation estimate is computed (not quadrature).
     *
     * \param min_growth is the minimum number of points to add to the grid, e.g., if the model can be executed
     *      in parallel then minimum number of points is needed to ensure occupancy for all computing resources.
     *      The value of \b min_growth can never be less than 1, but the actual number of points
     *      can exceed the \b min_growth depending on the weights and growth of the one dimensional rule.
     *
     * \param output indicates which output to use to compute the estimate, using -1 indicates to use all outputs
     *      (possible with Sequence and Fourier grids only, Global grids require a specific output).
     *
     * \param level_limits (if not empty) will be used to overwrite the currently set limits.
     *      The limits must be either empty or have size getNumDimensions();
     *      if empty, the current set of limits will be used.
     *
     * \throws std::runtime_error if called during dynamic construction, or there are no loaded points,
     *      or there are no outputs.
     *
     * \throws std::invalid_argument if \b min_growth is not positive, \b output is out of range,
     *      or \b level_limits has the wrong size.
     */
    void setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits);
    /*!
     * \brief Overload using raw-array.
     *
     * Identical to setAnisotropicRefinement() with the exception that level_limits is a raw array,
     * empty is equivalent to \b nullptr and the size is not checked.
     */
    void setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const int *level_limits = nullptr);

    /*!
     * \brief Estimate the anisotropic rates of coefficient decay for different direction.
     *
     * Available for Global, Sequence and Fourier grids, the anisotropic coefficients describe the space
     * of the dominant basis functions needed to construct optimal approximation to the model.
     * See the documentation of TasGrid::TypeDepth of the different formulas.
     * The estimate requires that values have been loaded in the grid, i.e.,
     * getNumOutputs() > 0 and getNumLoaded() > 0.
     *
     * \param type is the assumed type of optimal space, i.e., total degree, total degree with log-correction, or hyperbolic cross section.
     * \param output determines the output to use for the inference of the coefficients,
     *      for Global grid it must be between 0 and getNumOutputs() -1, in other cases,
     *      it can also be set to -1 to indicate "all outputs" or for each basis use the larges coefficient among all outputs.
     *
     * \returns the \b xi and \b eta (if used) parameters of the estimate indicated by \b type.
     *
     * \throws std::runtime_error if the grid is empty, has no outputs or no loaded points.
     * \throws std::invalid_argument if the output is out of range.
     */
    std::vector<int> estimateAnisotropicCoefficients(TypeDepth type, int output) const{ std::vector<int> w; estimateAnisotropicCoefficients(type, output, w); return w; }
    /*!
     * \brief Overload that writes the result to a parameter.
     *
     * The inputs are equivalent to estimateAnisotropicCoefficients()
     * but the result is returned into \b weights.
     */
    void estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const;

    /*!
     * \brief Refine the grid based on the surplus coefficients, Sequence grids and Global grids with a sequence rule.
     *
     * Using the relative magnitude of the surplus coefficients, add more points to the grid
     * and set them to needed. The approach differs from the local (polynomial or wavelet) case
     * in the interpretation of the surpluses, instead of local-spacial estimate
     * the surpluses are interpreted as indicators of needed polynomial basis functions.
     *
     * \param tolerance indicates the cutoff threshold for refinement, points will not be
     *      included once the magnitude of the relative surplus drops below the tolerance.
     * \param output indicates the output to use for the surpluses, the Sequence grid
     *      accepts -1 to indicate the use of all outputs.
     * \param level_limits indicates a new set of limits, if empty the currently set limits
     *      will be used.
     *
     * \throws std::runtime_error if the called during the construction process, if the grid is empty,
     *      of if the grid has no outputs or values.
     * \throws std::invalid_argument if the \b output is out of range, \b tolerance is negative,
     *      or if the \b level_limits has the wrong size.
     */
    void setSurplusRefinement(double tolerance, int output, std::vector<int> const &level_limits);
    //! \brief Overload that uses array for the level limits.
    void setSurplusRefinement(double tolerance, int output, const int *level_limits = nullptr);

    /*!
     * \brief Refine the grid based on the surplus coefficients, Local-Polynomial and Wavelet grids.
     *
     * Using the relative magnitude of the surplus coefficients, add more points to the grid
     * and set them to needed. This method uses the hierarchical and local structure of the one dimensional
     * rule and interprets the surplus as a local error indicator.
     * The refinement can be combined with tests for completeness of the hierarchy
     * (prioritizing parents or even the entire ancestry) or local anisotropy that
     * can manifest even in globally isotropic cases. See TasGrid::TypeRefinement
     * for details.
     *
     * \param tolerance indicates the cutoff threshold for refinement, points will not be
     *      included once the magnitude of the relative surplus drops below the tolerance.
     * \param criteria indicates how to prioritize the hierarchy and/or local anisotropy.
     * \param output indicates a specific output to use for the refinement, by default (when -1)
     *      all model outputs will be considered together.
     * \param level_limits indicates a new set of limits, if empty the currently set limits
     *      will be used.
     * \param scale_correction is a set of weights that would multiply the surpluses before
     *      the tolerance test; the correction can be used to modify the threshold test (e.g.,
     *      multiply the surplus by the integral of the basis functions) or to guide
     *      the refinement towards certain regions of the domain.
     *      The values of the correction terms are organized in order that matches the order
     *      of getNumLoaded() and there is one weight per active output (either 1 or getNumOutputs()
     *      depending whether \b output is -1 or positive).
     *      If the scale correction is empty, then no correction is used (i.e., using correction of 1.0).
     *
     * \throws std::runtime_error if called during construction, or if the grid is empty or has no
     *      outputs or loaded points, of if the grid has incompatible type (not local polynomial or wavelet).
     *
     * \throws std::invalid_argument if \b output is out of range, of if the level limits and/or
     *      scale correction have incorrect size.
     */
    void setSurplusRefinement(double tolerance, TypeRefinement criteria, int output,
                              std::vector<int> const &level_limits, std::vector<double> const &scale_correction = std::vector<double>()); // -1 indicates using all outputs
    /*!
     * \brief Overload that uses raw-arrays.
     *
     * The \b scale_correction is a potentially large vector and using a raw array avoids a large data-copy
     * when calling from C/Python/Fortran interfaces.
     * Otherwise the method behaves the same, but does not throw if the arrays have incorrect size (will probably segfault).
     */
    void setSurplusRefinement(double tolerance, TypeRefinement criteria, int output = -1, const int *level_limits = nullptr, const double *scale_correction = nullptr);

    /*!
     * \brief Remove all needed points from the grid.
     *
     * Once a refinement is set, but before the new values have been loaded, the refinement can be canceled
     * with this command. After this call, all needed points will be erased.
     */
    void clearRefinement();
    /*!
     * \brief Merges the loaded and needed points into a single grid, resets all loaded values to zero.
     *
     * This method allows refinement to be used in cases where the model values cannot be computed at the grid points,
     * e.g., when working with random or unstructured data.
     * Once a refinement is set, the new points can be merged without model values which will result in a larger grid
     * with all (previous and new) loaded values reset to 0.0. Afterwards, the setHierarchicalCoefficients() method
     * can be used to set a new set of coefficients, e.g., inferred from the data and the hierarchical basis values.
     */
    void mergeRefinement(); // merges all points but resets all loaded values to 0.0

    /*!
     * \brief Begin a dynamic construction procedure.
     *
     * Initializes the internal data-structures needed for the construction procedure (which is pretty cheap)
     * and makes a call to \b clearRefinement().
     *
     * \b Note: after this call, setSurplusRefinement() and setAnisotropicRefinement()
     * cannot be called until finishConstruction() is issued.
     *
     * \b Note: the construction process can be initiated before any model values have been loaded,
     * in such case, the initial set of points will always come first in a call to getCandidateConstructionPoints().
     */
    void beginConstruction();
    /*!
     * \brief Returns \b true if the dynamic construction procedure has been initialized, \b false otherwise.
     *
     * Simply returns the internal flag.
     */
    bool isUsingConstruction() const{ return usingDynamicConstruction; }
    /*!
     * \brief Generate a sorted list of points weighted by descending importance.
     *
     * The \e importance is inferred from the user provided anisotropic weighs
     * using the formula specified by the \b type, see TasGrid::TypeDepth.
     * The full tensor types will fallback to \b type_level.
     *
     * Unlike the batch procedures in setAnisotropicRefinement(), there is no expectation that
     * the entire batch is processed, in fact only a small subset of the most important
     * points should be computed and loaded, then the getCandidateConstructionPoints()
     * should be called again and it will return an updated list of points.
     * Enough points should be used from the top of the list to ensure occupancy across
     * computing resources, e.g., CPU cores or MPI ranks,
     * but the bottom set of points will contribute less and less to the overall accuracy.
     *
     * \param type sets the formula to use when weighting the potential points, see TasGrid::TypeDepth.
     * \param anisotropic_weights are the xi and eta parameters for the formula,
     *      the vector must have the correct size, either getNumDimensions() or twice
     *      as much to handle the curved weights.
     * \param level_limits (if not empty) will be used to overwrite the currently set limits.
     *      The limits must be either empty or have size getNumDimensions();
     *      if empty, the current set of limits will be used.
     *
     * \returns A vector organized in strips of length getNumDimensions() that indicate
     *      the coordinates of the points to use as model inputs, unlike the getPoints()
     *      command, the points here are arranged in decreasing importance.
     *
     * \throws std::runtime_error if the grid is empty, Local Polynomial or this is called before beginConstruction().
     *
     * \throws std::invalid_argument if the \b anisotropic_weights or \b level_limits have incorrect size.
     */
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, std::vector<int> const &anisotropic_weights = std::vector<int>(),
                                                       std::vector<int> const &level_limits = std::vector<int>());
    /*!
     * \brief Essentially the same as \b getCandidateConstructionPoints() but the weights are obtained from a call to \b estimateAnisotropicCoefficients().
     *
     * This method is the construction equivalent to setAnisotropicRefinement().
     * One notable difference is that this function will not throw if there are no loaded points,
     * instead isotropic coefficient will be used
     * until enough points are loaded so that estimates for the coefficients can be computed.
     *
     * \param type sets the formula to use when weighting the potential points, see TasGrid::TypeDepth.
     * \param output indicates which coefficients will be used to estimate the anisotropic decay rate,
     *      when working with Sequence and Fourier grids this can be set to -1 to use all outputs.
     * \param level_limits (if not empty) will be used to overwrite the currently set limits.
     *      The limits must be either empty or have size getNumDimensions();
     *      if empty, the current set of limits will be used.
     *
     * \returns a vector organized in strips of length getNumDimensions() that indicate
     *      the coordinates of the points to use as model inputs, unlike the getPoints()
     *      command, the points here are arranged in decreasing importance.
     *
     * \throws std::runtime_error if the grid is empty, Local Polynomial or this is called before beginConstruction().
     *
     * \throws std::invalid_argument if the \b level_limits have incorrect size or \b output is out of range.
     */
    std::vector<double> getCandidateConstructionPoints(TypeDepth type, int output, std::vector<int> const &level_limits = std::vector<int>());

    /*!
     * \brief Returns a sorted list of points weighted by descending importance using the hierarchical surpluses.
     *
     * This is the construction equivalent to the Local Polynomial setSurplusRefinement().
     * The inputs are the same as setSurplusRefinement() except the returned points will be sorted by
     * decreasing surpluses. Similar to the other getCandidateConstructionPoints() overloads,
     * this can be called before any model values have been loaded.
     */
    std::vector<double> getCandidateConstructionPoints(double tolerance, TypeRefinement criteria, int output = -1,
                                                       std::vector<int> const &level_limits = std::vector<int>(),
                                                       std::vector<double> const &scale_correction = std::vector<double>());
    /*!
     * \brief Add pairs of points with associated model values.
     *
     * This is the construction equivalent to loadNeededPoints(), the main difference is that any
     * number of points can be loaded here and the points can be in any arbitrary order
     * (they have to correspond to the model values in this call only).
     *
     * \param x is a vector with strips of size getNumDimensions() indicating the points where the model
     *      values were computed. The points do not have to be in any order; however, every point has to match
     *      a potential grid points.
     * \param y is a vector with strips of size getNumOutputs() indicating the model outputs corresponding
     *      each of the provided points.
     *
     * \throws std::runtime_error if the number of strips in \b y are less than those in \b x.
     *
     * \b Note: regardless of the grid type and rule, as the depth/level increases the points become dense
     *      in the domain; thus every point is theoretically a potential grid points.
     *      However, if a random point is chosen than the level may easily overflow the range of the \b int type,
     *      the safest option is to make sure the points match the ones returned by getCandidateConstructionPoints().
     */
    void loadConstructedPoints(std::vector<double> const &x, std::vector<double> const &y);
    /*!
     * \brief Same as \b loadConstructedPoint() but using arrays in place of vectors (array size is not checked)
     *
     * Does not throw on incorrect array size, but will likely segfault.
     */
    void loadConstructedPoints(const double x[], int numx, const double y[]);
    /*!
     * \brief End the procedure, clears flags and unused constructed points, can go back to using regular refinement
     *
     * After this call, the construction methods getCandidateConstructionPoints() and loadConstructedPoint()
     * cannot be used until a new call to beginConstruction().
     *
     * \b Note: finalizing the construction process can potentially lead to loss of data.
     * The constructed points can be loaded in any order, but the points need to satisfy constraints
     * before being incorporated within a grid. Some grids require that points form a lower complete set,
     * or the point must form a connected graph, or there must be enough points to complete a tensor
     * (when the rules grow by more than one point). In such scenarios, the points and values are stored
     * in a temporary structure until enough data is present to add them to the grid.
     * Finalizing the construction will delete all data for points that have not been incorporated yet.
     */
    void finishConstruction();

    /*!
     * \brief Return a reference to the internal data-structure that stores the hierarchical coefficients.
     *
     * All types of grids (except Global grids), use a hierarchical basis representation where the interpolant
     * is expressed as a set of coefficients times the basis functions.
     * For Local Polynomial and Wavelet grids, the coefficients are commonly known as hierarchical surpluses;
     * for Sequence grid, the coefficients correspond to the Newton polynomials;
     * for Fourier grids, the coefficients are the linear combination of discrete Fourier transform coefficients.
     *
     * The returned array has getNumLoaded() strips of size getNumOutputs() where each strip corresponds to
     * one basis function. In the case of Fourier grids, the coefficients are complex numbers and
     * (for performance reasons) the real and complex parts are stored separately, i.e.,
     * the first getNumLoaded() strips hold the real parts and there is a second set of getNumLoaded()
     * strips that hold the complex components. If the grid is empty or there are no outputs or
     * no loaded points, then this returns \b nullptr.
     *
     * \b Note: modifying the coefficients through this pointer leads to undefined behavior,
     *      use setHierarchicalCoefficients() instead.
     */
    const double* getHierarchicalCoefficients() const;
    /*!
     * \brief Overwrites the current set of coefficients (and loaded values) with the ones provided.
     *
     * Discards the current set of loaded values and the associated hierarchical coefficients,
     * and replaces both with the data provided here. The coefficients are overwritten,
     * while the values are inferred, i.e., the opposed from the use case of loadNeededPoints()
     * where the model values are provided and the coefficients are computed.
     *
     * \param c is a vector of getNumLoaded() strips of size getNumOutputs(),
     *      each strip corresponds to the coefficients of one basis function.
     *      Fourier coefficients are an exceptions as they require twice as many strips where the
     *      first set corresponds to the real components and the second set corresponds to the
     *      complex components of the coefficients.
     *
     * \throws std::runtime_error is the number of coefficients is incorrect.
     */
    void setHierarchicalCoefficients(const std::vector<double> &c);
    /*!
     * \brief Overload that uses raw-arrays.
     *
     * Identical to setHierarchicalCoefficients() but does not check for the size of the array.
     */
    void setHierarchicalCoefficients(const double c[]);

    /*!
     * \brief Computes the values of the hierarchical function basis at the specified points.
     *
     * The method for getInterpolationWeights() computes the values of the nodal basis function,
     * e.g., Lagrange polynomials, while this computes the values of the hierarchical functions, e.g.,
     * Newton polynomials. The call is optimized for multiple points at a time.
     * The output consists of strips of size getNumPoints() and one strip corresponds to one
     * point provided in \b x. Effectively this constructs a matrix in column major format
     * where each column corresponds to a single input point \b x and each row corresponds
     * to a single grid point.
     *
     * When working with Fourier grids, the real and complex part of each basis is interlaced,
     * i.e., the strip size is twice as large; see the array overload.
     *
     * \param[in] x has strips of size getNumDimensions() indicating a point in the domain
     *      where the basis is to be computed.
     *
     * \param[out] y will have the same number of strips with size getNumPoints()
     *      (or twice that to accommodate the real and complex parts of the basis).
     *
     * \b Note: if the the output from getHierarchicalCoefficients() is interpreted as
     *      a matrix in column major format with leading dimension of getNumOutputs(),
     *      and if the output of this call is interpreted as a matrix in column major
     *      format with leading dimension getNumPoints(), then the result of the product
     *      of the coefficients times the basis will be equal to the result of evaluateBatch().
     */
    void evaluateHierarchicalFunctions(std::vector<double> const &x, std::vector<double> &y) const;
    /*!
     * \brief Overload that returns the result.
     *
     * Useful for direct initialization of vectors.
     */
    std::vector<double> evaluateHierarchicalFunctions(std::vector<double> const &x) const{
        std::vector<double> y;
        evaluateHierarchicalFunctions(x, y);
        return y;
    }
    /*!
     * \brief Array overload, the inputs must have pre-allocated and correct size.
     *
     * The size of \b x must be \b num_x times getNumDimensions() and the size of \b y
     * must be \b num_x times getNumPoints() (or twice that for the Fourier grids).
     *
     * Example of returning the result in a vector of complex numbers:
     * \code
     * TasGrid::TasmanianSparseGrid grid = TasGrid::makeFourierGrid(...);
     * int num_x = ...; // set the number of points
     * std::vector<double> x(grid.getNumDimensions() * num_x);
     * // initialize x
     * std::vector<std::complex<double>> y(grid.getNumPoints() * num_x);
     * grid.evaluateHierarchicalFunctions(x.data(), num_x, reinterpret_cast<double*>(y.data()));
     * // at this point y is loaded with the complex numbers
     * \endcode
     */
    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const;

    /*!
     * \brief Constructs a sparse matrix with the values of the hierarchical basis functions.
     *
     * The local basis functions associated with Local Polynomial and Wavelet grids
     * means that only a small set of the basis functions will be non-zero at any
     * given point in the domain. Effectively, the matrix build by
     * evaluateHierarchicalFunctions() will be sparse.
     * This method generates the sparse matrix in compressed format in non-zeroes per entry in \b x.
     *
     * \param[in] x has strips of size getNumDimensions() indicating a point in the domain
     *      where the basis is to be computed.
     * \param[out] pntr will have size one more than the number of strips in \b x,
     *      pntr[i] indicate the starting offsets of the entries for the i-th point,
     *      (the first entry will always be zero) and pntr.back() is the total number of non-zeros.
     * \param[out] indx holds the indexes of the supported basis functions,
     *      i.e., indx[pntr[i]] ... indx[pntr[i+1] - 1] are the indexes of the basis functions
     *      supported over the i-th point.
     * \param[out] vals are the numeric values of the basis functions corresponding to
     *      the indexes in \b indx.
     */
    void evaluateSparseHierarchicalFunctions(const std::vector<double> &x, std::vector<int> &pntr, std::vector<int> &indx, std::vector<double> &vals) const;

    /*!
     * \brief Returns the integrals of the hierarchical basis functions.
     *
     * Returns a vector of size getNumPoints() that holds the integral of each
     * basis function. If the output of getHierarchicalCoefficients() is interpreted as a matrix,
     * the product of that matrix times this vector is the same as the output of TasmanianSparseGrid::integrate().
     *
     * One use case of the integrals is to add scale correction to the surplus refinement,
     * e.g., rescale the coefficients by the integral of the basis function:
     * \code
     * auto grid = TasGrid::makeLocalPolynomialGrid(...);
     * // ... load values ... //
     * auto correction = grid.integrateHierarchicalFunctions();
     * grid.setSurplusRefinement(1.E-4, TasGrid::refine_classic, 0, {}, correction);
     * \endcode
     */
    std::vector<double> integrateHierarchicalFunctions() const{
        std::vector<double> integrals;
        integrateHierarchicalFunctions(integrals);
        return integrals;
    }
    /*!
     * \brief Overload where the vector is passes as a parameter.
     *
     * \param integrals will be resized to getNumPoints() and overwritten with
     *      the integrals of the hierarchical basis functions.
     */
    void integrateHierarchicalFunctions(std::vector<double> &integrals) const{
        integrals.resize(getNumPoints());
        integrateHierarchicalFunctions(integrals.data());
    }
    /*!
     * \brief Overload using raw-arrays.
     *
     * Useful mostly for the C/Python/Fortran interfaces.
     *
     * \param integrals must have size getNumPoints() and will be overwritten with
     *      the integrals of the hierarchical basis functions.
     */
    void integrateHierarchicalFunctions(double integrals[]) const;

    /*!
     * \brief Returns the powers of the polynomial that span the effective basis, Global and Sequence grids only.
     *
     * Different rules have different growth in number of points and different exactness
     * with respect to integration and interpolation.
     * This method returns the actual polynomial space spanned by the basis used in the grid.
     *
     * \param interpolation determines whether to look for the space of exactly interpolated
     *      polynomials (\b true), or the space of exactly integrated polynomials (\b false).
     *
     * \returns a vector with getNumPoints() number of strips each having getNumDimensions() entries,
     *      each integer corresponds to the polynomial power in that direction, e.g.,
     *      for a two dimensional grid (0, 0, 1, 0) indicates the constant in all directions
     *      and linear in the first input constant in the second.
     *
     * \throws std::runtime_error if the grid is not Global or Sequence.
     */
    std::vector<int> getGlobalPolynomialSpace(bool interpolation) const;

    /*!
     * \brief Prints short human-readable text describing the grid properties.
     *
     * Among the outputs are type of the grid, number of inputs/outputs/points,
     * one dimensional rule, and currently selected acceleration type.
     *
     * \param os indicates the stream to use for the text output,
     *      defaults to std::cout.
     */
    void printStats(std::ostream &os = std::cout) const;

    /*!
     * \brief Set the number of threads to use at the back-end.
     *
     * Used
     */
    void setNumThreads(size_t new_num_threads){
        num_threads = new_num_threads;
        if (!empty()) base->setNumThreads(new_num_threads);
    }
    /*!
     * \brief Change the current acceleration mode to the one specified.
     *
     * Sets a new acceleration mode and releases the cached data-structures for the old mode.
     *
     * \param acc is the new acceleration mode, see TasGrid::TypeAcceleration.
     *
     * \b Note: any acceleration method can be set regardless whether it is available.
     *      If the selected mode has not been enabled in CMake, this will select the
     *      "next-best" mode, thus enableAcceleration() is a suggestion rather than a hard request.
     *      See getAccelerationType().
     */
    void enableAcceleration(TypeAcceleration acc);
    /*!
     * \brief Set the preferred back-end algorithm for Local Polynomial grids.
     *
     * Usually the Local Polynomial grids use sparse data-structures and sparse linear algebra,
     * but if the fill not sufficiently small then dense methods can yield better performance
     * at the expense of higher memory footprint.
     * In TasGrid::accel_cpu_blas mode, the dense algorithm will be used when the fill increases
     * above the 10% threshold. In CUDA mode, even if the fill is computed, switching modes
     * incurs a large overhead and thus the sparse algorithm is always favored.
     * Sparse computations work better on newer GPU architectures (e.g., Pascal and Volta)
     * and consumer (gaming) cards with reduced double-precision capabilities,
     * but older devices coupled with small grids may work better with the dense algorithm.
     * Hence, Tasmanian includes the manual option to select the desired algorithm.
     *
     * \param favor if set to \b true will force the sparse back-end, if set to \b false will
     *      force the dense back-end. If the mode has been forced already, calling the method
     *      with the converse \b favor will reset mode to the default (automatic) selection.
     */
    void favorSparseAcceleration(bool favor);
    /*!
     * \brief Returns the current effective acceleration mode.
     *
     * Returns the acceleration mode that will be used, i.e., the one selected internally
     * based on the request made in enableAcceleration().
     */
    TypeAcceleration getAccelerationType() const{ return acceleration; }
    /*!
     * \brief Returns whether a specific mode can be enabled.
     *
     * Based on the CMake compile time options, this method returns \b true
     * for all acceleration modes that will be enabled, and \b false for all
     * modes that will be replaced by a fallback alternative.
     */
    static bool isAccelerationAvailable(TypeAcceleration acc);

    /*!
     * \brief Select the current CUDA device.
     *
     * Select the CUDA device to be used for the CUDA acceleration types,
     * default device is \b 0.
     * \param new_gpu_id is the CUDA device ID of the new device,
     *      the number is between 0 and getNumGPUs() - 1.
     *
     * \throws std::runtime_error if the \b new_gpu_id is out of range.
     */
    void setGPUID(int new_gpu_id);
    /*!
     * \brief Returns the currently set CUDA device.
     *
     * Does not throw if CUDA is not enabled in CMake, instead the default \b 0 is returned.
     */
    int getGPUID() const{ return gpu_id; }
    /*!
     * \brief Return the number of visible CUDA devices.
     *
     * Simple wrapper to cudaGetDeviceCount().
     *
     * Use the \b tasgrid command line tool to see all available devices:
     * \code
     *     tasgrid -v
     * \endcode
     */
    static int getNumGPUs();
    /*!
     * \brief Return the available device memory, in units of MB.
     *
     * Simple wrapper to cudaGetDeviceProperties() returning totalGlobalMem divided by 2^20.
     * \param gpu is the CUDA device ID to be queried, if the device is out of range
     *      then \b 0 will be returned (i.e., Tasmanian will not throw).
     */
    static int getGPUMemory(int gpu);
    /*!
     * \brief Return the CUDA device name.
     *
     * Simple wrapper to cudaGetDeviceProperties() returning "name".
     * \param gpu is the CUDA device ID to be queried, if the device is out of range
     *      then empty string will be returned (i.e., Tasmanian will not throw).
     * \returns the CUDA device name.
     */
    static std::string getGPUName(int gpu);

    /*!
     * \brief Computes the values of the hierarchical function basis at the specified points (CUDA version).
     *
     * Equivalent to evaluateHierarchicalFunctions() but using arrays allocated on the CUDA device.
     * \tparam FloatType must be either float or double to indicate the precision used by the CUDA kernels.
     *
     * \param gpu_x must have size getNumDimensions() times \b cpu_num_x and must be allocated on
     *      the currently set CUDA device.
     * \param cpu_num_x is an integer (located on the CPU memory) indicating the number of points.
     * \param gpu_y must have size getNumPoints() times \b cpu_num_x and must be allocated on
     *      the currently set CUDA device.
     *
     * \throws std::runtime_error if the grid is Global or Wavelet or if the currently set acceleration mode
     *      is not compatible, i.e., TasGrid::accel_none or TasGrid::accel_cpu_blas.
     *      Also, if CUDA has not been enabled at compile time.
     *
     * \b Note: will not work for LocalPolynomial grids with order bigger than 2.
     */
    template<typename FloatType>
    void evaluateHierarchicalFunctionsGPU(const FloatType gpu_x[], int cpu_num_x, FloatType gpu_y[]) const;
    /*!
     * \brief Computes the values of the hierarchical function basis at the specified points (sparse/CUDA version).
     *
     * Equivalent to evaluateSparseHierarchicalFunctions() but using arrays allocated on the CUDA device.
     * \tparam FloatType must be either float or double to indicate the precision used by the CUDA kernels.
     *
     * \param[in] gpu_x must have size getNumDimensions() times \b cpu_num_x and must be allocated on
     *      the currently set CUDA device.
     * \param[in] cpu_num_x is an integer (located on the CPU memory) indicating the number of points.
     * \param[out] gpu_pntr will be allocated to size \b cpu_num_x + 1 and will hold the offsets of
     *      indexes for each point, the current memory will not be freed.
     * \param[out] gpu_indx will be allocated to the number of non-zeros and will hold the indexes
     *      of the non-zeros for each point, the current memory will not be freed.
     * \param[out] gpu_vals will be allocated to the number of non-zeros and will hold the non-zero
     *      values of the basis function, the current memory will not be freed.
     * \param[out] num_nz is an integer located on the CPU, will be overwritten to the total
     *      number of non-zeros.
     *
     * \throws std::runtime_error if the grid is not Local Polynomial or if the currently set acceleration mode
     *      is not compatible, i.e., TasGrid::accel_none or TasGrid::accel_cpu_blas.
     *      Also, if CUDA has not been enabled at compile time.
     *
     * \b Note: will not work for LocalPolynomial grids with order bigger than 2.
     */
    template<typename FloatType>
    void evaluateSparseHierarchicalFunctionsGPU(const FloatType gpu_x[], int cpu_num_x, int* &gpu_pntr, int* &gpu_indx, FloatType* &gpu_vals, int &num_nz) const;

    //! \brief Signature compatible with TasDREAM::DreamPDF, TasDREAM::DreamModel amd TasDREAM::DreamMergedLikelyModel.
    using EvaluateCallable = std::function<void(std::vector<double> const&, std::vector<double>&)>;

    /*!
     * \brief Custom conversion to a callable method using the TasDREAM::DreamPDF signature.
     *
     * This conversion allows an instance of TasmanianSparseGrid to be passed as input
     * to any method that expects TasDREAM::DreamPDF, TasDREAM::DreamModel or TasDREAM::DreamMergedLikelyModel.
     */
    operator EvaluateCallable() const{
        return [&](std::vector<double> const &x, std::vector<double> &y)->void{ evaluateBatch(x, y); };
    }

    //! \brief Signature of the domain inside lambda, identical to TasDREAM::DreamDomain.
    using DomainInsideSignature = std::function<bool(std::vector<double> const &)>; // trick for Doxygen formatting purposes

    /*!
     * \brief Returns a lambda object that satisfies the TasDREAM::DreamDomain signature.
     *
     * This method allows for the domain currently set in the grid to be passed to the
     * Tasmanian DREAM sampling templates in place of the \b inside() object.
     */
     DomainInsideSignature getDomainInside() const{
        auto rule = getRule();
        if ((rule == TasGrid::rule_gausshermite) || (rule == TasGrid::rule_gausshermiteodd)){ // unbounded domain
            return [](std::vector<double> const &)->bool{ return true; };
        }else if ((rule == TasGrid::rule_gausslaguerre) || (rule == TasGrid::rule_gausslaguerreodd)){ // bounded from below
            if (domain_transform_a.empty()){
                return [](std::vector<double> const &x)->bool{
                    for(auto const &v : x) if (v < 0.0) return false;
                    return true;
                };
            }else{
                size_t dims = (size_t) getNumDimensions();
                return [=](std::vector<double> const &x)->bool{
                    for(size_t i=0; i<dims; i++) if (x[i] < domain_transform_a[i]) return false;
                    return true;
                };
            }
        }else{
            if (domain_transform_a.empty()){
                if (isFourier()){
                    return [](std::vector<double> const &x)->bool{
                        for(auto const &v : x) if ((v < 0.0) || (v > 1.0)) return false;
                        return true;
                    };
                }else{
                    return [](std::vector<double> const &x)->bool{
                        for(auto const &v : x) if ((v < -1.0) || (v > 1.0)) return false;
                        return true;
                    };
                }
            }else{
                size_t dims = (size_t) getNumDimensions();
                return [=](std::vector<double> const &x)->bool{
                    for(size_t i=0; i<dims; i++)
                        if ((x[i] < domain_transform_a[i]) || (x[i] > domain_transform_b[i])) return false;
                    return true;
                };
            }
        }
    }

    /*!
     * \brief Removes all points from the grid that have relative surplus less than the \b tolerance.
     *
     * The \b output and \b scale_correction variables have the same effects as in the call to setSurplusRefinement().
     * The purpose of this call is to reduce the number of points and thus the memory footprint of the grid.
     * As such, points will be removed with no regard of preserving lower completeness or connectivity of the hierarchical graphs;
     * therefore, it is possible that the grid no longer has a valid state with respect to the update and refinement algorithms.
     * Calling loadNeededPoints() or any refinement or construction method after the removal of points may lead to undefined behavior;
     * get, evaluate and file I/O methods are safe to call.
     *
     * \param tolerance the cut-off tolerance for the point removal.
     * \param output is the output to use for the tolerance test, can be set to -1 to use all outputs.
     * \param scale_correction is the same as in the call to setSurplusRefinement().
     *
     * \throws std::runtime_error if the grid is not Local Polynomial.
     */
    void removePointsByHierarchicalCoefficient(double tolerance, int output = -1, const double *scale_correction = nullptr);

    #ifndef __TASMANIAN_DOXYGEN_SKIP_INTERNAL
    /*!
     * \internal
     * \brief Count the number of non-zeros in a call to evaluateSparseHierarchicalFunctions().
     *
     * Used by the C/Python/Fortran interfaces, counts the number of non-zeros so memory
     * can be pre-allocated by the corresponding language before a call to evaluateSparseHierarchicalFunctionsStatic().
     * \endinternal
     */
    int evaluateSparseHierarchicalFunctionsGetNZ(const double x[], int num_x) const;
    /*!
     * \internal
     * \brief Assumes pre-allocated arrays, otherwise identical to evaluateSparseHierarchicalFunctions()
     *
     * See evaluateSparseHierarchicalFunctionsGetNZ().
     * \endinternal
     */
    void evaluateSparseHierarchicalFunctionsStatic(const double x[], int num_x, int pntr[], int indx[], double vals[]) const;
    /*!
     * \internal
     * \brief Return a reference to the multi-indexes of the loaded points.
     *
     * Testing and debugging purposes mostly.
     * \endinternal
     */
    const int* getPointsIndexes() const;
    /*!
     * \internal
     * \brief Return a reference to the multi-indexes of the needed points.
     *
     * Testing and debugging purposes mostly.
     * \endinternal
     */
    const int* getNeededIndexes() const;

    /*!
     * \internal
     * \brief Return a reference to the internal grid.
     *
     * Casts the internal unique_ptr to \b T and returns the result.
     * \endinternal
     */
    template<class T> inline T* get(){ return (T*) base.get(); }
    /*!
     * \internal
     * \brief Overload using const.
     *
     * Same as get(), but the method is const and returns a const reference.
     * \endinternal
     */
    template<class T> inline T const* get() const{ return (T*) base.get(); }
    #endif // __TASMANIAN_DOXYGEN_SKIP_INTERNAL

protected:
    #ifndef __TASMANIAN_DOXYGEN_SKIP_INTERNAL
    /*!
     * \internal
     * \brief Reset the grid to empty.
     *
     * Resets the grid, all cache data structures, and the acceleration mode.
     * \endinternal
     */
    void clear();

    /*!
     * \internal
     * \brief Maps canonical points to the transformed equivalent.
     *
     * Given an array of canonical points \b x with size \b num_points and \b num_dimensions for given \b rule,
     * convert them to transformed points applying only the linear transform.
     * \endinternal
     */
    void mapCanonicalToTransformed(int num_dimensions, int num_points, TypeOneDRule rule, double x[]) const;
    /*!
     * \internal
     * \brief Apply the inverse map from transformed to canonical points.
     *
     * Similar to mapCanonicalToTransformed(), uses the inverse transform.
     * \endinternal
     */
    template<typename FloatType> void mapTransformedToCanonical(int num_dimensions, int num_points, TypeOneDRule rule, FloatType x[]) const;
    /*!
     * \internal
     * \brief Returns the quadrature scale factor associated with the linear transform.
     *
     * Uses the \b num_dimensions and \b rule, as well as private members \b alpha and \b beta.
     * \endinternal
     */
    double getQuadratureScale(int num_dimensions, TypeOneDRule rule) const;

    /*!
     * \internal
     * \brief Applies the non-linear transformation to the points.
     *
     * Using the set conformal transform (private variables), applies the non-linear transform
     * to the points \b x with given number of dimensions and number of points.
     * \endinternal
     */
    void mapConformalCanonicalToTransformed(int num_dimensions, int num_points, double x[]) const;
    /*!
     * \internal
     * \brief Applies the inverse non-linear transformation to the points.
     *
     * Using the set conformal transform (private variables), applies the inverse non-linear transform.
     * \endinternal
     */
    template<typename FloatType> void mapConformalTransformedToCanonical(int num_dimensions, int num_points, Data2D<FloatType> &x) const;
    /*!
     * \internal
     * \brief Computes the quadrature weight correction for the conformal map.
     *
     * Using the set conformal transform (private variables), compute the quadrature weights correction.
     * \endinternal
     */
    void mapConformalWeights(int num_dimensions, int num_points, double weights[]) const;

    /*!
     * \internal
     * \brief Returns a raw-array with the canonical points, combines both transformations.
     *
     * If no transforms have been set, then return an alias to \b x.
     * Otherwise, \b x_temp is resized, filled with the transformed points and an alias is returned.
     * Canonical points are given in \b x, and \b num_x holds the number of points.
     * This method applies both linear and non-linear transforms.
     * \endinternal
     */
    template<typename FloatType> const FloatType* formCanonicalPoints(const FloatType *x, Data2D<FloatType> &x_temp, int num_x) const;
    #ifdef Tasmanian_ENABLE_CUDA
    /*!
     * \internal
     * \brief Returns a CUDA raw-array with the canonical points, linear transform only.
     *
     * Similar to formCanonicalPoints() except the input and output arrays/vectors are
     * allocated on the current CUDA device. Works with single and double precision \b T.
     * \endinternal
     */
    template<typename T>
    const T* formCanonicalPointsGPU(const T *gpu_x, int num_x, CudaVector<T> &gpu_x_temp) const;
    #endif
    /*!
     * \internal
     * \brief Applies both linear and non-linear transformation to the canonical points.
     *
     * The raw-array \b x holds \b num_points and will be overwritten with the result
     * of the application of both transforms.
     * \endinternal
     */
    void formTransformedPoints(int num_points, double x[]) const; // when calling get***Points()

    /*!
     * \internal
     * \brief Write the grid to a stream using ASCII format.
     *
     * Uses an open stream, writes the grid.
     * \endinternal
     */
    void writeAscii(std::ostream &ofs) const;
    /*!
     * \internal
     * \brief Read the grid from a stream using ASCII format.
     *
     * Uses an open stream, read the grid.
     * \throws std::runtime_error if Tasmanian detects a problem with the file format.
     * \endinternal
     */
    void readAscii(std::istream &ifs);
    /*!
     * \internal
     * \brief Write the grid to a stream using binary format.
     *
     * Uses an open stream, writes the grid.
     * \endinternal
     */
    void writeBinary(std::ostream &ofs) const;
    /*!
     * \internal
     * \brief Read the grid from a stream using binary format.
     *
     * Uses an open stream, read the grid.
     * \throws std::runtime_error if Tasmanian detects a problem with the file format.
     * \endinternal
     */
    void readBinary(std::istream &ifs);
    #endif // __TASMANIAN_DOXYGEN_SKIP_INTERNAL

private:
    std::unique_ptr<BaseCanonicalGrid> base;

    std::vector<double> domain_transform_a, domain_transform_b;
    std::vector<int> conformal_asin_power;
    std::vector<int> llimits;

    size_t num_threads;
    TypeAcceleration acceleration;
    int gpu_id;

    bool usingDynamicConstruction;

    #ifdef Tasmanian_ENABLE_CUDA
    mutable std::unique_ptr<CudaEngine> engine;
    mutable std::unique_ptr<AccelerationDomainTransform> acc_domain;
    #endif
};

/*!
 * \ingroup TasmanianSG
 * \brief Returns an empty sparse grid.
 *
 * Usage:
 * \code
 * auto grid = TasGrid::makeEmpty(); // equivalent to TasGrid::TasmanianSparseGrid grid;
 * grid = TasGrid::makeEmpty();      // equivalent to grid.clear();
 * \endcode
 * Useful for some MPI calls where some MPI ranks must pass a dummy empty grid.
 */
inline TasmanianSparseGrid makeEmpty(){ return TasmanianSparseGrid(); }

/*!
 * \ingroup TasmanianSG
 * \brief Factory method, creates a new grid and calls TasmanianSparseGrid::makeGlobalGrid().
 *
 * Allows for one-line initialization, i.e.,
 * \code
 * auto grid = TasGrid::makeGlobalGrid(...);
 * \endcode
 * as opposed to
 * \code
 * TasGrid::TasmanianSparseGrid grid;
 * grid.makeGlobalGrid(...);
 * \endcode
 */
inline TasmanianSparseGrid
makeGlobalGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule,
               std::vector<int> const &anisotropic_weights = std::vector<int>(), double alpha = 0.0, double beta = 0.0,
               const char* custom_filename = nullptr, std::vector<int> const &level_limits = std::vector<int>()){
    TasmanianSparseGrid grid;
    grid.makeGlobalGrid(dimensions, outputs, depth, type, rule, anisotropic_weights, alpha, beta, custom_filename, level_limits);
    return grid;
}

/*!
 * \ingroup TasmanianSG
 * \brief Factory method, creates a new grid and calls TasmanianSparseGrid::makeSequenceGrid().
 *
 * Allows for one-line initialization, see TasGrid::makeGlobalGrid().
 */
inline TasmanianSparseGrid
makeSequenceGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule,
                 std::vector<int> const &anisotropic_weights = std::vector<int>(), std::vector<int> const &level_limits = std::vector<int>()){
    TasmanianSparseGrid grid;
    grid.makeSequenceGrid(dimensions, outputs, depth, type, rule, anisotropic_weights, level_limits);
    return grid;
}

/*!
 * \ingroup TasmanianSG
 * \brief Factory method, creates a new grid and calls TasmanianSparseGrid::makeLocalPolynomialGrid().
 *
 * Allows for one-line initialization, see TasGrid::makeGlobalGrid().
 */
inline TasmanianSparseGrid
makeLocalPolynomialGrid(int dimensions, int outputs, int depth, int order = 1, TypeOneDRule rule = rule_localp, std::vector<int> const &level_limits = std::vector<int>()){
    TasmanianSparseGrid grid;
    grid.makeLocalPolynomialGrid(dimensions, outputs, depth, order, rule, level_limits);
    return grid;
}

/*!
 * \ingroup TasmanianSG
 * \brief Factory method, creates a new grid and calls TasmanianSparseGrid::makeWaveletGrid().
 *
 * Allows for one-line initialization, see TasGrid::makeGlobalGrid().
 */
inline TasmanianSparseGrid
makeWaveletGrid(int dimensions, int outputs, int depth, int order = 1, std::vector<int> const &level_limits = std::vector<int>()){
    TasmanianSparseGrid grid;
    grid.makeWaveletGrid(dimensions, outputs, depth, order, level_limits);
    return grid;
}

/*!
 * \ingroup TasmanianSG
 * \brief Factory method, creates a new grid and calls TasmanianSparseGrid::makeFourierGrid().
 *
 * Allows for one-line initialization, see TasGrid::makeGlobalGrid().
 */
inline TasmanianSparseGrid
makeFourierGrid(int dimensions, int outputs, int depth, TypeDepth type,
                std::vector<int> const &anisotropic_weights = std::vector<int>(), std::vector<int> const &level_limits = std::vector<int>()){
    TasmanianSparseGrid grid;
    grid.makeFourierGrid(dimensions, outputs, depth, type, anisotropic_weights, level_limits);
    return grid;
}

/*!
 * \ingroup TasmanianSG
 * \brief Factory method, creates a new grid and calls TasmanianSparseGrid::read().
 *
 * Allows for one-line initialization, makes a new grid and reads from a file.
 * \param filename same as TasmanianSparseGrid::read().
 * \returns a new grid that is read from the file.
 */
inline TasmanianSparseGrid readGrid(const char *filename){
    TasmanianSparseGrid grid;
    grid.read(filename);
    return grid;
}

/*!
 * \ingroup TasmanianSG
 * \brief Overload using std::string.
 *
 * Same as readGrid() but the filename is given as a string.
 */
inline TasmanianSparseGrid readGrid(std::string const &filename){ return readGrid(filename.c_str()); }

/*!
 * \ingroup TasmanianSG
 * \brief Returns a grid that is a copy of the source.
 *
 * Creates a new grid and calls TasmanianSparseGrid::copyGrid() from the source.
 * \param source is the grid to copy from.
 * \param outputs_begin same as TasmanianSparseGrid::copyGrid().
 * \param outputs_end same as TasmanianSparseGrid::copyGrid().
 *
 * \returns a new grid that is a copy of the source.
 */
inline TasmanianSparseGrid copyGrid(TasmanianSparseGrid const &source, int outputs_begin = 0, int outputs_end = -1){
    TasmanianSparseGrid grid;
    grid.copyGrid(source, outputs_begin, outputs_end);
    return grid;
}

}

#endif
