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

#ifndef __TASMANIAN_SPARSE_GRID_CPP
#define __TASMANIAN_SPARSE_GRID_CPP

#include "TasmanianSparseGrid.hpp"

#include "tsgUtils.hpp"

namespace TasGrid{

const char* TasmanianSparseGrid::getVersion(){ return TASMANIAN_VERSION_STRING; }
const char* TasmanianSparseGrid::getLicense(){ return TASMANIAN_LICENSE; }
const char* TasmanianSparseGrid::getGitCommitHash(){ return TASMANIAN_GIT_COMMIT_HASH; }
const char* TasmanianSparseGrid::getCmakeCxxFlags(){ return TASMANIAN_CXX_FLAGS; }
int TasmanianSparseGrid::getVersionMajor(){ return TASMANIAN_VERSION_MAJOR; }
int TasmanianSparseGrid::getVersionMinor(){ return TASMANIAN_VERSION_MINOR; }
bool TasmanianSparseGrid::isOpenMPEnabled(){
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif // _OPENMP
}
bool TasmanianSparseGrid::isCudaEnabled(){
    #ifdef Tasmanian_ENABLE_CUDA
    return true;
    #else
    return false;
    #endif
}
bool TasmanianSparseGrid::isHipEnabled(){
    #ifdef Tasmanian_ENABLE_HIP
    return true;
    #else
    return false;
    #endif
}
bool TasmanianSparseGrid::isDpcppEnabled(){
    #ifdef Tasmanian_ENABLE_DPCPP
    return true;
    #else
    return false;
    #endif
}

TasmanianSparseGrid::TasmanianSparseGrid() : acceleration(std::make_unique<AccelerationContext>()), using_dynamic_construction(false){}

TasmanianSparseGrid::TasmanianSparseGrid(const TasmanianSparseGrid &source) :
        acceleration(std::make_unique<AccelerationContext>()), using_dynamic_construction(false){
    copyGrid(&source);
}

TasmanianSparseGrid& TasmanianSparseGrid::operator=(TasmanianSparseGrid const &source){
    copyGrid(&source);
    return *this;
}

void TasmanianSparseGrid::clear(){
    base = std::monostate{};
    domain_transform_a = std::vector<double>();
    domain_transform_b = std::vector<double>();
    conformal_asin_power = std::vector<int>();
    llimits = std::vector<int>();
    using_dynamic_construction = false;
#ifdef Tasmanian_ENABLE_GPU
    acc_domain.clear();
#endif // Tasmanian_ENABLE_GPU
}

void TasmanianSparseGrid::write(const char *filename, bool binary) const{
    std::ofstream ofs;
    if (binary == mode_binary){
        ofs.open(filename, std::ios::out | std::ios::binary);
    }else{
        ofs.open(filename);
    }
    if (!ofs.good()) throw std::runtime_error(std::string("ERROR: occurred when trying to write to file: ") + filename);
    write(ofs, binary);
    ofs.close();
}
void TasmanianSparseGrid::read(const char *filename){
    std::ifstream ifs;
    char TSG[3];
    bool binary_format = mode_ascii;
    ifs.open(filename, std::ios::in | std::ios::binary);
    if (!ifs.good()) throw std::runtime_error(std::string("ERROR: occurred when trying to open file: ") + filename);
    ifs.read(TSG, 3 * sizeof(char));
    if ((TSG[0] == 'T') && (TSG[1] == 'S') && (TSG[2] == 'G')){
        binary_format = mode_binary;
    }
    ifs.close();
    if (binary_format == mode_binary){
        ifs.open(filename, std::ios::in | std::ios::binary);
    }else{
        ifs.open(filename);
    }
    if (!ifs.good()) throw std::runtime_error(std::string("ERROR: occurred when trying to open file: ") + filename);
    read(ifs, binary_format);
    ifs.close();
}

void TasmanianSparseGrid::write(std::ostream &ofs, bool binary) const{
    if (binary == mode_binary){
        writeBinary(ofs);
    }else{
        writeAscii(ofs);
    }
}
void TasmanianSparseGrid::read(std::istream &ifs, bool binary){
    if (binary == mode_binary){
        readBinary(ifs);
    }else{
        readAscii(ifs);
    }
}

void TasmanianSparseGrid::makeGlobalGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule, const int *anisotropic_weights, double alpha, double beta, const char* custom_filename, const int *level_limits){
    makeGlobalGrid(dimensions, outputs, depth, type, rule,
                   Utils::copyArray(anisotropic_weights, (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions),
                   alpha, beta, custom_filename, Utils::copyArray(level_limits, dimensions));
}
void TasmanianSparseGrid::makeGlobalGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule, const std::vector<int> &anisotropic_weights, double alpha, double beta, const char* custom_filename, const std::vector<int> &level_limits){
    if (dimensions < 1) throw std::invalid_argument("ERROR: makeGlobalGrid() requires positive dimensions");
    if (outputs < 0) throw std::invalid_argument("ERROR: makeGlobalGrid() requires non-negative outputs");
    if (depth < 0) throw std::invalid_argument("ERROR: makeGlobalGrid() requires non-negative depth");
    if (!OneDimensionalMeta::isGlobal(rule)) throw std::invalid_argument("ERROR: makeGlobalGrid() requires a global rule");
    if ((rule == rule_customtabulated) && (custom_filename == 0)) throw std::invalid_argument("ERROR: makeGlobalGrid() with custom tabulated rule requires a filename");
    size_t expected_aw_size = (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions;
    if ((!anisotropic_weights.empty()) && (anisotropic_weights.size() != expected_aw_size)) throw std::invalid_argument("ERROR: makeGlobalGrid() requires anisotropic_weights with either 0 or dimenions entries");
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dimensions)) throw std::invalid_argument("ERROR: makeGlobalGrid() requires level_limits with either 0 or dimensions entries");
    clear();
    llimits = level_limits;
    base = GridGlobal(acceleration.get(), dimensions, outputs, depth, type, rule, anisotropic_weights, alpha, beta, custom_filename, llimits);
}
void TasmanianSparseGrid::makeGlobalGrid(int dimensions, int outputs, int depth, TypeDepth type, CustomTabulated &&crule,
                                         const int *anisotropic_weights, const int *level_limits){
    makeGlobalGrid(dimensions, outputs, depth, type, std::move(crule),
                   Utils::copyArray(anisotropic_weights, (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions),
                   Utils::copyArray(level_limits, dimensions));
}
void TasmanianSparseGrid::makeGlobalGrid(int dimensions, int outputs, int depth, TypeDepth type, CustomTabulated &&rule, std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits){
    if (dimensions < 1) throw std::invalid_argument("ERROR: makeGlobalGrid() requires positive dimensions");
    if (outputs < 0) throw std::invalid_argument("ERROR: makeGlobalGrid() requires non-negative outputs");
    if (depth < 0) throw std::invalid_argument("ERROR: makeGlobalGrid() requires non-negative depth");
    size_t expected_aw_size = (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions;
    if ((!anisotropic_weights.empty()) && (anisotropic_weights.size() != expected_aw_size)) throw std::invalid_argument("ERROR: makeGlobalGrid() requires anisotropic_weights with either 0 or dimenions entries");
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dimensions)) throw std::invalid_argument("ERROR: makeGlobalGrid() requires level_limits with either 0 or dimensions entries");
    clear();
    llimits = level_limits;
    base = GridGlobal(acceleration.get(), dimensions, outputs, depth, type, std::move(rule), anisotropic_weights, llimits);
}


void TasmanianSparseGrid::makeSequenceGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule, const int *anisotropic_weights, const int *level_limits){
    makeSequenceGrid(dimensions, outputs, depth, type, rule,
                     Utils::copyArray(anisotropic_weights, (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions),
                     Utils::copyArray(level_limits, dimensions));
}
void TasmanianSparseGrid::makeSequenceGrid(int dimensions, int outputs, int depth, TypeDepth type, TypeOneDRule rule, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    if (dimensions < 1) throw std::invalid_argument("ERROR: makeSequenceGrid() requires positive dimensions");
    if (outputs < 0) throw std::invalid_argument("ERROR: makeSequenceGrid() requires non-negative outputs");
    if (depth < 0) throw std::invalid_argument("ERROR: makeSequenceGrid() requires non-negative depth");
    if (!OneDimensionalMeta::isSequence(rule)){
        std::string message = "ERROR: makeSequenceGrid() is called with rule: " + IO::getRuleString(rule) + ", which is not a sequence rule";
        throw std::invalid_argument(message);
    }
    size_t expected_aw_size = (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions;
    if ((!anisotropic_weights.empty()) && (anisotropic_weights.size() != expected_aw_size)) throw std::invalid_argument("ERROR: makeSequenceGrid() requires anisotropic_weights with either 0 or dimensions entries");
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dimensions)) throw std::invalid_argument("ERROR: makeSequenceGrid() requires level_limits with either 0 or dimensions entries");
    clear();
    llimits = level_limits;
    if (outputs == 0)
        base.emplace<GridSequence>(acceleration.get(), dimensions, depth, type, rule, anisotropic_weights, llimits);
    else
        base.emplace<GridSequence>(acceleration.get(), dimensions, outputs, depth, type, rule, anisotropic_weights, llimits);
}

void TasmanianSparseGrid::makeLocalPolynomialGrid(int dimensions, int outputs, int depth, int order, TypeOneDRule rule, const int *level_limits){
    makeLocalPolynomialGrid(dimensions, outputs, depth, order, rule, Utils::copyArray(level_limits, dimensions));
}
void TasmanianSparseGrid::makeLocalPolynomialGrid(int dimensions, int outputs, int depth, int order, TypeOneDRule rule, const std::vector<int> &level_limits){
    if (dimensions < 1) throw std::invalid_argument("ERROR: makeLocalPolynomialGrid() requires positive dimensions");
    if (outputs < 0) throw std::invalid_argument("ERROR: makeLocalPolynomialGrid() requires non-negative outputs");
    if (depth < 0) throw std::invalid_argument("ERROR: makeLocalPolynomialGrid() requires non-negative depth");
    if (order < -1){
        std::string message = "ERROR: makeLocalPolynomialGrid() is called with order: " + std::to_string(order) + ", but the order cannot be less than -1.";
        throw std::invalid_argument(message);
    }
    if (!OneDimensionalMeta::isLocalPolynomial(rule)){
        std::string message = "ERROR: makeLocalPolynomialGrid() is called with rule: " + IO::getRuleString(rule) + ", which is not a local polynomial rule";
        throw std::invalid_argument(message);
    }
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dimensions)) throw std::invalid_argument("ERROR: makeLocalPolynomialGrid() requires level_limits with either 0 or dimensions entries");
    clear();
    llimits = level_limits;
    base.emplace<GridLocalPolynomial>(acceleration.get(), dimensions, outputs, depth, order, rule, llimits);
}

void TasmanianSparseGrid::makeWaveletGrid(int dimensions, int outputs, int depth, int order, const int *level_limits){
    makeWaveletGrid(dimensions, outputs, depth, order, Utils::copyArray(level_limits, dimensions));
}
void TasmanianSparseGrid::makeWaveletGrid(int dimensions, int outputs, int depth, int order, const std::vector<int> &level_limits){
    if (dimensions < 1) throw std::invalid_argument("ERROR: makeWaveletGrid() requires positive dimensions");
    if (outputs < 0) throw std::invalid_argument("ERROR: makeWaveletGrid() requires non-negative outputs");
    if (depth < 0) throw std::invalid_argument("ERROR: makeWaveletGrid() requires non-negative depth");
    if ((order != 1) && (order != 3)){
        std::string message = "ERROR: makeWaveletGrid() is called with order: " + std::to_string(order) + ", but wavelets are implemented only for orders 1 and 3.";
        throw std::invalid_argument(message);
    }
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dimensions)) throw std::invalid_argument("ERROR: makeWaveletGrid() requires level_limits with either 0 or dimensions entries");
    clear();
    llimits = level_limits;
    base.emplace<GridWavelet>(acceleration.get(), dimensions, outputs, depth, order, llimits);
}

void TasmanianSparseGrid::makeFourierGrid(int dimensions, int outputs, int depth, TypeDepth type, const int* anisotropic_weights, const int* level_limits){
    makeFourierGrid(dimensions, outputs, depth, type, Utils::copyArray(anisotropic_weights, (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions),
                    Utils::copyArray(level_limits, dimensions));
}
void TasmanianSparseGrid::makeFourierGrid(int dimensions, int outputs, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    if (dimensions < 1) throw std::invalid_argument("ERROR: makeFourierGrid() requires positive dimensions");
    if (outputs < 0) throw std::invalid_argument("ERROR: makeFourierGrid() requires non-negative outputs");
    if (depth < 0) throw std::invalid_argument("ERROR: makeFourierGrid() requires non-negative depth");
    size_t expected_aw_size = (OneDimensionalMeta::isTypeCurved(type)) ? 2*dimensions : dimensions;
    if ((!anisotropic_weights.empty()) && (anisotropic_weights.size() != expected_aw_size)) throw std::invalid_argument("ERROR: makeFourierGrid() requires anisotropic_weights with either 0 or dimensions entries");
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dimensions)) throw std::invalid_argument("ERROR: makeFourierGrid() requires level_limits with either 0 or dimensions entries");
    clear();
    llimits = level_limits;
    base.emplace<GridFourier>(acceleration.get(), dimensions, outputs, depth, type, anisotropic_weights, llimits);
}

void TasmanianSparseGrid::copyGrid(const TasmanianSparseGrid *source, int outputs_begin, int outputs_end){
    if (outputs_end == -1) outputs_end = source->getNumOutputs();
    clear();

    std::visit([&](auto const &g) -> void {
        using T = std::decay_t<decltype(g)>;
        if constexpr (std::is_same_v<T, std::monostate>)
            base.emplace<std::monostate>();
        else
            base.emplace<T>(acceleration.get(), std::addressof(g), outputs_begin, outputs_end);
    }, source->base);

    if (source->domain_transform_a.size() > 0){
        setDomainTransform(source->domain_transform_a, source->domain_transform_b);
    }
    conformal_asin_power = source->conformal_asin_power;
    llimits = source->llimits;
    using_dynamic_construction = source->using_dynamic_construction;
}

void TasmanianSparseGrid::updateGlobalGrid(int depth, TypeDepth type, const int *anisotropic_weights, const int *level_limits){
    if (empty()) throw std::runtime_error("ERROR: updateGlobalGrid() called, but the grid is empty");
    updateGlobalGrid(depth, type, Utils::copyArray(anisotropic_weights, (OneDimensionalMeta::isTypeCurved(type)) ? 2*getNumDimensions() : getNumDimensions()),
                     Utils::copyArray(level_limits, getNumDimensions()));
}
void TasmanianSparseGrid::updateGlobalGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    updateGrid(depth, type, anisotropic_weights, level_limits);
}

void TasmanianSparseGrid::updateSequenceGrid(int depth, TypeDepth type, const int *anisotropic_weights, const int *level_limits){
    if (empty()) throw std::runtime_error("ERROR: updateSequenceGrid called, but the grid is empty");
    updateSequenceGrid(depth, type, Utils::copyArray(anisotropic_weights, (OneDimensionalMeta::isTypeCurved(type)) ? 2*getNumDimensions() : getNumDimensions()),
                       Utils::copyArray(level_limits, getNumDimensions()));
}
void TasmanianSparseGrid::updateSequenceGrid(int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    updateGrid(depth, type, anisotropic_weights, level_limits);
}

void TasmanianSparseGrid::updateFourierGrid(int depth, TypeDepth type, const int *anisotropic_weights, const int *level_limits){
    updateGrid(depth, type, anisotropic_weights, level_limits);
}
void TasmanianSparseGrid::updateFourierGrid(int depth, TypeDepth type, std::vector<int> const &anisotropic_weights,
                                            std::vector<int> const &level_limits){
    updateGrid(depth, type, anisotropic_weights, level_limits);
}
void TasmanianSparseGrid::updateGrid(int depth, TypeDepth type, const int *anisotropic_weights, const int *level_limits){
    if (empty()) throw std::runtime_error("ERROR: updateGrid() called, but the grid is empty");
    updateGrid(depth, type,
               Utils::copyArray(anisotropic_weights, (OneDimensionalMeta::isTypeCurved(type)) ? 2*getNumDimensions() : getNumDimensions()),
               Utils::copyArray(level_limits, getNumDimensions()));
}
void TasmanianSparseGrid::updateGrid(int depth, TypeDepth type, std::vector<int> const &anisotropic_weights, std::vector<int> const &level_limits){
    if (empty()) throw std::runtime_error("ERROR: updateGrid() called, but the grid is empty");
    int dims = getNumDimensions();
    if (depth < 0) throw std::invalid_argument("ERROR: cannot update with a negative depth");
    size_t expected_aw_size = (OneDimensionalMeta::isTypeCurved(type)) ? 2*dims : dims;
    if (not anisotropic_weights.empty() and anisotropic_weights.size() != expected_aw_size) throw std::invalid_argument("ERROR: in updateGrid() anisotropic_weights must be either empty or has size equal to dimenions or twice dimenions based on the type of the update.");
    if (not level_limits.empty() and level_limits.size() != (size_t) dims) throw std::invalid_argument("ERROR: in updateGrid() level_limits must be either empty or must have size equal to the number of dimensions");
    if (not level_limits.empty()) llimits = level_limits; // if level_limits is empty, use the existing llimits (if any)
    if (isGlobal()){
        get<GridGlobal>()->updateGrid(depth, type, anisotropic_weights, llimits);
    }else if (isSequence()){
        get<GridSequence>()->updateGrid(depth, type, anisotropic_weights, llimits);
    }else if (isFourier()){
        get<GridFourier>()->updateGrid(depth, type, anisotropic_weights, llimits);
    }else{
        throw std::runtime_error("ERROR: an update operation can be performed only on Global, Sequence and Fourier grids.");
    }
}

TypeOneDRule TasmanianSparseGrid::getRule() const{
    return std::visit([](auto const &g) -> TypeOneDRule{
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
            return g.getRule();
        } else
            return rule_none;
    }, base);
}

const char* TasmanianSparseGrid::getCustomRuleDescription() const{ return (isGlobal()) ? get<GridGlobal>()->getCustomRuleDescription() : ""; }

void TasmanianSparseGrid::getLoadedPoints(double *x) const{
    int const num = std::visit([&](auto const &g) -> int{
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
            g.getLoadedPoints(x);
            return g.getNumLoaded();
        } else
            return 0;
    }, base);
    formTransformedPoints(num, x);
}
void TasmanianSparseGrid::getNeededPoints(double *x) const{
    int const num = std::visit([&](auto const &g) -> int{
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
            g.getNeededPoints(x);
            return g.getNumNeeded();
        } else
            return 0;
    }, base);
    formTransformedPoints(num, x);
}
void TasmanianSparseGrid::getPoints(double *x) const{
    int const num = std::visit([&](auto const &g) -> int{
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
            g.getPoints(x);
            return g.getNumPoints();
        } else
            return 0;
    }, base);
    formTransformedPoints(num, x);
}

void TasmanianSparseGrid::getQuadratureWeights(double *weights) const{
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.getQuadratureWeights(weights);
    }, base);
    int const num_dimensions = getNumDimensions();
    int const num_points     = getNumPoints();
    mapConformalWeights(num_dimensions, num_points, weights);
    if (domain_transform_a.size() != 0){
        double scale = getQuadratureScale(num_dimensions, getRule());
        #pragma omp parallel for schedule(static)
        for(int i=0; i<num_points; i++) weights[i] *= scale;
    }
}
std::vector<double> TasmanianSparseGrid::getInterpolationWeights(std::vector<double> const &x) const{
    std::vector<double> w;
    getInterpolationWeights(x, w);
    return w;
}
void TasmanianSparseGrid::getInterpolationWeights(const std::vector<double> &x, std::vector<double> &weights) const{
    int const num_dimensions = getNumDimensions();
    if (x.size() != (size_t) num_dimensions) throw std::runtime_error("ERROR: getInterpolationWeights() incorrect size of x, must be same as getNumDimensions()");
    weights.resize((size_t) getNumPoints());
    getInterpolationWeights(x.data(), weights.data());
}
void TasmanianSparseGrid::getInterpolationWeights(const double x[], double weights[]) const{
    Data2D<double> x_tmp;
    auto const &pnts = formCanonicalPoints(x, x_tmp, 1);
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.getInterpolationWeights(pnts, weights);
    }, base);
}

std::vector<double> TasmanianSparseGrid::getDifferentiationWeights(std::vector<double> const &x) const{
    std::vector<double> w;
    getDifferentiationWeights(x, w);
    return w;
}
void TasmanianSparseGrid::getDifferentiationWeights(const std::vector<double> &x, std::vector<double> &weights) const{
    int const num_dimensions = getNumDimensions();
    if (x.size() != (size_t) num_dimensions)
        throw std::runtime_error("ERROR: getDifferentiationWeights() incorrect size of x, must be same as getNumDimensions()");
    weights.resize((size_t) getNumPoints() * (size_t) getNumDimensions());
    getDifferentiationWeights(x.data(), weights.data());
}
void TasmanianSparseGrid::getDifferentiationWeights(const double x[], double weights[]) const{
    // See differentiate() for how the domain transforms are taken into consideration.
    Data2D<double> x_tmp;
    auto const &pnts = formCanonicalPoints(x, x_tmp, 1);
    // Jacobian of f(.) at g(x).
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.getDifferentiationWeights(pnts, weights);
    }, base);
    // Jacobian of f(g(.)) at x.
    if (not domain_transform_a.empty()) {
        int num_dimensions = getNumDimensions();
        int num_points = getNumPoints();
        std::vector<double> jacobian_g_diag = diffCanonicalTransform<double>();
        #pragma omp parallel for
        for(int i=0; i<num_points; i++)
            for(int j=0; j<num_dimensions; j++)
                weights[i * num_dimensions + j] = weights[i * num_dimensions + j] * jacobian_g_diag[j];
    }
}

void TasmanianSparseGrid::loadNeededValues(const double *vals){
    if (empty()) throw std::runtime_error("Cannot load model values into an empty grid!");
    std::visit([&](auto &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.loadNeededValues(vals);
    }, base);
}
void TasmanianSparseGrid::loadNeededValues(const std::vector<double> &vals){
    size_t nump = (size_t) getNumNeeded();
    if (nump == 0) nump = (size_t) getNumPoints();
    nump *= (size_t) getNumOutputs();
    if (vals.size() != nump) throw std::runtime_error("ERROR: loadNeededPoints() given the wrong number of inputs, should be getNumNeeded() * getNumOutputs() or (if getNumNeeded() == 0) getNumPoints() * getNumOutputs()");
    loadNeededValues(vals.data());
}
const double* TasmanianSparseGrid::getLoadedValues() const{
    return std::visit([](auto const &g) -> double const * {
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
            return g.getLoadedValues();
        } else {
            return nullptr;
        }
    }, base);
}

void TasmanianSparseGrid::evaluate(const double x[], double y[]) const{
    Data2D<double> x_tmp;
    auto const &pnts = formCanonicalPoints(x, x_tmp, 1);
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.evaluate(pnts, y);
    }, base);
}

template<typename FloatType> void TasmanianSparseGrid::evaluateBatch(const std::vector<FloatType> &x, std::vector<FloatType> &y) const{
    if (empty()) return; // should probably throw
    int num_outputs = getNumOutputs();
    size_t num_x = x.size() / getNumDimensions();
    y.resize(num_outputs * num_x);
    evaluateBatch(x.data(), (int) num_x, y.data());
}
template void TasmanianSparseGrid::evaluateBatch<float>(const std::vector<float> &x, std::vector<float> &y) const;
template void TasmanianSparseGrid::evaluateBatch<double>(const std::vector<double> &x, std::vector<double> &y) const;
void TasmanianSparseGrid::evaluateBatch(const double x[], int num_x, double y[]) const{
    Data2D<double> x_tmp;
    auto const &pnts = formCanonicalPoints(x, x_tmp, num_x);
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.evaluateBatch(pnts, num_x, y);
    }, base);
}

#ifdef Tasmanian_ENABLE_GPU
void TasmanianSparseGrid::evaluateBatch(const float x[], int num_x, float y[]) const{
    if (!( (getAccelerationType() == accel_gpu_cuda) || (getAccelerationType() == accel_gpu_magma) ))
        throw std::runtime_error("ERROR: batch evaluations in single precision require CUDA or MAGMA acceleration to be enabled");
    Data2D<float> x_tmp;
    float const *x_canonical = formCanonicalPoints(x, x_tmp, num_x);
    acceleration->setDevice();
    GpuVector<float> gpu_x(acceleration.get(), getNumDimensions(), num_x, x_canonical),
                     gpu_result(acceleration.get(), num_x, getNumOutputs());
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.evaluateBatchGPU(gpu_x.data(), num_x, gpu_result.data());
    }, base);
    gpu_result.unload(acceleration.get(), y);
}
template<typename FloatType> void TasmanianSparseGrid::evaluateBatchGPU(const FloatType gpu_x[], int cpu_num_x, FloatType gpu_y[]) const{
    if (not acceleration->on_gpu()) throw std::runtime_error("ERROR: evaluateBatchGPU() requires that a cuda gpu acceleration is enabled.");
    acceleration->setDevice();
    GpuVector<FloatType> gpu_temp_x;
    auto const &pnts = formCanonicalPointsGPU(gpu_x, cpu_num_x, gpu_temp_x);
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.evaluateBatchGPU(pnts, cpu_num_x, gpu_y);
    }, base);
}
#else
template<typename FloatType> void TasmanianSparseGrid::evaluateBatchGPU(const FloatType[], int, FloatType[]) const{
    throw std::runtime_error("ERROR: batch evaluations GPU to GPU require Tasmanian_ENABLE_CUDA or Tasmanian_ENABLE_HIP");
}
void TasmanianSparseGrid::evaluateBatch(const float[], int, float[]) const{
    throw std::runtime_error("ERROR: batch evaluations in single precision require Tasmanian_ENABLE_CUDA or Tasmanian_ENABLE_HIP");
}
#endif
template void TasmanianSparseGrid::evaluateBatchGPU<float>(const float[], int, float[]) const;
template void TasmanianSparseGrid::evaluateBatchGPU<double>(const double[], int, double[]) const;

void TasmanianSparseGrid::integrate(double q[]) const{
    if (conformal_asin_power.size() != 0){
        int num_points = getNumPoints();
        std::vector<double> correction(num_points, 1.0);
        mapConformalWeights(getNumDimensions(), num_points, correction.data());
        std::visit([&](auto const &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.integrate(q, correction.data());
        }, base);
    }else{
        std::visit([&](auto const &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.integrate(q, nullptr);
        }, base);
    }
    if (domain_transform_a.size() != 0){
        double scale = getQuadratureScale(getNumDimensions(), getRule());
        for(int k=0; k<getNumOutputs(); k++) q[k] *= scale;
    }
}

void TasmanianSparseGrid::differentiate(const double x[], double jacobian[]) const {
    // For a grid with a transformed-to-canonical operator g(x) and model f(x) over the canonical domain, this returns the
    // Jacobian of f(g(x)), i.e., [Jacobian of f(.) at g(x)] * [Jacobian of g(.) at x].
    Data2D<double> x_tmp;
    // Jacobian of f(.) at g(x).
    auto const &pnts = formCanonicalPoints(x, x_tmp, 1);
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.differentiate(pnts, jacobian);
    }, base);
    // Jacobian of f(g(.)) at x.
    if (not domain_transform_a.empty()) {
        int num_dimensions = getNumDimensions();
        int num_outputs = getNumOutputs();
        std::vector<double> jacobian_g_diag = diffCanonicalTransform<double>();
        for(int j=0; j<num_dimensions; j++)
            for(int k=0; k<num_outputs; k++)
                jacobian[k * num_dimensions + j] = jacobian[k * num_dimensions + j] * jacobian_g_diag[j];
    }
}

void TasmanianSparseGrid::evaluate(const std::vector<double> &x, std::vector<double> &y) const{
    if (x.size() != (size_t) getNumDimensions()) throw std::runtime_error("ERROR: in evaluate() x must match getNumDimensions()");
    y.resize((size_t) getNumOutputs());
    evaluate(x.data(), y.data());
}
void TasmanianSparseGrid::integrate(std::vector<double> &q) const{
    size_t num_outputs = getNumOutputs();
    q.resize(num_outputs);
    integrate(q.data());
}
void TasmanianSparseGrid::differentiate(std::vector<double> const &x, std::vector<double> &jacobian) const {
    size_t num_outputs = getNumOutputs();
    size_t num_dimensions = getNumDimensions();
    jacobian.resize(num_outputs * num_dimensions);
    differentiate(x.data(), jacobian.data());
}

void TasmanianSparseGrid::setDomainTransform(const double a[], const double b[]){
    if (empty())
        throw std::runtime_error("ERROR: cannot call setDomainTransform on uninitialized grid!");
    int num_dimensions = getNumDimensions();
    domain_transform_a.resize(num_dimensions); std::copy(a, a + num_dimensions, domain_transform_a.data());
    domain_transform_b.resize(num_dimensions); std::copy(b, b + num_dimensions, domain_transform_b.data());
    #ifdef Tasmanian_ENABLE_GPU
    acc_domain.clear();
    #endif
}
bool TasmanianSparseGrid::isSetDomainTransfrom() const{
    return (domain_transform_a.size() != 0);
}
void TasmanianSparseGrid::clearDomainTransform(){
    domain_transform_a.resize(0);
    domain_transform_b.resize(0);
    #ifdef Tasmanian_ENABLE_GPU
    acc_domain.clear();
    #endif
}
void TasmanianSparseGrid::getDomainTransform(double a[], double b[]) const{
    if (empty() || (domain_transform_a.size() == 0))
        throw std::runtime_error("ERROR: cannot call getDomainTransform on uninitialized grid or if no transform has been set!");
    std::copy(domain_transform_a.begin(), domain_transform_a.end(), a);
    std::copy(domain_transform_b.begin(), domain_transform_b.end(), b);
}
void TasmanianSparseGrid::setDomainTransform(const std::vector<double> &a, const std::vector<double> &b){
    if (empty()) throw std::runtime_error("ERROR: cannot call setDomainTransform on uninitialized grid!");
    size_t num_dimensions = (size_t) getNumDimensions();
    if ((a.size() != num_dimensions) || (b.size() != num_dimensions)){
        std::string message = "ERROR: setDomainTransform() is called with a.size() = " + std::to_string(a.size()) + " and b.size() = " + std::to_string(b.size()) + ", but both should have length equal to getNumDimensions(), which is: " + std::to_string(num_dimensions);
        throw std::invalid_argument(message);
    }
    domain_transform_a = a; // copy assignment
    domain_transform_b = b;
    #ifdef Tasmanian_ENABLE_GPU
    acc_domain.clear();
    #endif
}
void TasmanianSparseGrid::getDomainTransform(std::vector<double> &a, std::vector<double> &b) const{
    a = domain_transform_a; // copy assignment
    b = domain_transform_b;
}

void TasmanianSparseGrid::mapCanonicalToTransformed(int num_dimensions, int num_points, TypeOneDRule rule, double x[]) const{
    if ((rule == rule_gausslaguerre) || (rule == rule_gausslaguerreodd)){ // canonical (0, +infty)
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] /= domain_transform_b[j];
            x[i] += domain_transform_a[j];
        }
    }else if ((rule == rule_gausshermite) || (rule == rule_gausshermiteodd)){ // (-infty, +infty)
        std::vector<double> sqrt_b(num_dimensions);
        for(int j=0; j<num_dimensions; j++) sqrt_b[j] = std::sqrt(domain_transform_b[j]);
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] /= sqrt_b[j];
            x[i] += domain_transform_a[j];
        }
    }else if (rule == rule_fourier){
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] *= domain_transform_b[j]-domain_transform_a[j];
            x[i] += domain_transform_a[j];
        }
    }else{ // canonical [-1,1]
        std::vector<double> rate(num_dimensions);
        std::vector<double> shift(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            rate[j]  = 0.5* (domain_transform_b[j] - domain_transform_a[j]);
            shift[j] = 0.5* (domain_transform_b[j] + domain_transform_a[j]);
        }
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] *= rate[j];
            x[i] += shift[j];
        }
    }
}
template<typename FloatType> void TasmanianSparseGrid::mapTransformedToCanonical(int num_dimensions, int num_points, TypeOneDRule rule, FloatType x[]) const{
    if ((rule == rule_gausslaguerre) || (rule == rule_gausslaguerreodd)){ // canonical (0, +infty)
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] -= (FloatType) domain_transform_a[j];
            x[i] *= (FloatType) domain_transform_b[j];
        }
    }else if ((rule == rule_gausshermite) || (rule == rule_gausshermiteodd)){ // (-infty, +infty)
        std::vector<double> sqrt_b(num_dimensions);
        for(int j=0; j<num_dimensions; j++) sqrt_b[j] = std::sqrt(domain_transform_b[j]);
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] -= (FloatType) domain_transform_a[j];
            x[i] *= (FloatType) sqrt_b[j];
        }
    }else if (rule == rule_fourier){   // map to [0,1]^d
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] -= (FloatType) domain_transform_a[j];
            x[i] /= (FloatType) (domain_transform_b[j]-domain_transform_a[j]);
        }
    }else{ // canonical [-1,1]
        std::vector<double> rate(num_dimensions);
        std::vector<double> shift(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            rate[j]  = 2.0 / (domain_transform_b[j] - domain_transform_a[j]);
            shift[j] = (domain_transform_b[j] + domain_transform_a[j]) / (domain_transform_b[j] - domain_transform_a[j]);
        }
        for(int i=0; i<num_points * num_dimensions; i++){
            int j = i % num_dimensions;
            x[i] *= (FloatType) rate[j];
            x[i] -= (FloatType) shift[j];
        }
    }
}
template void TasmanianSparseGrid::mapTransformedToCanonical<float>(int num_dimensions, int num_points, TypeOneDRule rule, float x[]) const;
template void TasmanianSparseGrid::mapTransformedToCanonical<double>(int num_dimensions, int num_points, TypeOneDRule rule, double x[]) const;

double TasmanianSparseGrid::getQuadratureScale(int num_dimensions, TypeOneDRule rule) const{
    double scale = 1.0;
    // gauss- (chebyshev1, chebyshev2, gegenbauer) are special case of jacobi
    // points and weight are computed differently for better stability
    // the transform is the same, just have to set the effective alpha/beta for each case
    if ((rule == rule_gausschebyshev1)    || (rule == rule_gausschebyshev2)    || (rule == rule_gaussgegenbauer)    || (rule == rule_gaussjacobi) ||
        (rule == rule_gausschebyshev1odd) || (rule == rule_gausschebyshev2odd) || (rule == rule_gaussgegenbauerodd) || (rule == rule_gaussjacobiodd)){
        double alpha = ((rule == rule_gausschebyshev1) || (rule == rule_gausschebyshev1odd)) ? -0.5 :
                       ((rule == rule_gausschebyshev2) || (rule == rule_gausschebyshev2odd)) ?  0.5 :
                       get<GridGlobal>()->getAlpha();
        double beta = ((rule == rule_gausschebyshev1) || (rule == rule_gausschebyshev1odd)) ? -0.5 :
                      ((rule == rule_gausschebyshev2) || (rule == rule_gausschebyshev2odd)) ?  0.5 :
                      ((rule == rule_gaussgegenbauer) || (rule == rule_gaussgegenbauerodd)) ? get<GridGlobal>()->getAlpha() :
                      get<GridGlobal>()->getBeta();
        for(int j=0; j<num_dimensions; j++) scale *= pow(0.5*(domain_transform_b[j] - domain_transform_a[j]), alpha + beta + 1.0);
    }else if ((rule == rule_gausslaguerre) || (rule == rule_gausslaguerreodd)){
        for(int j=0; j<num_dimensions; j++) scale *= pow(domain_transform_b[j], -(1.0 + get<GridGlobal>()->getAlpha()));
    }else if ((rule == rule_gausshermite) || (rule == rule_gausshermiteodd)){
        double power = -0.5 * (1.0 + get<GridGlobal>()->getAlpha());
        for(int j=0; j<num_dimensions; j++) scale *= pow(domain_transform_b[j], power);
    }else if (rule == rule_fourier){
        for(int j=0; j<num_dimensions; j++) scale *= (domain_transform_b[j] - domain_transform_a[j]);
    }else{
        for(int j=0; j<num_dimensions; j++) scale *= (domain_transform_b[j] - domain_transform_a[j]) / 2.0;
    }
    return scale;
}

void TasmanianSparseGrid::setConformalTransformASIN(std::vector<int> const &truncation){
    if (empty()) throw std::runtime_error("ERROR: cannot call setConformalTransformASIN on uninitialized grid!");
    clearConformalTransform();
    conformal_asin_power = truncation;
}
bool TasmanianSparseGrid::isSetConformalTransformASIN() const{ return (conformal_asin_power.size() != 0); }
void TasmanianSparseGrid::clearConformalTransform(){
    conformal_asin_power.clear();
}
std::vector<int> TasmanianSparseGrid::getConformalTransformASIN() const{
    if (empty() || (conformal_asin_power.size() == 0))
        throw std::runtime_error("ERROR: cannot call getDomainTransform on uninitialized grid or if no transform has been set!");
    return conformal_asin_power;
}

void TasmanianSparseGrid::mapConformalCanonicalToTransformed(int num_dimensions, int num_points, double x[]) const{
    if (conformal_asin_power.size() != 0){
        // precompute constants, transform is sum exp(c_k + p_k * log(x))
        std::vector<std::vector<double>> c(num_dimensions), p(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            c[j].resize(conformal_asin_power[j] + 1);
            p[j].resize(conformal_asin_power[j] + 1);
        }
        double lgamma_half = std::lgamma(0.5);
        std::vector<double> cm(num_dimensions, 0.0);
        for(int j=0; j<num_dimensions; j++){
            double factorial = 0.0;
            for(int k=0; k<=conformal_asin_power[j]; k++){
                p[j][k] = (double)(2*k+1);
                c[j][k] = std::lgamma(0.5 + ((double) k)) - lgamma_half - std::log(p[j][k]) - factorial;
                cm[j] += std::exp(c[j][k]);
                factorial += std::log((double)(k+1));
            }
        }
        Utils::Wrapper2D<double> xwrap(num_dimensions, x);
        for(int i=0; i<num_points; i++){
            double *this_x = xwrap.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (this_x[j] != 0.0){ // zero maps to zero and makes the log unstable
                    double sign = (this_x[j] > 0.0) ? 1.0 : -1.0;
                    double logx = std::log(std::abs(this_x[j]));
                    this_x[j] = 0.0;
                    for(int k=0; k<=conformal_asin_power[j]; k++){
                        this_x[j] += std::exp(c[j][k] + p[j][k] * logx);
                    }
                    this_x[j] *= sign / cm[j];
                }
            }
        }
    }
}
template<typename FloatType> void TasmanianSparseGrid::mapConformalTransformedToCanonical(int num_dimensions, int num_points, Data2D<FloatType> &x) const{
    if (conformal_asin_power.size() != 0){
        // precompute constants, transform is sum exp(c_k + p_k * log(x))
        std::vector<std::vector<double>> c(num_dimensions), p(num_dimensions), dc(num_dimensions), dp(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            c[j].resize(conformal_asin_power[j] + 1);
            p[j].resize(conformal_asin_power[j] + 1);
            dc[j].resize(conformal_asin_power[j] + 1);
            dp[j].resize(conformal_asin_power[j] + 1);
        }
        double lgamma_half = std::lgamma(0.5);
        std::vector<double> cm(num_dimensions, 0.0);
        for(int j=0; j<num_dimensions; j++){
            double factorial = 0.0;
            for(int k=0; k<=conformal_asin_power[j]; k++){
                p[j][k] = (double)(2*k+1);
                c[j][k] = std::lgamma(0.5 + ((double) k)) - lgamma_half - std::log(p[j][k]) - factorial;
                cm[j] += std::exp(c[j][k]);
                dp[j][k] = (double)(2*k);
                dc[j][k] = std::lgamma(0.5 + ((double) k)) - lgamma_half - factorial;
                factorial += std::log((double)(k+1));
            }
        }
        for(int i=0; i<num_points; i++){
            FloatType *this_x = x.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (this_x[j] != 0.0){ // zero maps to zero and makes the log unstable
                    double sign = (this_x[j] > 0.0) ? 1.0 : -1.0;
                    this_x[j] = (FloatType) std::abs(this_x[j]);
                    double b = this_x[j];
                    double logx = log(this_x[j]);
                    double r = this_x[j];
                    double dr = 1.0;
                    for(int k=1; k<=conformal_asin_power[j]; k++){
                        r  += std::exp( c[j][k] +  p[j][k] * logx);
                        dr += std::exp(dc[j][k] + dp[j][k] * logx);
                   }
                    r /= cm[j];
                    r -= b; // transformed_x -b = 0
                    while(std::abs(r) > Maths::num_tol){
                        this_x[j] -= (FloatType) (r * cm[j] / dr);

                        logx = std::log(std::abs(this_x[j]));
                        r = this_x[j];
                        dr = 1.0;
                        for(int k=1; k<=conformal_asin_power[j]; k++){
                            r  += std::exp( c[j][k] +  p[j][k] * logx);
                            dr += std::exp(dc[j][k] + dp[j][k] * logx);
                       }
                        r /= cm[j];
                        r -= b;
                    }
                    this_x[j] *= (FloatType) sign;
                }
            }
        }
    }
}
template void TasmanianSparseGrid::mapConformalTransformedToCanonical<float>(int num_dimensions, int num_points, Data2D<float> &x) const;
template void TasmanianSparseGrid::mapConformalTransformedToCanonical<double>(int num_dimensions, int num_points, Data2D<double> &x) const;

void TasmanianSparseGrid::mapConformalWeights(int num_dimensions, int num_points, double weights[]) const{
    if (conformal_asin_power.size() != 0){
        // precompute constants, transform is sum exp(c_k + p_k * log(x))
        Data2D<double> x(num_dimensions, num_points);
        std::visit([&](auto const &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.getPoints(x.getStrip(0));
        }, base);
        std::vector<std::vector<double>> c(num_dimensions), p(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            c[j].resize(conformal_asin_power[j] + 1);
            p[j].resize(conformal_asin_power[j] + 1);
        }
        double lgamma_half = std::lgamma(0.5);
        std::vector<double> cm(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            double factorial = 0.0;
            cm[j] = 0.0;
            for(int k=0; k<=conformal_asin_power[j]; k++){
                p[j][k] = (double)(2*k);
                c[j][k] = std::lgamma(0.5 + ((double) k)) - lgamma_half - factorial;
                factorial += std::log((double)(k+1));
                cm[j] += std::exp(c[j][k] - std::log((double)(2*k+1)));
            }
        }
        for(int i=0; i<num_points; i++){
            const double *this_x = x.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (this_x[j] != 0.0){ // derivative at zero is 1/cm[j] and zero makes the log unstable
                    double logx = std::log(std::abs(this_x[j]));
                    double trans = 1.0;
                    for(int k=1; k<=conformal_asin_power[j]; k++){
                        trans += std::exp(c[j][k] + p[j][k] * logx);
                   }
                    weights[i] *= trans / cm[j];
                }else{
                    weights[i] /= cm[j];
                }
            }
        }
    }
}

template<typename FloatType> const FloatType* TasmanianSparseGrid::formCanonicalPoints(const FloatType *x, Data2D<FloatType> &x_temp, int num_x) const{
    if ((domain_transform_a.size() != 0) || (conformal_asin_power.size() != 0)){
        int num_dimensions = getNumDimensions();
        x_temp = Data2D<FloatType>(num_dimensions, num_x, std::vector<FloatType>(x, x + Utils::size_mult(num_dimensions, num_x)));
        mapConformalTransformedToCanonical(num_dimensions, num_x, x_temp);
        if (domain_transform_a.size() != 0) mapTransformedToCanonical(num_dimensions, num_x, getRule(), x_temp.getStrip(0));
        return x_temp.getStrip(0);
    }else{
        return x;
    }
}
template const float* TasmanianSparseGrid::formCanonicalPoints(const float *x, Data2D<float> &x_temp, int num_x) const;
template const double* TasmanianSparseGrid::formCanonicalPoints<double>(const double *x, Data2D<double> &x_temp, int num_x) const;

template<typename FloatType> std::vector<double> TasmanianSparseGrid::diffCanonicalTransform() const {
    int num_dimensions = getNumDimensions();
    std::vector<double> jacobian_diag(num_dimensions, 1.0);
    if (conformal_asin_power.size() != 0) {
        throw std::runtime_error("ERROR: in diffCanonicalTransform() derivatives/Jacobians are not available for conformal mappings");
    }
    if (domain_transform_a.size() != 0) {
        // Based on the logic in mapTransformedToCanonical(). These are all affine transforms.
        TypeOneDRule rule = getRule();
        if (rule == rule_gausslaguerre or rule == rule_gausslaguerreodd)
            for(int j=0; j<num_dimensions; j++)
                jacobian_diag[j] = (FloatType) domain_transform_b[j];
        else if (rule == rule_gausshermite or rule == rule_gausshermiteodd)
            for(int j=0; j<num_dimensions; j++)
                jacobian_diag[j] = (FloatType) std::sqrt(domain_transform_b[j]);
        else if (rule == rule_fourier)
            for(int j=0; j< num_dimensions; j++)
                jacobian_diag[j] = (FloatType) (1.0 / (domain_transform_b[j]-domain_transform_a[j]));
        else
            for(int j=0; j<num_dimensions; j++)
                jacobian_diag[j] = (FloatType) (2.0 / (domain_transform_b[j] - domain_transform_a[j]));
    }
    return jacobian_diag;
}

void TasmanianSparseGrid::formTransformedPoints(int num_points, double x[]) const{
    mapConformalCanonicalToTransformed(getNumDimensions(), num_points, x); // internally switch based on the conformal transform
    if (domain_transform_a.size() != 0){ // check the basic domain
        mapCanonicalToTransformed(getNumDimensions(), num_points, getRule(), x);
    }
}

template<typename T>
const T* TasmanianSparseGrid::formCanonicalPointsGPU(const T *gpu_x, int num_x, GpuVector<T> &gpu_x_temp) const{
    if (!domain_transform_a.empty()){
        if (!acc_domain)
            acc_domain.set(acceleration.get(), domain_transform_a, domain_transform_b);
        acc_domain.getCanonicalPoints(isFourier(), gpu_x, num_x, gpu_x_temp);
        return gpu_x_temp.data();
    }else{
        return gpu_x;
    }
}

void TasmanianSparseGrid::setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const int *level_limits){
    if (using_dynamic_construction) throw std::runtime_error("ERROR: setAnisotropicRefinement() called before finishConstruction()");
    if (empty()) throw std::runtime_error("ERROR: calling setAnisotropicRefinement() for a grid that has not been initialized");
    setAnisotropicRefinement(type, min_growth, output, Utils::copyArray(level_limits, getNumDimensions()));
}
void TasmanianSparseGrid::setAnisotropicRefinement(TypeDepth type, int min_growth, int output, const std::vector<int> &level_limits){
    if (using_dynamic_construction) throw std::runtime_error("ERROR: setAnisotropicRefinement() called before finishConstruction()");
    if (empty()) throw std::runtime_error("ERROR: calling setAnisotropicRefinement() for a grid that has not been initialized");
    if (min_growth < 1) throw std::invalid_argument("ERROR: setAnisotropicRefinement() requires positive min_growth");
    int dims = getNumDimensions();
    int outs = getNumOutputs();
    if (outs == 0) throw std::runtime_error("ERROR: calling setAnisotropicRefinement() for a grid that has no outputs");
    if (getNumLoaded() == 0) throw std::runtime_error("ERROR: calling setAnisotropicRefinement() for a grid with no loaded values");
    if ((output < -1) || (output >= outs)) throw std::invalid_argument("ERROR: calling setAnisotropicRefinement() with invalid output");
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dims)) throw std::invalid_argument("ERROR: setAnisotropicRefinement() requires level_limits with either 0 or dimenions entries");

    if (!level_limits.empty()) llimits = level_limits;
    if (isSequence()){
        get<GridSequence>()->setAnisotropicRefinement(type, min_growth, output, llimits);
    }else if (isGlobal()){
        if (OneDimensionalMeta::isNonNested(get<GridGlobal>()->getRule())){
            throw std::runtime_error("ERROR: setAnisotropicRefinement() called for a global grid with non-nested rule");
        }else{
            get<GridGlobal>()->setAnisotropicRefinement(type, min_growth, output, llimits);
        }
    }else if (isFourier()){
        get<GridFourier>()->setAnisotropicRefinement(type, min_growth, output, llimits);
    }else{
        throw std::runtime_error("ERROR: setAnisotropicRefinement() called for a grid that is neither Sequence, nor Global with a sequence rule, nor Fourier");
    }
}

void TasmanianSparseGrid::estimateAnisotropicCoefficients(TypeDepth type, int output, std::vector<int> &weights) const{
    if (empty()) throw std::runtime_error("ERROR: calling estimateAnisotropicCoefficients() for a grid that has not been initialized");
    int outs = getNumOutputs();
    if (outs == 0) throw std::runtime_error("ERROR: calling estimateAnisotropicCoefficients() for a grid that has no outputs");
    if (getNumLoaded() == 0) throw std::runtime_error("ERROR: calling estimateAnisotropicCoefficients() for a grid with no loaded values");
    if ((output < -1) || (output >= outs)) throw std::invalid_argument("ERROR: calling estimateAnisotropicCoefficients() with invalid output");

    if (isSequence()){
        get<GridSequence>()->estimateAnisotropicCoefficients(type, output, weights);
    }else if (isGlobal()){
        if (OneDimensionalMeta::isNonNested(get<GridGlobal>()->getRule())){
            throw std::runtime_error("ERROR: estimateAnisotropicCoefficients called for a Global grid with non-nested rule");
        }else{
            get<GridGlobal>()->estimateAnisotropicCoefficients(type, output, weights);
        }
    }else if (isFourier()){
        get<GridFourier>()->estimateAnisotropicCoefficients(type, output, weights);
    }else{
        throw std::runtime_error("ERROR: estimateAnisotropicCoefficients called for a grid that is neither Sequence nor Global with a sequence rule");
    }
}

void TasmanianSparseGrid::setSurplusRefinement(double tolerance, int output, const int *level_limits){
    if (empty()) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid that has not been initialized");
    setSurplusRefinement(tolerance, output, Utils::copyArray(level_limits, getNumDimensions()));
}
void TasmanianSparseGrid::setSurplusRefinement(double tolerance, int output, const std::vector<int> &level_limits){
    if (using_dynamic_construction) throw std::runtime_error("ERROR: setSurplusRefinement() called before finishConstruction()");
    if (empty()) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid that has not been initialized");
    int dims = getNumDimensions();
    int outs = getNumOutputs();
    if (outs == 0) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid that has no outputs");
    if (getNumLoaded() == 0) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid with no loaded values");
    if ((output < -1) || (output >= outs)) throw std::invalid_argument("ERROR: calling setSurplusRefinement() with invalid output");
    if (tolerance < 0.0) throw std::invalid_argument("ERROR: calling setSurplusRefinement() with invalid tolerance (must be non-negative)");
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dims)) throw std::invalid_argument("ERROR: setSurplusRefinement() requires level_limits with either 0 or dimenions entries");

    if (!level_limits.empty()) llimits = level_limits;
    if (isSequence()){
        get<GridSequence>()->setSurplusRefinement(tolerance, output, llimits);
    }else if (isGlobal()){
        if (OneDimensionalMeta::isSequence(get<GridGlobal>()->getRule())){
            get<GridGlobal>()->setSurplusRefinement(tolerance, output, llimits);
        }else{
            throw std::runtime_error("ERROR: setSurplusRefinement called for a Global grid with non-sequence rule");
        }
    }else{
        throw std::runtime_error("ERROR: setSurplusRefinement(double, int) called for a grid that is neither Sequence nor Global with a sequence rule");
    }
}

void TasmanianSparseGrid::setSurplusRefinement(double tolerance, TypeRefinement criteria, int output, const int *level_limits, const double *scale_correction){
    if (using_dynamic_construction) throw std::runtime_error("ERROR: setSurplusRefinement() called before finishConstruction()");
    if (empty()) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid that has not been initialized");
    int dims = getNumDimensions();
    int outs = getNumOutputs();
    if (outs == 0) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid that has no outputs");
    if (getNumLoaded() == 0) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid with no loaded values");
    if ((output < -1) || (output >= outs)) throw std::invalid_argument("ERROR: calling setSurplusRefinement() with invalid output");
    if (isFourier())
        throw std::runtime_error("ERROR: setSurplusRefinement(double, TypeRefinement) called for a Fourier grid.");
    if (tolerance < 0.0) throw std::invalid_argument("ERROR: calling setSurplusRefinement() with invalid tolerance (must be non-negative)");

    if (level_limits != 0) // can only happen if calling directly with int*, the vector version always passes null for level_limits
        llimits = Utils::copyArray(level_limits, dims); // if level_limits is null, we want to keep llimits unchanged

    if (isLocalPolynomial()){
        get<GridLocalPolynomial>()->setSurplusRefinement(tolerance, criteria, output, llimits, scale_correction);
    }else if (isWavelet()){
        get<GridWavelet>()->setSurplusRefinement(tolerance, criteria, output, llimits);
    }else{
        setSurplusRefinement(tolerance, output, std::vector<int>()); // new level limits are already set above
    }
}
void TasmanianSparseGrid::setSurplusRefinement(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits, const std::vector<double> &scale_correction){
    if (empty()) throw std::runtime_error("ERROR: calling setSurplusRefinement() for a grid that has not been initialized");
    int dims = getNumDimensions();
    size_t nscale = (size_t) getNumNeeded();
    if (output != -1) nscale *= (size_t) getNumOutputs();
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dims)) throw std::invalid_argument("ERROR: setSurplusRefinement() requires level_limits with either 0 or dimenions entries");
    if ((!scale_correction.empty()) && (scale_correction.size() != nscale)) throw std::invalid_argument("ERROR: setSurplusRefinement() incorrect size for scale_correction");

    if (!level_limits.empty()) llimits = level_limits;
    setSurplusRefinement(tolerance, criteria, output, nullptr, (scale_correction.empty()) ? nullptr : scale_correction.data());
}

void TasmanianSparseGrid::clearRefinement(){
    std::visit([](auto &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.clearRefinement();
    }, base);
}
void TasmanianSparseGrid::mergeRefinement(){
    std::visit([](auto &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.mergeRefinement();
    }, base);
}

void TasmanianSparseGrid::beginConstruction(){
    if (empty()) throw std::runtime_error("ERROR: cannot start construction for an empty grid.");
    if (not using_dynamic_construction){
        if (getNumLoaded() > 0) clearRefinement();
        using_dynamic_construction = true;
        std::visit([](auto &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.beginConstruction();
        }, base);
    }
}
std::vector<double> TasmanianSparseGrid::getCandidateConstructionPoints(TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits){
    if (not using_dynamic_construction) throw std::runtime_error("ERROR: getCandidateConstructionPoints() called before beginConstruction()");
    if (isLocalPolynomial() || isWavelet()) throw std::runtime_error("ERROR: getCandidateConstructionPoints() anisotropic version called for local polynomial grid");
    size_t dims = (size_t) getNumDimensions();
    if ((!level_limits.empty()) && (level_limits.size() != (size_t) dims)) throw std::invalid_argument("ERROR: getCandidateConstructionPoints() requires level_limits with either 0 or num-dimensions entries");
    if ((type == type_curved) || (type == type_ipcurved) || (type == type_qpcurved)){
        if (anisotropic_weights.size() != 2 * dims) throw std::invalid_argument("ERROR: getCandidateConstructionPoints() called with curved type and incorrect size for anisotropic_weights (must be twice the number of dimensions)");
    }else{
        if (anisotropic_weights.size() != dims) throw std::invalid_argument("ERROR: getCandidateConstructionPoints() called with incorrect size for anisotropic_weights (must match number of dimensions)");
    }

    if (!level_limits.empty()) llimits = level_limits;
    std::vector<double> x;
    if (isGlobal()){
        x = get<GridGlobal>()->getCandidateConstructionPoints(type, anisotropic_weights, llimits);
    }else if (isSequence()){
        x = get<GridSequence>()->getCandidateConstructionPoints(type, anisotropic_weights, llimits);
    }else{ // Fourier
        x = get<GridFourier>()->getCandidateConstructionPoints(type, anisotropic_weights, llimits);
    }
    formTransformedPoints((int) x.size() / getNumDimensions(), x.data());
    return x;
}
std::vector<double> TasmanianSparseGrid::getCandidateConstructionPoints(TypeDepth type, int output, const std::vector<int> &level_limits){
    if (not using_dynamic_construction) throw std::runtime_error("ERROR: getCandidateConstructionPoints() called before beginConstruction()");
    if (isLocalPolynomial() || isWavelet()) throw std::runtime_error("ERROR: getCandidateConstructionPoints() anisotropic version called for local polynomial grid");
    size_t dims = (size_t) getNumDimensions();
    if ((!level_limits.empty()) && (level_limits.size() != dims)) throw std::invalid_argument("ERROR: getCandidateConstructionPoints() requires level_limits with either 0 or num-dimensions entries");
    int outs = getNumOutputs();
    if (outs == 0) throw std::runtime_error("ERROR: calling getCandidateConstructionPoints() for a grid that has no outputs");
    if ((output < -1) || (output >= outs)) throw std::invalid_argument("ERROR: calling getCandidateConstructionPoints() with invalid output");

    if (!level_limits.empty()) llimits = level_limits;
    std::vector<double> x;
    if (isGlobal()){
        x = get<GridGlobal>()->getCandidateConstructionPoints(type, output, llimits);
    }else if (isSequence()){
        x = get<GridSequence>()->getCandidateConstructionPoints(type, output, llimits);
    }else{ // Fourier
        x = get<GridFourier>()->getCandidateConstructionPoints(type, output, llimits);
    }
    formTransformedPoints((int) x.size() / getNumDimensions(), x.data());
    return x;
}
std::vector<double> TasmanianSparseGrid::getCandidateConstructionPoints(double tolerance, TypeRefinement criteria,
                                                                        int output, const std::vector<int> &level_limits, const std::vector<double> &scale_correction){
    if (not using_dynamic_construction) throw std::runtime_error("ERROR: getCandidateConstructionPoints() called before beginConstruction()");
    if (!isLocalPolynomial() && !isWavelet()) throw std::runtime_error("ERROR: getCandidateConstructionPoints() surplus version called for non-local polynomial or wavelet grid");
    size_t dims = (size_t) getNumDimensions();
    if ((!level_limits.empty()) && (level_limits.size() != dims)) throw std::invalid_argument("ERROR: getCandidateConstructionPoints() requires level_limits with either 0 or num-dimensions entries");
    int outs = getNumOutputs();
    if (outs == 0) throw std::runtime_error("ERROR: calling getCandidateConstructionPoints() for a grid that has no outputs");
    if ((output < -1) || (output >= outs)) throw std::invalid_argument("ERROR: calling getCandidateConstructionPoints() with invalid output");

    if (!level_limits.empty()) llimits = level_limits;
    auto x = (isWavelet()) ? get<GridWavelet>()->getCandidateConstructionPoints(tolerance, criteria, output, llimits) :
        get<GridLocalPolynomial>()->getCandidateConstructionPoints(tolerance, criteria, output, llimits, ((scale_correction.empty()) ? nullptr : scale_correction.data()));
    formTransformedPoints((int) x.size() / getNumDimensions(), x.data());
    return x;
}
void TasmanianSparseGrid::loadConstructedPoints(const std::vector<double> &x, const std::vector<double> &y){
    int numx = (int) x.size() / getNumDimensions();
    if (y.size() < Utils::size_mult(numx, getNumOutputs())) throw std::runtime_error("ERROR: loadConstructedPoint() called with incorrect size for y");
    loadConstructedPoints(x.data(), numx, y.data());
}
void TasmanianSparseGrid::loadConstructedPoints(const double x[], int numx, const double y[]){
    if (not using_dynamic_construction) throw std::runtime_error("ERROR: loadConstructedPoint() called before beginConstruction()");
    Data2D<double> x_tmp;
    const double *x_canonical = formCanonicalPoints(x, x_tmp, numx);
    if (numx == 1)
        std::visit([&](auto &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.loadConstructedPoint(x_canonical, Utils::copyArray(y, getNumOutputs()));
        }, base);

    else
        std::visit([&](auto &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.loadConstructedPoint(x_canonical, numx, y);
        }, base);
}
void TasmanianSparseGrid::finishConstruction(){
    if (using_dynamic_construction)
        std::visit([](auto &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.finishConstruction();
        }, base);
    using_dynamic_construction = false;
}

void TasmanianSparseGrid::removePointsByHierarchicalCoefficient(double tolerance, int output, const double *scale_correction){
    if (!isLocalPolynomial()){
        throw std::runtime_error("ERROR: removePointsBySurplus() called for a grid that is not Local Polynomial.");
    }else{
        if (get<GridLocalPolynomial>()->removePointsByHierarchicalCoefficient(tolerance, output, scale_correction) == 0){
            clear();
        }
    }
}
void TasmanianSparseGrid::removePointsByHierarchicalCoefficient(int num_new_points, int output, const double *scale_correction){
    if (!isLocalPolynomial()){
        throw std::runtime_error("ERROR: removePointsBySurplus() called for a grid that is not Local Polynomial.");
    }else{
        if (num_new_points == 0){ clear(); return; }
        get<GridLocalPolynomial>()->removePointsByHierarchicalCoefficient(num_new_points, output, scale_correction);
    }
}

void TasmanianSparseGrid::evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const{
    Data2D<double> x_tmp;
    auto const &pnts = formCanonicalPoints(x, x_tmp, num_x);
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.evaluateHierarchicalFunctions(pnts, num_x, y);
    }, base);
}
void TasmanianSparseGrid::evaluateHierarchicalFunctions(const std::vector<double> &x, std::vector<double> &y) const{
    if (empty()) throw std::runtime_error("ERROR: cannot call evaluateHierarchicalFunctions() on an empty grid");
    int num_points = getNumPoints();
    size_t num_x = x.size() / getNumDimensions();
    size_t expected_size = num_points * num_x * (isFourier() ? 2 : 1);
    y.resize(expected_size);
    evaluateHierarchicalFunctions(x.data(), (int) num_x, y.data());
}

template<typename T>
void TasmanianSparseGrid::evaluateHierarchicalFunctionsGPU(const T gpu_x[], int cpu_num_x, T gpu_y[]) const{
    if (not AccelerationMeta::isAvailable(accel_gpu_cuda))
        throw std::runtime_error("ERROR: evaluateHierarchicalFunctionsGPU() called, but the library was not compiled without Tasmanian_ENABLE_CUDA=ON and Tasmanian_ENABLE_HIP=ON");
    if (not acceleration->on_gpu()) throw std::runtime_error("ERROR: evaluateHierarchicalFunctionsGPU() requires that a cuda gpu acceleration is enabled.");
    acceleration->setDevice();
    GpuVector<T> gpu_temp_x;
    auto const &pnts = formCanonicalPointsGPU(gpu_x, cpu_num_x, gpu_temp_x);
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.evaluateHierarchicalFunctionsGPU(pnts, cpu_num_x, gpu_y);
    }, base);
}
template<typename T>
void TasmanianSparseGrid::evaluateSparseHierarchicalFunctionsGPU(const T gpu_x[], int cpu_num_x, int* &gpu_pntr, int* &gpu_indx, T* &gpu_vals, int &num_nz) const{
    if (not AccelerationMeta::isAvailable(accel_gpu_cuda))
        throw std::runtime_error("ERROR: evaluateSparseHierarchicalFunctionsGPU() called, but the library was not compiled without Tasmanian_ENABLE_CUDA=ON and Tasmanian_ENABLE_HIP=ON");
    if (!isLocalPolynomial()) throw std::runtime_error("ERROR: evaluateSparseHierarchicalFunctionsGPU() is allowed only for local polynomial grid.");
    if (not acceleration->on_gpu()) throw std::runtime_error("ERROR: evaluateSparseHierarchicalFunctionsGPU() requires that a cuda gpu acceleration is enabled.");
    acceleration->setDevice();
    GpuVector<T> gpu_temp_x;
    const T *gpu_canonical_x = formCanonicalPointsGPU(gpu_x, cpu_num_x, gpu_temp_x);
    GpuVector<int> vec_pntr, vec_indx;
    GpuVector<T> vec_vals;
    get<GridLocalPolynomial>()->buildSparseBasisMatrixGPU(gpu_canonical_x, cpu_num_x, vec_pntr, vec_indx, vec_vals);
    num_nz = (int) vec_indx.size();
    gpu_pntr = vec_pntr.eject();
    gpu_indx = vec_indx.eject();
    gpu_vals = vec_vals.eject();
}

template void TasmanianSparseGrid::evaluateHierarchicalFunctionsGPU<float>(float const gpu_x[], int cpu_num_x, float gpu_y[]) const;
template void TasmanianSparseGrid::evaluateHierarchicalFunctionsGPU<double>(double const gpu_x[], int cpu_num_x, double gpu_y[]) const;
template void TasmanianSparseGrid::evaluateSparseHierarchicalFunctionsGPU<float>(const float[], int, int*&, int*&, float*&, int&) const;
template void TasmanianSparseGrid::evaluateSparseHierarchicalFunctionsGPU<double>(const double[], int, int*&, int*&, double*&, int&) const;

void TasmanianSparseGrid::evaluateSparseHierarchicalFunctions(const std::vector<double> &x, std::vector<int> &pntr, std::vector<int> &indx, std::vector<double> &vals) const{
    if (!isLocalPolynomial() && !isWavelet()) throw std::runtime_error("ERROR: evaluateSparseHierarchicalFunctions() called for a grid that is neither local polynomial not wavelet");
    int num_x = ((int) x.size()) / getNumDimensions();
    Data2D<double> x_tmp;
    const double *x_canonical = formCanonicalPoints(x.data(), x_tmp, num_x);
    if (isLocalPolynomial()){
        get<GridLocalPolynomial>()->buildSpareBasisMatrix(x_canonical, num_x, 32, pntr, indx, vals);
    }else{
        int num_points = getNumPoints();
        std::vector<double> dense_vals(((size_t) num_points) * ((size_t) num_x));
        std::visit([&](auto const &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.evaluateHierarchicalFunctions(x_canonical, num_x, dense_vals.data());
        }, base);
        int num_nz = 0;
        for(int i=0; i<num_points * num_x; i++) if (dense_vals[i] != 0.0) num_nz++;
        pntr.resize(num_x+1);
        indx.resize(num_nz);
        vals.resize(num_nz);
        num_nz = 0;
        for(int i=0; i<num_x; i++){
            pntr[i] = num_nz;
            for(int j=0; j<num_points; j++){
                if (dense_vals[i*num_points + j] != 0){
                    indx[num_nz] = j;
                    vals[num_nz++] = dense_vals[i*num_points + j];
                }
            }
        }
        pntr[num_x] = num_nz;
    }
}
int TasmanianSparseGrid::evaluateSparseHierarchicalFunctionsGetNZ(const double x[], int num_x) const{
    Data2D<double> x_tmp;
    const double *x_canonical = formCanonicalPoints(x, x_tmp, num_x);
    if (isLocalPolynomial()){
        return get<GridLocalPolynomial>()->getSpareBasisMatrixNZ(x_canonical, num_x);
    }else if (isWavelet()){
        int num_points = getNumPoints();
        Data2D<double> dense_vals(num_points, num_x);
        get<GridWavelet>()->evaluateHierarchicalFunctions(x_canonical, num_x, dense_vals.data());
        return static_cast<int>(dense_vals.getTotalEntries() - std::count(dense_vals.begin(), dense_vals.end(), 0.0));
    }else if (empty()){
        return 0;
    }else{
        throw std::runtime_error("ERROR: evaluateSparseHierarchicalFunctionsGetNZ() called for a grid that is neither local polynomial not wavelet");
    }
}
void TasmanianSparseGrid::evaluateSparseHierarchicalFunctionsStatic(const double x[], int num_x, int pntr[], int indx[], double vals[]) const{
    if (empty()) return;
    Data2D<double> x_tmp;
    const double *x_canonical = formCanonicalPoints(x, x_tmp, num_x);
    if (isLocalPolynomial()){
        get<GridLocalPolynomial>()->buildSpareBasisMatrixStatic(x_canonical, num_x, 32, pntr, indx, vals);
    }else if (isWavelet()){
        int num_points = getNumPoints();
        Data2D<double> dense_vals(num_points, num_x);
        std::visit([&](auto const &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.evaluateHierarchicalFunctions(x_canonical, num_x, dense_vals.getStrip(0));
        }, base);
        int num_nz = 0;
        for(int i=0; i<num_x; i++){
            pntr[i] = num_nz;
            const double *v = dense_vals.getStrip(i);
            for(int j=0; j<num_points; j++){
                if (v[j] != 0.0){
                    indx[num_nz] = j;
                    vals[num_nz] = v[j];
                    num_nz++;
                }
            }
        }
        pntr[num_x] = num_nz;
    }else{
        throw std::runtime_error("ERROR: evaluateSparseHierarchicalFunctionsStatic() called for a grid that is neither local polynomial not wavelet");
    }
}

std::vector<double> TasmanianSparseGrid::getHierarchicalSupport() const{
    std::vector<double> support = std::visit([&](auto const &g) -> std::vector<double>{
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            return g.getSupport();
        else
            return {};
    }, base);

    if (!domain_transform_a.empty()){
        std::vector<double> correction(domain_transform_a.size());
        std::transform(domain_transform_a.begin(), domain_transform_a.end(), domain_transform_b.begin(),
                       correction.begin(), [](double a, double b)->double{ return 0.5 * (b - a); });

        for(auto is = support.begin(); is < support.end(); ){
            for(auto c : correction) *is++ *= c;
        }
    }

    return support;
}

void TasmanianSparseGrid::setHierarchicalCoefficients(const std::vector<double> &c){
    size_t num_coeffs = Utils::size_mult(getNumOutputs(), getNumPoints()) * ((isFourier()) ? 2 : 1);
    if (c.size() != num_coeffs) throw std::runtime_error("ERROR: setHierarchicalCoefficients() called with wrong size of the coefficients.");
    setHierarchicalCoefficients(c.data());
}
void TasmanianSparseGrid::setHierarchicalCoefficients(const double c[]){
    std::visit([&](auto &g) -> void {
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
            g.setHierarchicalCoefficients(c);
        }
    }, base);
}
void TasmanianSparseGrid::integrateHierarchicalFunctions(double integrals[]) const{
    if (empty()) throw std::runtime_error("ERROR: cannot compute the integrals for a basis in an empty grid.");
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.integrateHierarchicalFunctions(integrals);
    }, base);
    if (domain_transform_a.size() != 0){
        double scale = getQuadratureScale(getNumDimensions(), getRule());
        for(int i=0; i<getNumPoints(); i++) integrals[i] *= scale;
    }
}

std::vector<int> TasmanianSparseGrid::getGlobalPolynomialSpace(bool interpolation) const{
    if (isGlobal()){
        return get<GridGlobal>()->getPolynomialSpace(interpolation);
    }else if (isSequence()){
        return get<GridSequence>()->getPolynomialSpace(interpolation);
    }else{
        throw std::runtime_error("ERROR: getGlobalPolynomialSpace() called for a grid that is neither Global nor Sequence");
    }
}
const double* TasmanianSparseGrid::getHierarchicalCoefficients() const{
    return std::visit([](auto const &g) -> const double * {
        using T = std::decay_t<decltype(g)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return nullptr;
        } else if constexpr (std::is_same_v<T, GridGlobal>) {
            return g.getLoadedValues();
        } else if constexpr (std::is_same_v<T, GridFourier>) {
            return g.getFourierCoefs();
        } else {
            return g.getSurpluses();
        }
    }, base);
}
const int* TasmanianSparseGrid::getPointsIndexes() const{
    if (empty())
        throw std::runtime_error("ERROR: getPointIndexes() called for a grid that has not been initialized");
    return std::visit([](auto const &g) -> const int* {
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            return g.getPointIndexes();
        else
            return nullptr;
    }, base);
}
const int* TasmanianSparseGrid::getNeededIndexes() const{
    if (isLocalPolynomial()){
        return get<GridLocalPolynomial>()->getNeededIndexes();
    }else{
        throw std::runtime_error("ERROR: getPointIndexes() called for a grid that is not Local Polynomial");
    }
}

void TasmanianSparseGrid::printStats(std::ostream &os) const{
    using std::setw;

    const int L1 = 20;
    os << '\n';
    os << setw(L1) << "Grid Type:" << "  ";
    if (isGlobal()) os << "Global";
    if (isSequence()) os << "Sequence";
    if (isLocalPolynomial()) os << "Local Polynomial";
    if (isWavelet()) os << "Wavelets";
    if (isFourier()) os << "Fourier";
    if (!(isGlobal() || isSequence() || isLocalPolynomial() || isWavelet() || isFourier())) os << "none";
    os << '\n';

    os << setw(L1) << "Dimensions:" << "   " << getNumDimensions() << '\n';
    os << setw(L1) << "Outputs:" << "   " << getNumOutputs() << '\n';
    if (getNumOutputs() == 0){
        os << setw(L1) << "Nodes:" << "   " << getNumPoints() << '\n';
    }else{
        os << setw(L1) << "Loaded nodes:" << "   " << getNumLoaded() << '\n';
        os << setw(L1) << "Needed nodes:" << "   " << getNumNeeded() << '\n';
    }
    os << setw(L1) << "Rule:" << "  " << OneDimensionalMeta::getHumanString(getRule()) << '\n';
    if (getRule() == rule_customtabulated){
        os << setw(L1) << "Description:" << "  " << getCustomRuleDescription() << '\n';
    }
    if (isSetDomainTransfrom()){
        os << setw(L1) << "Domain:" << "  Custom" << '\n';
    }else{
        os << setw(L1) << "Domain:" << "  Canonical" << '\n';
    }

    if (isGlobal()){
        TypeOneDRule rr = getRule();
        if ((rr == rule_gaussgegenbauer) || (rr == rule_gausslaguerre) || (rr == rule_gausshermite) || (rr == rule_gaussgegenbauerodd) || (rr == rule_gausshermiteodd) ){
            os << setw(L1) << "Alpha:" << "   " << getAlpha() << '\n';
        }
        if (rr == rule_gaussjacobi){
            os << setw(L1) << "Alpha:" << "   " << getAlpha() << '\n';
            os << setw(L1) << "Beta:" << "   " << getBeta() << '\n';
        }
    }else if (isSequence()){
        // sequence rules are simple, nothing to specify here
    }else if (isLocalPolynomial()){
        os << setw(L1) << "Order:" << "   " << getOrder() << '\n';
    }else if (isWavelet()){
        os << setw(L1) << "Order:" << "   " << getOrder() << '\n';
    }else{
        // empty grid, show nothing, just like the sequence grid
    }
    os << setw(L1) << "Acceleration:" << "  " << AccelerationMeta::getIOAccelerationString(acceleration->mode) << '\n';
    if (isLocalPolynomial() or isWavelet()){
        os << setw(L1) << "Flavor:" << "  " << (
            (acceleration->algorithm_select == AccelerationContext::algorithm_autoselect) ? "auto" :
                ((acceleration->algorithm_select == AccelerationContext::algorithm_dense) ? "dense" : "sparse")
                                               ) << "\n";
    }
    if (AccelerationMeta::isAccTypeGPU(acceleration->mode)){
        os << setw(L1) << "GPU:" << "  " << getGPUID() << '\n';
    }

    os << std::endl;
}

void TasmanianSparseGrid::writeAscii(std::ostream &ofs) const{
    ofs << "TASMANIAN SG " << getVersion() << '\n';
    ofs << "WARNING: do not edit this manually\n";
    if (isGlobal()){
        ofs << "global\n";
    }else if (isSequence()){
        ofs << "sequence\n";
    }else if (isLocalPolynomial()){
        ofs << "localpolynomial\n";
    }else if (isWavelet()){
        ofs << "wavelet\n";
    }else if (isFourier()){
        ofs << "fourier\n";
    }else{
        ofs << "empty\n";
    }
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.write(ofs, mode_ascii);
    }, base);
    if (domain_transform_a.size() != 0){
        ofs << "custom\n";
        ofs << std::scientific; ofs.precision(17);
        for(int j=0; j<getNumDimensions(); j++){
            ofs << domain_transform_a[j] << " " << domain_transform_b[j] << '\n';
       }
    }else{
        ofs << "canonical\n";
    }
    if (conformal_asin_power.size() != 0){
        ofs << "asinconformal\n";
        IO::writeVector<mode_ascii, IO::pad_line>(conformal_asin_power, ofs);
    }else{
        ofs << "nonconformal\n";
    }
    if (!llimits.empty()){
        ofs << "limited\n";
        IO::writeVector<mode_ascii, IO::pad_line>(llimits, ofs);
    }else{
        ofs << "unlimited\n";
    }
    if (using_dynamic_construction){
        ofs << "constructing\n";
        std::visit([&](auto const &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.writeConstructionData(ofs, mode_ascii);
        }, base);
    }else{
        ofs << "static\n";
    }
    ofs << "TASMANIAN SG end" << std::endl;
}
void TasmanianSparseGrid::writeBinary(std::ostream &ofs) const{
    const char *TSG = "TSG5"; // last char indicates version (update only if necessary, no need to sync with getVersionMajor())
    ofs.write(TSG, 4 * sizeof(char)); // mark Tasmanian files
    // use Integers to indicate grid types, empty 'e', global 'g', sequence 's', pwpoly 'p', wavelet 'w', Fourier 'f'
    if (isGlobal()){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'g');
    }else if (isSequence()){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 's');
    }else if (isLocalPolynomial()){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'p');
    }else if (isWavelet()){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'w');
    }else if (isFourier()){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'f');
    }else{
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'e');
    }
    std::visit([&](auto const &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.write(ofs, mode_binary);
    }, base);
    // domain transform: custom 'y', canonical: 'n'
    if (domain_transform_a.size() != 0){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'y');
        IO::writeVector<mode_binary, IO::pad_none>(domain_transform_a, ofs);
        IO::writeVector<mode_binary, IO::pad_none>(domain_transform_b, ofs);
    }else{
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'n');
    }
    // conformal transforms: none 'n', asin 'a'
    if (conformal_asin_power.size() != 0){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'a');
        IO::writeVector<mode_binary, IO::pad_none>(conformal_asin_power, ofs);
    }else{
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'n');
    }
    if (!llimits.empty()){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'y');
        IO::writeVector<mode_binary, IO::pad_none>(llimits, ofs);
    }else{
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'n');
    }
    if (using_dynamic_construction){
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'c');
        std::visit([&](auto const &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.writeConstructionData(ofs, mode_binary);
        }, base);
    }else{
        IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 's');
    }
    IO::writeNumbers<mode_binary, IO::pad_none>(ofs, 'e'); // E stands for END
}
void TasmanianSparseGrid::readAscii(std::istream &ifs){
    std::vector<double> new_domain_transform_a, new_domain_transform_b;
    std::vector<int> new_conformal_asin_power;
    std::vector<int> new_llimits;
    bool new_using_dynamic_construction = false;

    std::string T;
    std::string message = ""; // used in case there is an exception
    ifs >> T;  if (!(T.compare("TASMANIAN") == 0)){ throw std::runtime_error("ERROR: wrong file format, first word in not 'TASMANIAN'"); }
    ifs >> T;  if (!(T.compare("SG") == 0)){ throw std::runtime_error("ERROR: wrong file format, second word in not 'SG'"); }
    getline(ifs, T); T.erase(0,1);
    if (!(T.compare(getVersion()) == 0)){
        // grids with version prior to 3.0 are not supported
        size_t dec = T.find(".");
        if (dec == std::string::npos) throw std::runtime_error("ERROR: wrong file format, cannot read the version number");
        int vmajor = stoi(T.substr(0, dec));
        int vminor = stoi(T.substr(dec+1));
        if (vmajor < 3) throw std::runtime_error("ERROR: file formats from versions prior to 3.0 are not supported");
        if ((vmajor > getVersionMajor()) || ((vmajor == getVersionMajor()) && (vminor > getVersionMinor()))){
            message += "ERROR: using future file format " + std::to_string(vmajor) + ", Tasmanian cannot time-travel.";
            throw std::runtime_error(message);
        }
        // else: using file format older than current but newer than 3.0, it is supposed to work
    }
    getline(ifs, T); if (!(T.compare("WARNING: do not edit this manually") == 0)){ throw std::runtime_error("ERROR: wrong file format, missing warning message"); }
    ifs >> T;
    clear();
    grid_variant new_base = std::monostate{};
    if (T.compare("global") == 0){
        new_base.emplace<GridGlobal>(acceleration.get());
        readGridVersion5<IO::mode_ascii_type>(ifs, std::get_if<GridGlobal>(&new_base));
    }else if (T.compare("sequence") == 0){
        new_base.emplace<GridSequence>(acceleration.get());
        readGridVersion5<IO::mode_ascii_type>(ifs, std::get_if<GridSequence>(&new_base));
    }else if (T.compare("localpolynomial") == 0){
        new_base.emplace<GridLocalPolynomial>(acceleration.get());
        readGridVersion5<IO::mode_ascii_type>(ifs, std::get_if<GridLocalPolynomial>(&new_base));
    }else if (T.compare("wavelet") == 0){
        new_base.emplace<GridWavelet>(acceleration.get());
        readGridVersion5<IO::mode_ascii_type>(ifs, std::get_if<GridWavelet>(&new_base));
    }else if (T.compare("fourier") == 0){
        new_base.emplace<GridFourier>(acceleration.get());
        readGridVersion5<IO::mode_ascii_type>(ifs, std::get_if<GridFourier>(&new_base));
    }else if (T.compare("empty") != 0){
        throw std::runtime_error("ERROR: wrong file format, unknown grid type (or corrupt file)");
    }
    getline(ifs, T); // read an empty line
    getline(ifs, T);

    int const num_dimensions = std::visit([](auto const &g) -> int {
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
                return g.getNumDimensions();
            } else {
                return int{0};
            }
        }, new_base);

    bool reached_eof = false;
    if (T.compare("TASMANIAN SG end") == 0){ // version 3.0 did not include domain transform
        reached_eof = true;
    }else if (T.compare("custom") == 0){ // handle domain transform
        new_domain_transform_a.resize(num_dimensions);
        new_domain_transform_b.resize(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            ifs >> new_domain_transform_a[j] >> new_domain_transform_b[j];
        }
        getline(ifs, T);
    }else if (T.compare("canonical") != 0){ // canonical transform requires no action
        throw std::runtime_error("ERROR: wrong file format, domain unspecified");
    }
    if (!reached_eof){ // handle conformal maps, added in version 5.0
        getline(ifs, T);
        if (T.compare("asinconformal") == 0){
            new_conformal_asin_power = IO::readVector<IO::mode_ascii_type, int>(ifs, num_dimensions);
            getline(ifs, T);
        }else if (T.compare("TASMANIAN SG end") == 0){
            // for compatibility with version 4.0/4.1 and the missing conformal maps
            reached_eof = true;
        }else if (T.compare("nonconformal") != 0){
            throw std::runtime_error("ERROR: wrong file format, conformal mapping is unspecified");
        }
    }
    if (!reached_eof){ // handle level limits, added in version 5.1
        getline(ifs, T);
        if (T.compare("limited") == 0){
            new_llimits = IO::readVector<IO::mode_ascii_type, int>(ifs, num_dimensions);
            getline(ifs, T);
        }else if (T.compare("unlimited") == 0){
            new_llimits = std::vector<int>();
        }else if (T.compare("TASMANIAN SG end") == 0){
            reached_eof = true;
        }else{
            throw std::runtime_error("ERROR: wrong file format, did not specify level limits");
        }
    }
    if (!reached_eof){ // handles additional data for dynamic construction, added in version 7.0 (development 6.1)
        getline(ifs, T);
        if (T.compare("constructing") == 0){
            new_using_dynamic_construction = true;
            std::visit([&](auto &g){
                if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                    g.readConstructionData(ifs, mode_ascii);
            }, new_base);
            getline(ifs, T); // clear the final std::endl after reading the block
        }else if (T.compare("TASMANIAN SG end") == 0){
            reached_eof = true;
        }else if (T.compare("static") != 0){ // static construction requires no additional work
            throw std::runtime_error("ERROR: wrong file format, did not specify construction method");
        }
    }
    if (!reached_eof){
        getline(ifs, T);
        if (!(T.compare("TASMANIAN SG end") == 0)){
            throw std::runtime_error("ERROR: wrong file format, did not end with 'TASMANIAN SG end' (possibly corrupt file)");
        }
    }

    base = std::move(new_base);
    domain_transform_a = std::move(new_domain_transform_a);
    domain_transform_b = std::move(new_domain_transform_b);
    conformal_asin_power = std::move(new_conformal_asin_power);
    llimits = std::move(new_llimits);
    using_dynamic_construction = new_using_dynamic_construction;
}
void TasmanianSparseGrid::readBinary(std::istream &ifs){
    std::vector<double> new_domain_transform_a, new_domain_transform_b;
    std::vector<int> new_conformal_asin_power;
    std::vector<int> new_llimits;
    bool new_using_dynamic_construction = false;

    std::vector<char>  TSG(4);
    ifs.read(TSG.data(), 4*sizeof(char));
    if ((TSG[0] != 'T') || (TSG[1] != 'S') || (TSG[2] != 'G')){
        throw std::runtime_error("ERROR: wrong binary file format, first 3 bytes are not 'TSG'");
    }
    if (TSG[3] != '5'){
        throw std::runtime_error("ERROR: wrong binary file format, version number is not '5'");
    }
    clear();

    grid_variant new_base = std::monostate{};

    switch (IO::readNumber<IO::mode_binary_type, char>(ifs)){
        case 'g':
            new_base.emplace<GridGlobal>(acceleration.get());
            readGridVersion5<IO::mode_binary_type>(ifs, std::get_if<GridGlobal>(&new_base));
            break;
        case 's':
            new_base.emplace<GridSequence>(acceleration.get());
            readGridVersion5<IO::mode_binary_type>(ifs, std::get_if<GridSequence>(&new_base));
            break;
        case 'p':
            new_base.emplace<GridLocalPolynomial>(acceleration.get());
            readGridVersion5<IO::mode_binary_type>(ifs, std::get_if<GridLocalPolynomial>(&new_base));
            break;
        case 'w':
            new_base.emplace<GridWavelet>(acceleration.get());
            readGridVersion5<IO::mode_binary_type>(ifs, std::get_if<GridWavelet>(&new_base));
            break;
        case 'f':
            new_base.emplace<GridFourier>(acceleration.get());
            readGridVersion5<IO::mode_binary_type>(ifs, std::get_if<GridFourier>(&new_base));
            break;
        case 'e':
            // monostate already set above
            break;
        default:
            throw std::runtime_error("ERROR: wrong binary file format, unknown grid type");
    };

    int const num_dimensions = std::visit([](auto const &g) -> int {
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>) {
                return g.getNumDimensions();
            } else {
                return int{0};
            }
        }, new_base);

    char flag = IO::readNumber<IO::mode_binary_type, char>(ifs);
    if (flag == 'y'){
        new_domain_transform_a = IO::readVector<IO::mode_binary_type, double>(ifs, num_dimensions);
        new_domain_transform_b = IO::readVector<IO::mode_binary_type, double>(ifs, num_dimensions);
    }else if (flag != 'n'){
        throw std::runtime_error("ERROR: wrong binary file format, wrong domain type");
    }

    flag = IO::readNumber<IO::mode_binary_type, char>(ifs); // conformal domain transform?
    if (flag == 'a'){
        new_conformal_asin_power = IO::readVector<IO::mode_binary_type, int>(ifs, num_dimensions);
    }else if (flag != 'n'){
        throw std::runtime_error("ERROR: wrong binary file format, wrong conformal transform type");
    }

    flag = IO::readNumber<IO::mode_binary_type, char>(ifs); // limits
    if (flag == 'y'){
        new_llimits = IO::readVector<IO::mode_binary_type, int>(ifs, num_dimensions);
    }else if (flag != 'n'){
        throw std::runtime_error("ERROR: wrong binary file format, wrong level limits");
    }

    bool reached_eof = false;
    flag = IO::readNumber<IO::mode_binary_type, char>(ifs); // construction data
    if (flag == 'c'){ // handles additional data for dynamic construction, added in version 7.0 (development 6.1)
        new_using_dynamic_construction = true;
        std::visit([&](auto &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.readConstructionData(ifs, mode_binary);
        }, new_base);
    }else if (flag == 'e'){
        reached_eof = true;
    }else if (flag != 's'){
        throw std::runtime_error("ERROR: wrong binary file format, wrong construction method specified");
    }

    if (!reached_eof){
        if (IO::readNumber<IO::mode_binary_type, char>(ifs) != 'e'){
            throw std::runtime_error("ERROR: wrong binary file format, did not reach correct end of Tasmanian block");
        }
    }

    base = std::move(new_base);
    domain_transform_a = std::move(new_domain_transform_a);
    domain_transform_b = std::move(new_domain_transform_b);
    conformal_asin_power = std::move(new_conformal_asin_power);
    llimits = std::move(new_llimits);
    using_dynamic_construction = new_using_dynamic_construction;
}

void TasmanianSparseGrid::enableAcceleration(TypeAcceleration acc){
    // if gpu acceleration has been disabled, then reset the domain and cache
    // note that this method cannot possibly change the gpu ID and switching
    // between variations of GPU accelerations on the same device will keep the same cache
    AccelerationContext::ChangeType change = acceleration->testEnable(acc, acceleration->device);
    std::visit([&](auto &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.updateAccelerationData(change);
    }, base);
    if (change == AccelerationContext::change_gpu_device)
        acc_domain.clear();
    acceleration->enable(acc, acceleration->device);
}

void TasmanianSparseGrid::enableAcceleration(TypeAcceleration acc, int new_gpu_id){
    AccelerationContext::ChangeType change = acceleration->testEnable(acc, new_gpu_id);
    std::visit([&](auto &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.updateAccelerationData(change);
    }, base);
    if (change == AccelerationContext::change_gpu_device)
        acc_domain.clear();
    acceleration->enable(acc, new_gpu_id);
}
void TasmanianSparseGrid::favorSparseAcceleration(bool favor){
    AccelerationContext::ChangeType change = acceleration->favorSparse(favor);
    std::visit([&](auto &g){
        if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
            g.updateAccelerationData(change);
    }, base);
}

#ifdef Tasmanian_ENABLE_CUDA
void TasmanianSparseGrid::setCuBlasHandle(void *handle){
    if (acceleration->on_gpu()){
        acceleration->engine->setCuBlasHandle(handle);
    }else{
        throw std::runtime_error("setCuBlasHandle() called with non-GPU acceleration mode.");
    }
}
void TasmanianSparseGrid::setCuSparseHandle(void *handle){
    if (acceleration->on_gpu()){
        acceleration->engine->setCuSparseHandle(handle);
    }else{
        throw std::runtime_error("setCuSparseHandle() called with non-GPU acceleration mode.");
    }
}
void TasmanianSparseGrid::setCuSolverHandle(void *handle){
    if (acceleration->on_gpu()){
        acceleration->engine->setCuSolverDnHandle(handle);
    }else{
        throw std::runtime_error("setCuSolverHandle() called with non-GPU acceleration mode.");
    }
}
#else
void TasmanianSparseGrid::setCuBlasHandle(void*){
    throw std::runtime_error("setCuBlasHandle() requires Tasmanian to be build with the CUDA backend.");
}
void TasmanianSparseGrid::setCuSparseHandle(void*){
    throw std::runtime_error("setCuSparseHandle() requires Tasmanian to be build with the CUDA backend.");
}
void TasmanianSparseGrid::setCuSolverHandle(void*){
    throw std::runtime_error("setCuSolverHandle() requires Tasmanian to be build with the CUDA backend.");
}
#endif

#ifdef Tasmanian_ENABLE_HIP
void TasmanianSparseGrid::setRocBlasHandle(void *handle){
    if (acceleration->on_gpu()){
        acceleration->engine->setRocBlasHandle(handle);
    }else{
        throw std::runtime_error("setRocBlasHandle() called with non-GPU acceleration mode.");
    }
}
void TasmanianSparseGrid::setRocSparseHandle(void *handle){
    if (acceleration->on_gpu()){
        acceleration->engine->setRocSparseHandle(handle);
    }else{
        throw std::runtime_error("setRocSparseHandle() called with non-GPU acceleration mode.");
    }
}
#else
void TasmanianSparseGrid::setRocBlasHandle(void*){
    throw std::runtime_error("setRocBlasHandle() requires Tasmanian to be build with the HIP/ROCm backend.");
}
void TasmanianSparseGrid::setRocSparseHandle(void*){
    throw std::runtime_error("setRocSparseHandle() requires Tasmanian to be build with the HIP/ROCm backend.");
}
#endif

#ifdef Tasmanian_ENABLE_DPCPP
void TasmanianSparseGrid::setSycleQueue(void *queue){
    if (acceleration->on_gpu()){
        acceleration->engine->setSyclQueue(queue);
    }else{
        throw std::runtime_error("setSyclQueue() called with non-GPU acceleration mode.");
    }
}
#else
void TasmanianSparseGrid::setSycleQueue(void*){
    throw std::runtime_error("setSyclQueue() requires Tasmanian to be build with the DPC++/SYCL backend.");
}
#endif

bool TasmanianSparseGrid::isAccelerationAvailable(TypeAcceleration acc){
    #ifdef Tasmanian_ENABLE_GPU
    if (acc == accel_gpu_default) return true;
    #endif
    return (acc == AccelerationMeta::getAvailableFallback(acc));
}

void TasmanianSparseGrid::setGPUID(int new_gpu_id){
    if (new_gpu_id != acceleration->device){
        AccelerationContext::ChangeType change = acceleration->testEnable(acceleration->mode, new_gpu_id);
        std::visit([&](auto &g){
            if constexpr (not std::is_same_v<std::decay_t<decltype(g)>, std::monostate>)
                g.updateAccelerationData(change);
        }, base);
        #ifdef Tasmanian_ENABLE_GPU
        if (change == AccelerationContext::change_gpu_device)
            acc_domain.clear();
        #endif
        acceleration->enable(acceleration->mode, new_gpu_id);
    }
}

int TasmanianSparseGrid::getGPUMemory(int gpu){
    if ((gpu < 0) || (gpu >= AccelerationMeta::getNumGpuDevices())) return 0;
    return (int) (AccelerationMeta::getTotalGPUMemory(gpu) / 1048576);
}
std::string TasmanianSparseGrid::getGPUName(int gpu){
    return AccelerationMeta::getGpuDeviceName(gpu);
}

}

#endif
