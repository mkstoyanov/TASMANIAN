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

#ifndef __TASMANIAN_SPARSE_GRID_LPOLY_CPP
#define __TASMANIAN_SPARSE_GRID_LPOLY_CPP

#include "tsgGridLocalPolynomial.hpp"
#include "tsgHiddenExternals.hpp"

namespace TasGrid{

GridLocalPolynomial::GridLocalPolynomial() : order(1), top_level(0), sparse_affinity(0)  {}
GridLocalPolynomial::~GridLocalPolynomial(){}

void GridLocalPolynomial::reset(bool clear_rule){
    clearAccelerationData();
    num_dimensions = num_outputs = top_level = 0;
    points = MultiIndexSet();
    needed = MultiIndexSet();
    values = StorageSet();
    if (clear_rule){ rule = std::unique_ptr<BaseRuleLocalPolynomial>(); order = 1; }
    parents = Data2D<int>();
    sparse_affinity = 0;
    surpluses.clear();
}
template<class T> std::unique_ptr<T> make_unique_ptr(){ return std::unique_ptr<T>(new T()); } // in C++14 this is called std::make_unique()
void GridLocalPolynomial::makeRule(TypeOneDRule crule){
    if (crule == rule_localp){
        rule = make_unique_ptr<templRuleLocalPolynomial<rule_localp, false>>();
    }else if (crule == rule_semilocalp){
        rule = make_unique_ptr<templRuleLocalPolynomial<rule_semilocalp, false>>();
    }else if (crule == rule_localp0){
        rule = make_unique_ptr<templRuleLocalPolynomial<rule_localp0, false>>();
    }else if (crule == rule_localpb){
        rule = make_unique_ptr<templRuleLocalPolynomial<rule_localpb, false>>();
    }
    if (order == 0){
        rule = make_unique_ptr<templRuleLocalPolynomial<rule_localp, true>>();
    }
    rule->setMaxOrder(order);
}

template<bool iomode> void GridLocalPolynomial::write(std::ostream &os) const{
    if (iomode == mode_ascii){ os << std::scientific; os.precision(17); }
    IO::writeNumbers<iomode, IO::pad_line>(os, num_dimensions, num_outputs, order, top_level);
    IO::writeRule<iomode>(rule->getType(), os);
    IO::writeFlag<iomode, IO::pad_auto>(!points.empty(), os);
    if (!points.empty()) points.write<iomode>(os);
    if (iomode == mode_ascii){ // backwards compatible: surpluses and needed, or needed and surpluses
        IO::writeFlag<iomode, IO::pad_auto>((surpluses.getNumStrips() != 0), os);
        if (surpluses.getNumStrips() != 0) IO::writeVector<iomode, IO::pad_line>(surpluses.getVector(), os);
        IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
        if (!needed.empty()) needed.write<iomode>(os);
    }else{
        IO::writeFlag<iomode, IO::pad_auto>(!needed.empty(), os);
        if (!needed.empty()) needed.write<iomode>(os);
        IO::writeFlag<iomode, IO::pad_auto>((surpluses.getNumStrips() != 0), os);
        if (surpluses.getNumStrips() != 0) IO::writeVector<iomode, IO::pad_line>(surpluses.getVector(), os);
    }
    IO::writeFlag<iomode, IO::pad_auto>((parents.getNumStrips() != 0), os);
    if (parents.getNumStrips() != 0) IO::writeVector<iomode, IO::pad_line>(parents.getVector(), os);

    IO::writeNumbers<iomode, IO::pad_rspace>(os, (int) roots.size());
    if (roots.size() > 0){ // the tree is empty, can happend when using dynamic construction
        IO::writeVector<iomode, IO::pad_line>(roots, os);
        IO::writeVector<iomode, IO::pad_line>(pntr, os);
        IO::writeVector<iomode, IO::pad_line>(indx, os);
    }

    if (num_outputs > 0) values.write<iomode>(os);
}

template<bool iomode> void GridLocalPolynomial::read(std::istream &is){
    reset();
    num_dimensions = IO::readNumber<iomode, int>(is);
    num_outputs = IO::readNumber<iomode, int>(is);
    order = IO::readNumber<iomode, int>(is);
    top_level = IO::readNumber<iomode, int>(is);
    TypeOneDRule crule = IO::readRule<iomode>(is);
    makeRule(crule);

    if (IO::readFlag<iomode>(is)) points.read<iomode>(is);
    if (iomode == mode_ascii){ // backwards compatible: surpluses and needed, or needed and surpluses
        if (IO::readFlag<iomode>(is))
            surpluses = IO::readData2D<iomode, double>(is, num_outputs, points.getNumIndexes());
        if (IO::readFlag<iomode>(is)) needed.read<iomode>(is);
    }else{
        if (IO::readFlag<iomode>(is)) needed.read<iomode>(is);
        if (IO::readFlag<iomode>(is))
            surpluses = IO::readData2D<iomode, double>(is, num_outputs, points.getNumIndexes());
    }
    if (IO::readFlag<iomode>(is))
        parents = IO::readData2D<iomode, int>(is, rule->getMaxNumParents() * num_dimensions, points.getNumIndexes());

    size_t num_points = (size_t) ((points.empty()) ? needed.getNumIndexes() : points.getNumIndexes());
    roots.resize((size_t) IO::readNumber<iomode, int>(is));
    if (roots.size() > 0){
        IO::readVector<iomode>(is, roots);
        pntr.resize(num_points + 1);
        IO::readVector<iomode>(is, pntr);
        if (pntr[num_points] > 0){
            indx.resize((size_t) pntr[num_points]);
            IO::readVector<iomode>(is, indx);
        }else{
            indx.resize(1);
            indx[0] = IO::readNumber<iomode, int>(is); // there is a special case when the grid has only one point without any children
        }
    }

    if (num_outputs > 0) values.read<iomode>(is);
}

template void GridLocalPolynomial::write<mode_ascii>(std::ostream &) const;
template void GridLocalPolynomial::write<mode_binary>(std::ostream &) const;
template void GridLocalPolynomial::read<mode_ascii>(std::istream &);
template void GridLocalPolynomial::read<mode_binary>(std::istream &);

void GridLocalPolynomial::makeGrid(int cnum_dimensions, int cnum_outputs, int depth, int corder, TypeOneDRule crule, const std::vector<int> &level_limits){
    reset();
    num_dimensions = cnum_dimensions;
    num_outputs = cnum_outputs;
    order = corder;

    TypeOneDRule effective_rule = ((crule == rule_semilocalp) && ((order == -1) || (order > 1))) ? rule_semilocalp : rule_localp; // semi-localp of order 1 is identical to localp
    if (crule == rule_localp0) effective_rule = rule_localp0;
    if (crule == rule_localpb) effective_rule = rule_localpb;

    makeRule(effective_rule);

    MultiIndexSet tensors = MultiIndexManipulations::selectTensors((size_t) num_dimensions, depth, type_level, [&](int i) -> int{ return i; }, std::vector<int>(), level_limits);

    needed = MultiIndexManipulations::generateNestedPoints(num_threads, tensors, [&](int l) -> int{ return rule->getNumPoints(l); });

    buildTree();

    if (num_outputs == 0){
        points = std::move(needed);
        needed = MultiIndexSet();
        parents = HierarchyManipulations::computeDAGup(points, rule.get());
    }else{
        values.resize(num_outputs, needed.getNumIndexes());
    }
}

void GridLocalPolynomial::copyGrid(const GridLocalPolynomial *pwpoly, int ibegin, int iend){
    num_dimensions = pwpoly->num_dimensions;
    num_outputs    = iend - ibegin;
    points = pwpoly->points;
    needed = pwpoly->needed;

    order     = pwpoly->order;
    top_level = pwpoly->top_level;

    surpluses = (num_outputs == pwpoly->num_outputs) ? pwpoly->surpluses : pwpoly->surpluses.splitData(ibegin, iend);
    values    = (num_outputs == pwpoly->num_outputs) ? pwpoly->values : pwpoly->values.splitValues(ibegin, iend);
    parents   = pwpoly->parents;

    roots = pwpoly->roots;
    pntr  = pwpoly->pntr;
    indx  = pwpoly->indx;

    makeRule(pwpoly->rule->getType());

    sparse_affinity = pwpoly->sparse_affinity;

    if (pwpoly->dynamic_values){
        dynamic_values = std::unique_ptr<SimpleConstructData>(new SimpleConstructData(*pwpoly->dynamic_values));
        if (num_outputs != pwpoly->num_outputs) dynamic_values->restrictData(ibegin, iend);
    }
}

GridLocalPolynomial::GridLocalPolynomial(int cnum_dimensions, int cnum_outputs, int corder, TypeOneDRule crule,
                                         std::vector<int> &&pnts, std::vector<double> &&vals, std::vector<double> &&surps) :
                                         order(corder), sparse_affinity(0){

    num_dimensions = cnum_dimensions;
    num_outputs = cnum_outputs;

    makeRule(crule);

    points = MultiIndexSet(num_dimensions, std::vector<int>(pnts));
    values.resize(num_outputs, points.getNumIndexes());
    values.setValues(std::vector<double>(vals));
    surpluses = Data2D<double>(num_outputs, points.getNumIndexes(), std::vector<double>(surps));

    buildTree();
}

void GridLocalPolynomial::getLoadedPoints(double *x) const{
    int num_points = points.getNumIndexes();
    Utils::Wrapper2D<double> split(num_dimensions, x);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<num_points; i++){
        const int *p = points.getIndex(i);
        double *xx = split.getStrip(i);
        for(int j=0; j<num_dimensions; j++){
            xx[j] = rule->getNode(p[j]);
        }
    }
}
void GridLocalPolynomial::getNeededPoints(double *x) const{
    int num_points = needed.getNumIndexes();
    Utils::Wrapper2D<double> split(num_dimensions, x);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<num_points; i++){
        const int *p = needed.getIndex(i);
        double *xx = split.getStrip(i);
        for(int j=0; j<num_dimensions; j++){
            xx[j] = rule->getNode(p[j]);
        }
    }
}
void GridLocalPolynomial::getPoints(double *x) const{
    if (points.empty()){ getNeededPoints(x); }else{ getLoadedPoints(x); }
}

void GridLocalPolynomial::evaluate(const double x[], double y[]) const{
    std::fill_n(y, num_outputs, 0.0);
    std::vector<int> sindx; // dummy variables, never references in mode 0 below
    std::vector<double> svals;
    walkTree<0>(points, x, sindx, svals, y);
}
void GridLocalPolynomial::evaluateBatch(const double x[], int num_x, double y[]) const{
    if (num_x == 1){ evaluate(x, y); return; }
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_outputs, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++)
        evaluate(xwrap.getStrip(i), ywrap.getStrip(i));
}

#ifdef Tasmanian_ENABLE_BLAS
void GridLocalPolynomial::evaluateBlas(const double x[], int num_x, double y[]) const{
    if ((sparse_affinity == 1) || ((sparse_affinity == 0) && (num_outputs <= 1024))){
        evaluateBatch(x, num_x, y);
        return;
    }

    std::vector<int> sindx, spntr;
    std::vector<double> svals;
    buildSpareBasisMatrix(x, num_x, 32, spntr, sindx, svals); // build sparse matrix corresponding to x

    int num_points = points.getNumIndexes();
    double nnz = (double) spntr[num_x];
    double total_size = ((double) num_x) * ((double) num_points);

    if ((sparse_affinity == -1) || ((sparse_affinity == 0) && (nnz / total_size > 0.1))){
        // potentially wastes a lot of memory
        Data2D<double> A(num_points, num_x, 0.0);
        for(int i=0; i<num_x; i++){
            double *row = A.getStrip(i);
            for(int j=spntr[i]; j<spntr[i+1]; j++) row[sindx[j]] = svals[j];
        }
        TasBLAS::denseMultiply(num_outputs, num_x, num_points, 1.0, surpluses.getStrip(0), A.getStrip(0), 0.0, y);
    }else{
        Utils::Wrapper2D<double> ywrap(num_outputs, y);
        #pragma omp parallel for
        for(int i=0; i<num_x; i++){
            double *this_y = ywrap.getStrip(i);
            std::fill(this_y, this_y + num_outputs, 0.0);
            for(int j=spntr[i]; j<spntr[i+1]; j++){
                double v = svals[j];
                const double *s = surpluses.getStrip(sindx[j]);
                for(int k=0; k<num_outputs; k++) this_y[k] += v * s[k];
            }
        }
    }
}
#endif

#ifdef Tasmanian_ENABLE_CUDA
void GridLocalPolynomial::loadNeededPointsCuda(CudaEngine *engine, const double *vals){
    updateValues(vals);

    std::vector<int> levels = HierarchyManipulations::computeLevels(points, rule.get());

    std::vector<Data2D<int>> lpnts = HierarchyManipulations::splitByLevels((size_t) num_dimensions, points.getVector(), levels);
    std::vector<Data2D<double>> lvals = HierarchyManipulations::splitByLevels((size_t) num_outputs, values.getVector(), levels);

    Data2D<double> allx(num_dimensions, points.getNumIndexes());
    getPoints(allx.getVector().data());

    std::vector<Data2D<double>> lx = HierarchyManipulations::splitByLevels((size_t) num_dimensions, allx.getVector(), levels);

    MultiIndexSet cumulative_poitns((size_t) num_dimensions, std::move(lpnts[0].getVector()));

    StorageSet cumulative_surpluses;
    cumulative_surpluses.resize(num_outputs, cumulative_poitns.getNumIndexes());
    cumulative_surpluses.setValues(std::move(lvals[0].getVector()));

    for(size_t l = 1; l < lpnts.size(); l++){ // loop over the levels
        // note that level_points.getNumIndexes() == lx[l].getNumStrips() == lvals[l].getNumStrips()
        MultiIndexSet level_points(num_dimensions, std::move(lpnts[l].getVector()));

        GridLocalPolynomial upper_grid(num_dimensions, num_outputs, order, getRule(),
                                       std::vector<int>(cumulative_poitns.getVector()), // copy cumulative_poitns
                                       std::vector<double>(Utils::size_mult(num_outputs, cumulative_poitns.getNumIndexes())),  // dummy values, will not be read or used
                                       std::vector<double>(cumulative_surpluses.getVector())); // copy the cumulative_surpluses
        upper_grid.sparse_affinity = sparse_affinity;

        Data2D<double> upper_evaluate(num_outputs, level_points.getNumIndexes());
        int batch_size = 20000; // needs tuning
        if (useDense()){ // dense uses lots of memory, try to keep it contained to about 4GB
            batch_size = 536870912 / upper_grid.getNumPoints() - 2 * (num_outputs + num_dimensions);
            if (batch_size < 100) batch_size = 100; // use at least 100 points
        }

        for(int i=0; i<level_points.getNumIndexes(); i += batch_size)
            upper_grid.evaluateCuda(engine, lx[l].getStrip(i), std::min(batch_size, level_points.getNumIndexes() - i), upper_evaluate.getStrip(i));

        double *level_surps = lvals[l].getStrip(0); // maybe use BLAS here
        const double *uv = upper_evaluate.getStrip(0);
        for(size_t i=0, s = upper_evaluate.getVector().size(); i < s; i++) level_surps[i] -= uv[i];

        cumulative_surpluses.addValues(cumulative_poitns, level_points, level_surps);
        cumulative_poitns.addMultiIndexSet(level_points);
    }

    surpluses = Data2D<double>(num_outputs, points.getNumIndexes(), std::move(cumulative_surpluses.getVector()));
}
void GridLocalPolynomial::evaluateCudaMixed(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    loadCudaSurpluses<double>();

    std::vector<int> sindx, spntr;
    std::vector<double> svals;

    if (num_x > 1){
        buildSpareBasisMatrix(x, num_x, 32, spntr, sindx, svals);
    }else{
        walkTree<2>(points, x, sindx, svals, nullptr);
    }
    engine->sparseMultiply(num_outputs, num_x, points.getNumIndexes(), 1.0, cuda_cache->surpluses, spntr, sindx, svals, y);
}
void GridLocalPolynomial::evaluateCuda(CudaEngine *engine, const double x[], int num_x, double y[]) const{
    if ((order == -1) || (order > 2) || (num_x == 1)){
        // GPU evaluations are available only for order 0, 1, and 2. Cubic will come later, but higher order will not be supported.
        // cannot use GPU to accelerate the evaluation of a single vector
        evaluateCudaMixed(engine, x, num_x, y);
        return;
    }
    CudaVector<double> gpu_x(num_dimensions, num_x, x), gpu_result(num_x, num_outputs);
    evaluateBatchGPU(engine, gpu_x.data(), num_x, gpu_result.data());
    gpu_result.unload(y);
}
template<typename T> void GridLocalPolynomial::evaluateBatchGPUtempl(CudaEngine *engine, const T gpu_x[], int cpu_num_x, T gpu_y[]) const{
    if ((order == -1) || (order > 2)) throw std::runtime_error("ERROR: GPU evaluations are availabe only for local polynomial grid with order 0, 1, and 2");
    loadCudaSurpluses<T>();
    int num_points = points.getNumIndexes();

    if (useDense()){
        CudaVector<T> gpu_basis(cpu_num_x, num_points);
        evaluateHierarchicalFunctionsGPU(gpu_x, cpu_num_x, gpu_basis.data());

        engine->denseMultiply(num_outputs, cpu_num_x, num_points, 1.0, getCudaCache(static_cast<T>(0.0))->surpluses, gpu_basis, 0.0, gpu_y);
    }else{
        CudaVector<int> gpu_spntr, gpu_sindx;
        CudaVector<T> gpu_svals;
        buildSparseBasisMatrixGPU(gpu_x, cpu_num_x, gpu_spntr, gpu_sindx, gpu_svals);

        engine->sparseMultiply(num_outputs, cpu_num_x, num_points, 1.0, getCudaCache(static_cast<T>(0.0))->surpluses, gpu_spntr, gpu_sindx, gpu_svals, 0.0, gpu_y);
    }
}
void GridLocalPolynomial::evaluateBatchGPU(CudaEngine *engine, const double gpu_x[], int cpu_num_x, double gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
void GridLocalPolynomial::evaluateBatchGPU(CudaEngine *engine, const float gpu_x[], int cpu_num_x, float gpu_y[]) const{
    evaluateBatchGPUtempl(engine, gpu_x, cpu_num_x, gpu_y);
}
void GridLocalPolynomial::evaluateHierarchicalFunctionsGPU(const double gpu_x[], int cpu_num_x, double *gpu_y) const{
    loadCudaBasis<double>();
    int num_points = getNumPoints();
    TasCUDA::devalpwpoly(order, rule->getType(), num_dimensions, cpu_num_x, num_points, gpu_x, cuda_cache->nodes.data(), cuda_cache->support.data(), gpu_y);
}
void GridLocalPolynomial::buildSparseBasisMatrixGPU(const double gpu_x[], int cpu_num_x, CudaVector<int> &gpu_spntr, CudaVector<int> &gpu_sindx, CudaVector<double> &gpu_svals) const{
    loadCudaBasis<double>();
    loadCudaHierarchy<double>();
    TasCUDA::devalpwpoly_sparse(order, rule->getType(), num_dimensions, cpu_num_x, getNumPoints(), gpu_x, cuda_cache->nodes, cuda_cache->support,
                                cuda_cache->hpntr, cuda_cache->hindx, cuda_cache->hroots, gpu_spntr, gpu_sindx, gpu_svals);
}
void GridLocalPolynomial::evaluateHierarchicalFunctionsGPU(const float gpu_x[], int cpu_num_x, float *gpu_y) const{
    loadCudaBasis<float>();
    int num_points = getNumPoints();
    TasCUDA::devalpwpoly(order, rule->getType(), num_dimensions, cpu_num_x, num_points, gpu_x, cuda_cachef->nodes.data(), cuda_cachef->support.data(), gpu_y);
}
void GridLocalPolynomial::buildSparseBasisMatrixGPU(const float gpu_x[], int cpu_num_x, CudaVector<int> &gpu_spntr, CudaVector<int> &gpu_sindx, CudaVector<float> &gpu_svals) const{
    loadCudaBasis<float>();
    loadCudaHierarchy<float>();
    TasCUDA::devalpwpoly_sparse(order, rule->getType(), num_dimensions, cpu_num_x, getNumPoints(), gpu_x, cuda_cachef->nodes, cuda_cachef->support,
                                cuda_cachef->hpntr, cuda_cachef->hindx, cuda_cachef->hroots, gpu_spntr, gpu_sindx, gpu_svals);
}
template<typename T> void GridLocalPolynomial::loadCudaBasis() const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaLocalPolynomialData<T>>();
    if (!ccache->nodes.empty()) return;

    Data2D<double> cpu_nodes(num_dimensions, getNumPoints());
    getPoints(cpu_nodes.getStrip(0));
    ccache->nodes.load(cpu_nodes.getVector());

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    ccache->support.load(
        [&](void)->Data2D<T>{
            if (rule->getType() == rule_localp){
                switch(order){
                case 0: return encodeSupportForGPU<0, rule_localp, T>(work);
                case 2: return encodeSupportForGPU<2, rule_localp, T>(work);
                default:
                    return encodeSupportForGPU<1, rule_localp, T>(work);
                }
            }else if (rule->getType() == rule_semilocalp){
                return encodeSupportForGPU<2, rule_semilocalp, T>(work);
            }else if (rule->getType() == rule_localpb){
                switch(order){
                case 2: return encodeSupportForGPU<2, rule_localpb, T>(work);
                default:
                    return encodeSupportForGPU<1, rule_localpb, T>(work);
                }
            }else{
                switch(order){
                case 2: return encodeSupportForGPU<2, rule_localp0, T>(work);
                default:
                    return encodeSupportForGPU<1, rule_localp0, T>(work);
                }
            }
        }().getVector());
}
void GridLocalPolynomial::clearCudaBasisHierarchy(){
    if (cuda_cache) cuda_cache->clearBasisHierarchy();
    if (cuda_cachef) cuda_cachef->clearBasisHierarchy();
}
template<typename T> void GridLocalPolynomial::loadCudaHierarchy() const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaLocalPolynomialData<T>>();
    if (!ccache->hpntr.empty()) return;

    ccache->hpntr.load(pntr);
    ccache->hindx.load(indx);
    ccache->hroots.load(roots);
}
template<typename T> void GridLocalPolynomial::loadCudaSurpluses() const{
    auto& ccache = getCudaCache(static_cast<T>(0.0));
    if (!ccache) ccache = std::make_unique<CudaLocalPolynomialData<T>>();
    if (ccache->surpluses.size() != 0) return;
    ccache->surpluses.load(surpluses.getVector());
}
void GridLocalPolynomial::clearCudaSurpluses(){
    if (cuda_cache) cuda_cache->surpluses.clear();
    if (cuda_cachef) cuda_cachef->surpluses.clear();
}
#endif


void GridLocalPolynomial::updateValues(double const *vals){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaSurpluses();
    #endif
    if (needed.empty()){
        values.setValues(vals);
    }else{
        #ifdef Tasmanian_ENABLE_CUDA
        clearCudaBasisHierarchy();
        #endif
        if (points.empty()){ // initial grid, just relabel needed as points (loaded)
            values.setValues(vals);
            points = std::move(needed);
            needed = MultiIndexSet();
        }else{ // merge needed and points
            values.addValues(points, needed, vals);
            points.addSortedIndexes(needed.getVector());
            needed = MultiIndexSet();
            buildTree();
        }
    }
}
void GridLocalPolynomial::loadNeededPoints(const double *vals){
    updateValues(vals);
    recomputeSurpluses();
}
void GridLocalPolynomial::mergeRefinement(){
    if (needed.empty()) return; // nothing to do
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaSurpluses();
    #endif
    int num_all_points = getNumLoaded() + getNumNeeded();
    size_t num_vals = ((size_t) num_all_points) * ((size_t) num_outputs);
    values.setValues(std::vector<double>(num_vals, 0.0));
    if (points.empty()){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        points.addMultiIndexSet(needed);
        needed = MultiIndexSet();
        buildTree();
    }
    surpluses.resize(num_outputs, num_all_points);
    surpluses.fill(0.0);
}

void GridLocalPolynomial::beginConstruction(){
    dynamic_values = std::unique_ptr<SimpleConstructData>(new SimpleConstructData);
    if (points.empty()){
        dynamic_values->initial_points = std::move(needed);
        needed = MultiIndexSet();
        roots.clear();
        pntr.clear();
        indx.clear();
    }
}
void GridLocalPolynomial::writeConstructionData(std::ostream &os, bool iomode) const{
    if (iomode == mode_ascii) dynamic_values->write<mode_ascii>(os); else dynamic_values->write<mode_binary>(os);
}
void GridLocalPolynomial::readConstructionData(std::istream &is, bool iomode){
    if (iomode == mode_ascii) dynamic_values = readSimpleConstructionData<mode_ascii>(num_dimensions, num_outputs, is);
    else dynamic_values = readSimpleConstructionData<mode_binary>(num_dimensions, num_outputs, is);
}
std::vector<double> GridLocalPolynomial::getCandidateConstructionPoints(double tolerance, TypeRefinement criteria, int output,
                                                                        std::vector<int> const &level_limits, double const *scale_correction){
    // combine the initial points with negative weights and the refinement candidates with surplus weights (no need to normalize, the sort uses relative values)
    MultiIndexSet refine_candidates = getRefinementCanidates(tolerance, criteria, output, level_limits, scale_correction);
    MultiIndexSet new_points = (dynamic_values->initial_points.empty()) ? std::move(refine_candidates) : refine_candidates.diffSets(dynamic_values->initial_points);

    // compute the weights for the new_points points
    std::vector<double> norm = getNormalization();

    int active_outputs = (output == -1) ? num_outputs : 1;
    Utils::Wrapper2D<double const> scale(active_outputs, scale_correction);
    std::vector<double> default_scale;
    if (scale_correction == nullptr){ // if no scale provided, assume default 1.0
        default_scale = std::vector<double>(Utils::size_mult(active_outputs, points.getNumIndexes()), 1.0);
        scale = Utils::Wrapper2D<double const>(active_outputs, default_scale.data());
    }

    auto getDominantSurplus = [&](int i)-> double{
        double dominant = 0.0;
        const double *s = surpluses.getStrip(i);
        const double *c = scale.getStrip(i);
        if (output == -1){
            for(int k=0; k<num_outputs; k++) dominant = std::max(dominant, c[k] * std::abs(s[k]) / norm[k]);
        }else{
            dominant = c[0] * std::abs(s[output]) / norm[output];
        }
        return dominant;
    };

    std::vector<double> refine_weights(new_points.getNumIndexes());

    #pragma omp parallel for
    for(int i=0; i<new_points.getNumIndexes(); i++){
        double weight = 0.0;
        std::vector<int> p(new_points.getIndex(i), new_points.getIndex(i) + num_dimensions); // get the point

        HierarchyManipulations::touchAllImmediateRelatives(p, points, rule.get(),
                                                            [&](int relative)->void{ weight = std::max(weight, getDominantSurplus(relative)); });
        refine_weights[i] = weight; // those will be inverted
    }

    // if using stable refinement, ensure the weight of the parents is never less than the children
    if (!new_points.empty() && ((criteria == refine_parents_first) || (criteria == refine_fds))){
        auto rlevels = HierarchyManipulations::computeLevels(new_points, rule.get());
        auto split = HierarchyManipulations::splitByLevels((size_t) num_dimensions, new_points.getVector(), rlevels);
        for(auto is = split.rbegin(); is != split.rend(); is++){
            for(int i=0; i<is->getNumStrips(); i++){
                std::vector<int> parent(is->getStrip(i), is->getStrip(i) + num_dimensions);
                double correction = refine_weights[new_points.getSlot(parent)]; // will never be missing
                for(auto &p : parent){
                    int r = p;
                    p = rule->getParent(r);
                    int ip = (p == -1) ? -1 : new_points.getSlot(parent); // if parent is among the refined
                    if (ip != -1) refine_weights[ip] += correction;
                    p = rule->getStepParent(r);
                    ip = (p == -1) ? -1 : new_points.getSlot(parent); // if parent is among the refined
                    if (ip != -1) refine_weights[ip] += correction;
                    p = r;
                }
            }
        }
    }else if (!new_points.empty() && (criteria == refine_stable)){
        // stable refinement, ensure that if level[i] < level[j] then weight[i] > weight[j]
        auto rlevels = HierarchyManipulations::computeLevels(new_points, rule.get());
        auto split = HierarchyManipulations::splitByLevels((size_t) num_dimensions, new_points.getVector(), rlevels);
        double max_weight = 0.0;
        for(auto is = split.rbegin(); is != split.rend(); is++){ // loop backwards in levels
            double correction = max_weight;
            for(int i=0; i<is->getNumStrips(); i++){
                int idx = new_points.getSlot(std::vector<int>(is->getStrip(i), is->getStrip(i) + num_dimensions));
                refine_weights[idx] += correction;
                max_weight = std::max(max_weight, refine_weights[idx]);
            }
        }
    }

    // compute the weights for the initial points
    std::vector<int> initial_levels =  HierarchyManipulations::computeLevels(dynamic_values->initial_points, rule.get());

    std::forward_list<NodeData> weighted_points;
    for(int i=0; i<dynamic_values->initial_points.getNumIndexes(); i++){
        std::vector<int> p(dynamic_values->initial_points.getIndex(i), dynamic_values->initial_points.getIndex(i) + num_dimensions); // write the point to vector
        weighted_points.push_front({p, {-1.0 / ((double) initial_levels[i])}});
    }
    for(int i=0; i<new_points.getNumIndexes(); i++){
        std::vector<int> p(new_points.getIndex(i), new_points.getIndex(i) + num_dimensions); // write the point to vector
        weighted_points.push_front({p, {1.0 / refine_weights[i]}});
    }

    // sort and return the sorted list
    weighted_points.sort([&](const NodeData &a, const NodeData &b)->bool{ return (a.value[0] < b.value[0]); });

    std::vector<double> x(dynamic_values->initial_points.getVector().size() + new_points.getVector().size());
    auto ix = x.begin();
    for(auto t = weighted_points.begin(); t != weighted_points.end(); t++)
        ix = std::transform(t->point.begin(), t->point.end(), ix, [&](int i)->double{ return rule->getNode(i); });
    return x;
}
std::vector<int> GridLocalPolynomial::getMultiIndex(const double x[]){
    std::vector<int> p(num_dimensions); // convert x to p, maybe expensive
    for(int j=0; j<num_dimensions; j++){
        int i = 0;
        while(std::abs(rule->getNode(i) - x[j]) > Maths::num_tol) i++;
        p[j] = i;
    }
    return p;
}
void GridLocalPolynomial::loadConstructedPoint(const double x[], const std::vector<double> &y){
    auto p = getMultiIndex(x);

    dynamic_values->initial_points.removeIndex(p);

    bool isConnected = false;
    HierarchyManipulations::touchAllImmediateRelatives(p, points, rule.get(),
                                                       [&](int)->void{ isConnected = true; });
    int lvl = rule->getLevel(p[0]);
    for(int j=1; j<num_dimensions; j++) lvl += rule->getLevel(p[j]);

    if (isConnected || (lvl == 0)){
        expandGrid(p, y);
        loadConstructedPoints();
    }else{
        dynamic_values->data.push_front({p, y});
    }
}
void GridLocalPolynomial::expandGrid(const std::vector<int> &point, const std::vector<double> &value){
    if (points.empty()){ // only one point
        points = MultiIndexSet((size_t) num_dimensions, std::vector<int>(point));
        values.resize(num_outputs, 1);
        values.setValues(std::vector<double>(value));
        surpluses.resize(num_outputs, 1);
        surpluses.getVector() = value; // the surplus of one point is the value itself
    }else{ // merge with existing points
        // compute the surplus for the point
        std::vector<double> xnode(num_dimensions), approximation(num_outputs), surp(num_outputs);
        std::transform(point.begin(), point.end(), xnode.begin(), [&](int i)->double{ return rule->getNode(i); });
        evaluate(xnode.data(), approximation.data());
        std::transform(approximation.begin(), approximation.end(), value.begin(), surp.begin(), [&](double e, double v)->double{ return v - e; });

        std::vector<int> graph = getSubGraph(point); // get the descendant nodes that must be updated later

        MultiIndexSet temp(num_dimensions, std::vector<int>(point));
        values.addValues(points, temp, value.data()); // added the value

        points.addSortedIndexes(point); // add the point
        int newindex = points.getSlot(point);
        surpluses.appendStrip(newindex, surp); // find the index of the new point

        for(auto &g : graph) if (g >= newindex) g++; // all points belowe the newindex have been shifted down by one spot

        std::vector<int> levels(points.getNumIndexes(), 0); // compute the levels, but only for the new indexes
        for(auto &g : graph){
            int const *pnt = points.getIndex(g);
            int l = rule->getLevel(pnt[0]);
            for(int j=1; j<num_dimensions; j++) l += rule->getLevel(pnt[j]);
            levels[g] = l;

            std::copy_n(values.getValues(g), num_outputs, surpluses.getStrip(g)); // reset the surpluses to the values (will be updated)
        }

        Data2D<int> dagUp = HierarchyManipulations::computeDAGup(points, rule.get());
        updateSurpluses(points, top_level + 1, levels, dagUp); // compute the current DAG and update the surplused for the descendants
    }
    buildTree(); // the tree is needed for evaluate(), must be rebuild every time the points set is updated
}
void GridLocalPolynomial::loadConstructedPoint(const double x[], int numx, const double y[]){
    Utils::Wrapper2D<const double> wrapx(num_dimensions, x);
    std::vector<std::vector<int>> pnts(numx);
    #pragma omp parallel for
    for(int i=0; i<numx; i++)
        pnts[i] = getMultiIndex(wrapx.getStrip(i));

    if (!dynamic_values->initial_points.empty()){
        Data2D<int> combined_pnts(num_dimensions, numx);
        for(int i=0; i<numx; i++)
            std::copy_n(pnts[i].begin(), num_dimensions, combined_pnts.getIStrip(i));
        dynamic_values->initial_points = dynamic_values->initial_points.diffSets(MultiIndexSet(combined_pnts));
    }

    Utils::Wrapper2D<const double> wrapy(num_outputs, y);
    for(int i=0; i<numx; i++)
        dynamic_values->data.push_front({std::move(pnts[i]), std::vector<double>(wrapy.getStrip(i), wrapy.getStrip(i) + num_outputs)});

    loadConstructedPoints();
}
void GridLocalPolynomial::loadConstructedPoints(){
    Data2D<int> candidates(num_dimensions, (int) std::distance(dynamic_values->data.begin(), dynamic_values->data.end()));
    for(struct{ int i; std::forward_list<NodeData>::iterator d; } p = {0, dynamic_values->data.begin()};
        p.d != dynamic_values->data.end(); p.i++, p.d++){
        std::copy_n(p.d->point.begin(), num_dimensions, candidates.getIStrip(p.i));
    }
    auto new_points = HierarchyManipulations::getLargestConnected(points, MultiIndexSet(candidates), rule.get());
    if (new_points.empty()) return;

    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaBasisHierarchy(); // the points will change, clear the cache
    clearCudaSurpluses();
    #endif

    auto vals = dynamic_values->extractValues(new_points);
    if (points.empty()){
        points = std::move(new_points);
        values.setValues(std::move(vals));
    }else{
        values.addValues(points, new_points, vals.data());
        points.addMultiIndexSet(new_points);
    }
    buildTree();
    recomputeSurpluses(); // costly, but the only option under the circumstances
}
void GridLocalPolynomial::finishConstruction(){ dynamic_values.reset(); }

std::vector<int> GridLocalPolynomial::getSubGraph(std::vector<int> const &point) const{
    std::vector<int> graph, p = point;
    std::vector<bool> used(points.getNumIndexes(), false);
    int max_1d_kids = rule->getMaxNumKids();
    int max_kids = max_1d_kids * num_dimensions;

    std::vector<int> monkey_count(1, 0), monkey_tail;

    while(monkey_count[0] < max_kids){
        if (monkey_count.back() < max_kids){
            int dim = monkey_count.back() / max_1d_kids;
            monkey_tail.push_back(p[dim]);
            p[dim] = rule->getKid(monkey_tail.back(), monkey_count.back() % max_1d_kids);
            int slot = points.getSlot(p);
            if ((slot == -1) || used[slot]){ // this kid is missing
                p[dim] = monkey_tail.back();
                monkey_tail.pop_back();
                monkey_count.back()++;
            }else{ // found kid, go deeper in the graph
                graph.push_back(slot);
                used[slot] = true;
                monkey_count.push_back(0);
            }
        }else{
            monkey_count.pop_back();
            int dim = monkey_count.back() / max_1d_kids;
            p[dim] = monkey_tail.back();
            monkey_tail.pop_back();
            monkey_count.back()++;
        }
    }

    return graph;
}

void GridLocalPolynomial::getInterpolationWeights(const double x[], double *weights) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;

    std::vector<int> active_points;
    std::vector<double> hbasis_values;
    std::fill_n(weights, work.getNumIndexes(), 0.0);

    // construct a sparse vector and apply transpose surplus transformation
    walkTree<1>(work, x, active_points, hbasis_values, nullptr);
    auto ibasis = hbasis_values.begin();
    for(auto i : active_points) weights[i] = *ibasis++;

    // apply the transpose of the surplus transformation
    Data2D<int> lparents;
    if (parents.getNumStrips() != work.getNumIndexes()) // if the current dag loaded in parents does not reflect the indexes in work
        lparents = HierarchyManipulations::computeDAGup(work, rule.get());

    const Data2D<int> &dagUp = (parents.getNumStrips() != work.getNumIndexes()) ? lparents : parents;

    std::vector<int> level(active_points.size());
    int active_top_level = 0;
    for(size_t i=0; i<active_points.size(); i++){
        const int *p = work.getIndex(active_points[i]);
        int current_level = rule->getLevel(p[0]);
        for(int j=1; j<num_dimensions; j++){
            current_level += rule->getLevel(p[j]);
        }
        if (active_top_level < current_level) active_top_level = current_level;
        level[i] = current_level;
    }

    std::vector<int> monkey_count(top_level+1);
    std::vector<int> monkey_tail(top_level+1);
    std::vector<bool> used(work.getNumIndexes());
    std::vector<double> node(num_dimensions);
    int max_parents = rule->getMaxNumParents() * num_dimensions;

    for(int l=active_top_level; l>0; l--){
        for(size_t i=0; i<active_points.size(); i++){
            if (level[i] == l){
                const int* p = work.getIndex(active_points[i]);
                for(int j=0; j<num_dimensions; j++) node[j] = rule->getNode(p[j]);

                std::fill(used.begin(), used.end(), false);

                monkey_count[0] = 0;
                monkey_tail[0] = active_points[i];
                int current = 0;

                while(monkey_count[0] < max_parents){
                    if (monkey_count[current] < max_parents){
                        int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                        if ((branch == -1) || used[branch]){
                            monkey_count[current]++;
                        }else{
                            const int *func = work.getIndex(branch);
                            double basis_value = rule->evalRaw(func[0], node[0]);
                            for(int j=1; j<num_dimensions; j++) basis_value *= rule->evalRaw(func[j], node[j]);
                            weights[branch] -= weights[active_points[i]] * basis_value;
                            used[branch] = true;

                            monkey_count[++current] = 0;
                            monkey_tail[current] = branch;
                        }
                    }else{
                        monkey_count[--current]++;
                    }
                }
            }
        }
    }
}

void GridLocalPolynomial::evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    Utils::Wrapper2D<double> ywrap(num_points, y);
    #pragma omp parallel for
    for(int i=0; i<num_x; i++){
        double const *this_x = xwrap.getStrip(i);
        double *this_y = ywrap.getStrip(i);
        bool dummy;
        for(int j=0; j<num_points; j++)
            this_y[j] = evalBasisSupported(work.getIndex(j), this_x, dummy);
    }
}

void GridLocalPolynomial::recomputeSurpluses(){
    int num_points = points.getNumIndexes();

    surpluses.resize(num_outputs, num_points);
    surpluses.getVector() = values.getVector(); // copy assignment

    Data2D<int> dagUp = HierarchyManipulations::computeDAGup(points, rule.get());

    std::vector<int> level = HierarchyManipulations::computeLevels(points, rule.get());

    updateSurpluses(points, top_level, level, dagUp);
}

void GridLocalPolynomial::updateSurpluses(MultiIndexSet const &work, int max_level, std::vector<int> const &level, Data2D<int> const &dagUp){
    int num_points = work.getNumIndexes();
    int max_parents = num_dimensions * rule->getMaxNumParents();

    std::vector<std::vector<int>> indexses_for_levels((size_t) max_level+1);
    for(int i=0; i<num_points; i++)
        if (level[i] > 0) indexses_for_levels[level[i]].push_back(i);

    for(int l=1; l<=max_level; l++){
        int level_size = (int) indexses_for_levels[l].size();
        #pragma omp parallel for schedule(dynamic)
        for(int s=0; s<level_size; s++){
            int i = indexses_for_levels[l][s];

            int const *p = work.getIndex(i);
            std::vector<double> x(num_dimensions);
            std::transform(p, p + num_dimensions, x.begin(), [&](int k)->double{ return rule->getNode(k); });
            double *surpi = surpluses.getStrip(i);

            std::vector<int> monkey_count(max_level + 1);
            std::vector<int> monkey_tail(max_level + 1);
            std::vector<bool> used(num_points, false);

            int current = 0;

            monkey_count[0] = 0;
            monkey_tail[0] = i;

            while(monkey_count[0] < max_parents){
                if (monkey_count[current] < max_parents){
                    int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                    if ((branch == -1) || (used[branch])){
                        monkey_count[current]++;
                    }else{
                        const double *branch_surp = surpluses.getStrip(branch);
                        double basis_value = evalBasisRaw(work.getIndex(branch), x.data());
                        for(int k=0; k<num_outputs; k++)
                            surpi[k] -= basis_value * branch_surp[k];
                        used[branch] = true;

                        monkey_count[++current] = 0;
                        monkey_tail[current] = branch;
                    }
                }else{
                    monkey_count[--current]++;
                }
            }
        }
    }
}

double GridLocalPolynomial::evalBasisRaw(const int point[], const double x[]) const{
    double f = rule->evalRaw(point[0], x[0]);
    for(int j=1; j<num_dimensions; j++) f *= rule->evalRaw(point[j], x[j]);
    return f;
}
double GridLocalPolynomial::evalBasisSupported(const int point[], const double x[], bool &isSupported) const{
    double f = rule->evalSupport(point[0], x[0], isSupported);
    if (!isSupported) return 0.0;
    for(int j=1; j<num_dimensions; j++){
        f *= rule->evalSupport(point[j], x[j], isSupported);
        if (!isSupported) return 0.0;
    }
    return f;
}

void GridLocalPolynomial::buildSpareBasisMatrix(const double x[], int num_x, int num_chunk, std::vector<int> &spntr, std::vector<int> &sindx, std::vector<double> &svals) const{
    std::vector<std::vector<int>> tindx;
    std::vector<std::vector<double>> tvals;
    std::vector<int> numnz;
    buildSparseMatrixBlockForm(x, num_x, num_chunk, numnz, tindx, tvals);

    spntr = std::vector<int>((size_t) num_x + 1);

    spntr[0] = 0;
    for(size_t i=1; i<spntr.size(); i++)
        spntr[i] += numnz[i-1] + spntr[i-1];

    sindx = std::vector<int>((size_t) spntr.back());
    svals = std::vector<double>((size_t) spntr.back());

    auto ii = sindx.begin();
    for(auto &idx : tindx) ii = std::copy(idx.begin(), idx.end(), ii);
    auto iv = svals.begin();
    for(auto &vls : tvals) iv = std::copy(vls.begin(), vls.end(), iv);
}
void GridLocalPolynomial::buildSpareBasisMatrixStatic(const double x[], int num_x, int num_chunk, int *spntr, int *sindx, double *svals) const{
    std::vector<std::vector<int>> tindx;
    std::vector<std::vector<double>> tvals;
    std::vector<int> numnz;
    buildSparseMatrixBlockForm(x, num_x, num_chunk, numnz, tindx, tvals);

    int nz = 0;
    for(int i=0; i<num_x; i++){
        spntr[i] = nz;
        nz += numnz[i];
    }
    spntr[num_x] = nz;
    size_t c = 0;
    for(auto &idx : tindx){
        std::copy(idx.begin(), idx.end(), &(sindx[c]));
        c += idx.size();
    }
    c = 0;
    for(auto &vls : tvals){
        std::copy(vls.begin(), vls.end(), &(svals[c]));
        c += vls.size();
    }
}
int GridLocalPolynomial::getSpareBasisMatrixNZ(const double x[], int num_x) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;

    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);
    std::vector<int> num_nz(num_x);

    #pragma omp parallel for
    for(int i=0; i<num_x; i++){
        std::vector<int> sindx;
        std::vector<double> svals;
        walkTree<1>(work, xwrap.getStrip(i), sindx, svals, nullptr);
        num_nz[i] = (int) sindx.size();
    }

    return std::accumulate(num_nz.begin(), num_nz.end(), 0);
}

void GridLocalPolynomial::buildSparseMatrixBlockForm(const double x[], int num_x, int num_chunk, std::vector<int> &numnz, std::vector<std::vector<int>> &tindx, std::vector<std::vector<double>> &tvals) const{
    // numnz will be resized to (num_x + 1) with the last entry set to a dummy zero
    numnz.resize((size_t) num_x);
    int num_blocks = num_x / num_chunk + ((num_x % num_chunk != 0) ? 1 : 0);

    tindx.resize(num_blocks);
    tvals.resize(num_blocks);

    const MultiIndexSet &work = (points.empty()) ? needed : points;
    Utils::Wrapper2D<double const> xwrap(num_dimensions, x);

    #pragma omp parallel for
    for(int b=0; b<num_blocks; b++){
        tindx[b].resize(0);
        tvals[b].resize(0);
        int chunk_size = (b < num_blocks - 1) ? num_chunk : (num_x - (num_blocks - 1) * num_chunk);
        for(int i = b * num_chunk; i < b * num_chunk + chunk_size; i++){
            numnz[i] = (int) tindx[b].size();
            walkTree<1>(work, xwrap.getStrip(i), tindx[b], tvals[b], nullptr);
            numnz[i] = (int) tindx[b].size() - numnz[i];
        }
    }
}

void GridLocalPolynomial::buildTree(){
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    int num_points = work.getNumIndexes();

    std::vector<int> level = HierarchyManipulations::computeLevels(work, rule.get());
    top_level = *std::max_element(level.begin(), level.end());

    Data2D<int> kids = HierarchyManipulations::computeDAGDown(work, rule.get());

    int max_kids = (int) kids.getStride();

    Data2D<int> tree(max_kids, num_points, -1);
    std::vector<bool> free(num_points, true);

    std::vector<int> monkey_count((size_t) (top_level + 1));
    std::vector<int> monkey_tail(monkey_count.size());

    roots = std::vector<int>();
    int next_root = 0; // zero is always a root and is included at index 0

    while(next_root != -1){
        roots.push_back(next_root);
        free[next_root] = false;

        monkey_tail[0] = next_root;
        monkey_count[0] = 0;
        size_t current = 0;

        while(monkey_count[0] < max_kids){
            if (monkey_count[current] < max_kids){
                int kid = kids.getStrip(monkey_tail[current])[monkey_count[current]];
                if ((kid == -1) || (!free[kid])){
                    monkey_count[current]++; // no kid, keep counting
                }else{
                    tree.getStrip(monkey_tail[current])[monkey_count[current]] = kid;
                    monkey_count[++current] = 0;
                    monkey_tail[current] = kid;
                    free[kid] = false;
                }
            }else{
                monkey_count[--current]++; // done with all kids here
            }
        }

        next_root = -1;
        int next_level = top_level + 1;
        for(int i=0; i<num_points; i++){
            if (free[i] && (level[i] < next_level)){
                next_root = i;
                next_level = level[i];
            }
        }
    }

    pntr = std::vector<int>((size_t) (num_points + 1), 0);
    for(int i=0; i<num_points; i++)
        pntr[i+1] = pntr[i] + (int) std::count_if(tree.getIStrip(i), tree.getIStrip(i) + max_kids, [](int k)->bool{ return (k > -1); });

    indx = std::vector<int>((size_t) ((pntr[num_points] > 0) ? pntr[num_points] : 1));
    std::copy_if(tree.getVector().begin(), tree.getVector().end(), indx.begin(), [](int t)->bool{ return (t > -1); });
}

void GridLocalPolynomial::integrateHierarchicalFunctions(double integrals[]) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;

    int n = 0;
    std::vector<double> w, x;

    if ((rule->getMaxOrder() == -1) || (rule->getMaxOrder() > 3) ){
        n = ((rule->getMaxOrder() == -1) ? top_level : rule->getMaxOrder()) / 2 + 1;
        OneDimensionalNodes::getGaussLegendre(n, w, x);
    }

    for(int i=0; i<work.getNumIndexes(); i++){
        const int* p = work.getIndex(i);
        integrals[i] = rule->getArea(p[0], n, w.data(), x.data());
        for(int j=1; j<num_dimensions; j++){
            integrals[i] *= rule->getArea(p[j], n, w.data(), x.data());
        }
    }
}
void GridLocalPolynomial::getQuadratureWeights(double *weights) const{
    const MultiIndexSet &work = (points.empty()) ? needed : points;
    integrateHierarchicalFunctions(weights);

    std::vector<int> monkey_count(top_level+1);
    std::vector<int> monkey_tail(top_level+1);

    double basis_value;

    Data2D<int> lparents;
    if (parents.getNumStrips() != work.getNumIndexes())
        lparents = HierarchyManipulations::computeDAGup(work, rule.get());

    const Data2D<int> &dagUp = (parents.getNumStrips() != work.getNumIndexes()) ? lparents : parents;

    int num_points = work.getNumIndexes();
    std::vector<int> level = HierarchyManipulations::computeLevels(work, rule.get());

    std::vector<double> node(num_dimensions);
    int max_parents = rule->getMaxNumParents() * num_dimensions;

    for(int l=top_level; l>0; l--){
        for(int i=0; i<num_points; i++){
            if (level[i] == l){
                const int* p = work.getIndex(i);
                for(int j=0; j<num_dimensions; j++) node[j] = rule->getNode(p[j]);

                std::vector<bool> used(work.getNumIndexes(), false);

                monkey_count[0] = 0;
                monkey_tail[0] = i;
                int current = 0;

                while(monkey_count[0] < max_parents){
                    if (monkey_count[current] < max_parents){
                        int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                        if ((branch == -1) || used[branch]){
                            monkey_count[current]++;
                        }else{
                            const int *func = work.getIndex(branch);
                            basis_value = rule->evalRaw(func[0], node[0]);
                            for(int j=1; j<num_dimensions; j++) basis_value *= rule->evalRaw(func[j], node[j]);
                            weights[branch] -= weights[i] * basis_value;
                            used[branch] = true;

                            monkey_count[++current] = 0;
                            monkey_tail[current] = branch;
                        }
                    }else{
                        monkey_count[--current]++;
                    }
                }
            }
        }
    }
}

void GridLocalPolynomial::integrate(double q[], double *conformal_correction) const{
    int num_points = points.getNumIndexes();
    std::fill(q, q + num_outputs, 0.0);

    if (conformal_correction == 0){
        std::vector<double> integrals(num_points);
        integrateHierarchicalFunctions(integrals.data());
        for(int i=0; i<num_points; i++){
            const double *s = surpluses.getStrip(i);
            double wi = integrals[i];
            for(int k=0; k<num_outputs; k++) q[k] += wi * s[k];
        }
    }else{
        std::vector<double> w(num_points);
        getQuadratureWeights(w.data());
        for(int i=0; i<num_points; i++){
            double wi = w[i] * conformal_correction[i];
            const double *vals = values.getValues(i);
            for(int k=0; k<num_outputs; k++) q[k] += wi * vals[k];
        }
    }
}

std::vector<double> GridLocalPolynomial::getNormalization() const{
    std::vector<double> norms(num_outputs);
    std::fill(norms.begin(), norms.end(), 0.0);
    for(int i=0; i<points.getNumIndexes(); i++){
        const double *v = values.getValues(i);
        for(int j=0; j<num_outputs; j++){
            if (norms[j] < std::abs(v[j])) norms[j] = std::abs(v[j]);
        }
    }
    return norms;
}

Data2D<int> GridLocalPolynomial::buildUpdateMap(double tolerance, TypeRefinement criteria, int output, const double *scale_correction) const{
    int num_points = points.getNumIndexes();
    Data2D<int> map2(num_dimensions, num_points);
    if (tolerance == 0.0){
        map2.fill(1); // if tolerance is 0, refine everything
        return map2;
    }else{
        map2.fill(0);
    }

    std::vector<double> norm = getNormalization();

    int active_outputs = (output == -1) ? num_outputs : 1;
    Utils::Wrapper2D<double const> scale(active_outputs, scale_correction);
    std::vector<double> default_scale;
    if (scale_correction == nullptr){
        default_scale = std::vector<double>(Utils::size_mult(active_outputs, num_points), 1.0);
        scale = Utils::Wrapper2D<double const>(active_outputs, default_scale.data());
    }

    if ((criteria == refine_classic) || (criteria == refine_parents_first)){
        #pragma omp parallel for
        for(int i=0; i<num_points; i++){
            bool small = true;
            const double *s = surpluses.getStrip(i);
            const double *c = scale.getStrip(i);
            if (output == -1){
                for(int k=0; k<num_outputs; k++) small = small && ((c[k] * std::abs(s[k]) / norm[k]) <= tolerance);
            }else{
                small = ((c[0] * std::abs(s[output]) / norm[output]) <= tolerance);
            }
            if (!small){
                int *m = map2.getStrip(i);
                std::fill(m, m + num_dimensions, 1);
            }
        }
    }else{
        // construct a series of 1D interpolants and use a refinement criteria that is a combination of the two hierarchical coefficients
        Data2D<int> dagUp = HierarchyManipulations::computeDAGup(points, rule.get());

        int max_1D_parents = rule->getMaxNumParents();

        HierarchyManipulations::SplitDirections split(points);

        #pragma omp parallel for
        for(int j=0; j<split.getNumJobs(); j++){ // split.getNumJobs() gives the number of 1D interpolants to construct
            int d = split.getJobDirection(j);
            int nump = split.getJobNumPoints(j);
            const int *pnts = split.getJobPoints(j);

            std::vector<int> global_to_pnts(num_points);
            std::vector<int> levels(nump);

            int max_level = 0;

            Data2D<double> vals;
            vals.resize(active_outputs, nump);

            for(int i=0; i<nump; i++){
                const double* v = values.getValues(pnts[i]);
                const int *p = points.getIndex(pnts[i]);
                if (output == -1){
                    std::copy(v, v + num_outputs, vals.getStrip(i));
                }else{
                    vals.getStrip(i)[0] = v[output];
                }
                global_to_pnts[pnts[i]] = i;
                levels[i] = rule->getLevel(p[d]);
                if (max_level < levels[i]) max_level = levels[i];
            }

            std::vector<int> monkey_count(max_level + 1);
            std::vector<int> monkey_tail(max_level + 1);

            for(int l=1; l<=max_level; l++){
                for(int i=0; i<nump; i++){
                    if (levels[i] == l){
                        const int *p = points.getIndex(pnts[i]);
                        double x = rule->getNode(p[d]);
                        double *valsi = vals.getStrip(i);

                        int current = 0;
                        monkey_count[0] = d * max_1D_parents;
                        monkey_tail[0] = pnts[i]; // uses the global indexes
                        std::vector<bool> used(nump, false);

                        while(monkey_count[0] < (d+1) * max_1D_parents){
                            if (monkey_count[current] < (d+1) * max_1D_parents){
                                int branch = dagUp.getStrip(monkey_tail[current])[monkey_count[current]];
                                if ((branch == -1) || (used[global_to_pnts[branch]])){
                                    monkey_count[current]++;
                                }else{
                                    const int *branch_point = points.getIndex(branch);
                                    double basis_value = rule->evalRaw(branch_point[d], x);
                                    const double *branch_vals = vals.getStrip(global_to_pnts[branch]);
                                    for(int k=0; k<active_outputs; k++) valsi[k] -= basis_value * branch_vals[k];

                                    used[global_to_pnts[branch]] = true;
                                    monkey_count[++current] = d * max_1D_parents;
                                    monkey_tail[current] = branch;
                                }
                            }else{
                                monkey_count[--current]++;
                            }
                        }
                    }
                }
            }

            // at this point, vals contains the one directional surpluses
            for(int i=0; i<nump; i++){
                const double *s = surpluses.getStrip(pnts[i]);
                const double *c = scale.getStrip(pnts[i]);
                const double *v = vals.getStrip(i);
                bool small = true;
                if (output == -1){
                    for(int k=0; k<num_outputs; k++){
                        small = small && (((c[k] * std::abs(s[k]) / norm[k]) <= tolerance) || ((c[k] * std::abs(v[k]) / norm[k]) <= tolerance));
                    }
                }else{
                    small = ((c[0] * std::abs(s[output]) / norm[output]) <= tolerance) || ((c[0] * std::abs(v[0]) / norm[output]) <= tolerance);
                }
                map2.getStrip(pnts[i])[d] = (small) ? 0 : 1;;
            }
        }
    }
    return map2;
}
MultiIndexSet GridLocalPolynomial::getRefinementCanidates(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits, const double *scale_correction) const{
    Data2D<int> pmap = buildUpdateMap(tolerance, criteria, output, scale_correction);

    bool useParents = (criteria == refine_fds) || (criteria == refine_parents_first);

    Data2D<int> refined(num_dimensions, 0);

    int num_points = points.getNumIndexes();

    if (level_limits.empty()){
        for(int i=0; i<num_points; i++){
            const int *map = pmap.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (map[j] == 1){ // if this dimension needs to be refined
                    if (!(useParents && addParent(points.getIndex(i), j, points, refined))){
                        addChild(points.getIndex(i), j, points, refined);
                    }
                }
            }
        }
    }else{
        for(int i=0; i<num_points; i++){
            const int *map = pmap.getStrip(i);
            for(int j=0; j<num_dimensions; j++){
                if (map[j] == 1){ // if this dimension needs to be refined
                    if (!(useParents && addParent(points.getIndex(i), j, points, refined))){
                        addChildLimited(points.getIndex(i), j, points, level_limits, refined);
                    }
                }
            }
        }
    }

    MultiIndexSet result = MultiIndexSet(refined);
    if (criteria == refine_stable)
        HierarchyManipulations::completeToLower(points, result, rule.get());
    return result;
}

bool GridLocalPolynomial::addParent(const int point[], int direction, const MultiIndexSet &exclude, Data2D<int> &destination) const{
    std::vector<int> dad(num_dimensions);
    std::copy_n(point, num_dimensions, dad.data());
    bool added = false;
    dad[direction] = rule->getParent(point[direction]);
    if ((dad[direction] != -1) && exclude.missing(dad)){
        destination.appendStrip(dad);
        added = true;
    }
    dad[direction] = rule->getStepParent(point[direction]);
    if ((dad[direction] != -1) && exclude.missing(dad)){
        destination.appendStrip(dad);
        added = true;
    }
    return added;
}
void GridLocalPolynomial::addChild(const int point[], int direction, const MultiIndexSet &exclude, Data2D<int> &destination) const{
    std::vector<int> kid(num_dimensions);
    std::copy_n(point, num_dimensions, kid.data());
    int max_1d_kids = rule->getMaxNumKids();
    for(int i=0; i<max_1d_kids; i++){
        kid[direction] = rule->getKid(point[direction], i);
        if ((kid[direction] != -1) && exclude.missing(kid)){
            destination.appendStrip(kid);
        }
    }
}
void GridLocalPolynomial::addChildLimited(const int point[], int direction, const MultiIndexSet &exclude, const std::vector<int> &level_limits, Data2D<int> &destination) const{
    std::vector<int> kid(num_dimensions);
    std::copy_n(point, num_dimensions, kid.data());
    int max_1d_kids = rule->getMaxNumKids();
    for(int i=0; i<max_1d_kids; i++){
        kid[direction] = rule->getKid(point[direction], i);
        if ((kid[direction] != -1)
            && ((level_limits[direction] == -1) || (rule->getLevel(kid[direction]) <= level_limits[direction]))
            && exclude.missing(kid)){
            destination.appendStrip(kid);
        }
    }
}

void GridLocalPolynomial::clearRefinement(){ needed = MultiIndexSet(); }
const double* GridLocalPolynomial::getSurpluses() const{
    return surpluses.getVector().data();
}
const int* GridLocalPolynomial::getNeededIndexes() const{
    return (needed.empty()) ? 0 : needed.getIndex(0);
}
void GridLocalPolynomial::setSurplusRefinement(double tolerance, TypeRefinement criteria, int output, const std::vector<int> &level_limits, const double *scale_correction){
    clearRefinement();

    needed = getRefinementCanidates(tolerance, criteria, output, level_limits, scale_correction);
}
int GridLocalPolynomial::removePointsByHierarchicalCoefficient(double tolerance, int output, const double *scale_correction){
    clearRefinement();
    int num_points = points.getNumIndexes();
    std::vector<bool> pmap(num_points); // point map, set to true if the point is to be kept, false otherwise

    std::vector<double> norm = getNormalization();

    int active_outputs = (output == -1) ? num_outputs : 1;
    Utils::Wrapper2D<double const> scale(active_outputs, scale_correction);
    std::vector<double> default_scale;
    if (scale_correction == nullptr){
        default_scale = std::vector<double>(Utils::size_mult(active_outputs, num_points), 1.0);
        scale = Utils::Wrapper2D<double const>(active_outputs, default_scale.data());
    }

    for(int i=0; i<num_points; i++){
        bool small = true;
        const double *s = surpluses.getStrip(i);
        const double *c = scale.getStrip(i);
        if (output == -1){
            for(int k=0; k<num_outputs; k++) small = small && ((c[k] * std::abs(s[k]) / norm[k]) <= tolerance);
        }else{
            small = ((c[0] * std::abs(s[output]) / norm[output]) <= tolerance);
        }
        pmap[i] = !small;
    }

    int num_kept = 0; for(int i=0; i<num_points; i++) num_kept += (pmap[i]) ? 1 : 0;

    if (num_kept == num_points) return num_points; // trivial case, remove nothing
    clearAccelerationData();

    // save a copy of the points and the values
    Data2D<int> point_kept(num_dimensions, num_kept);

    StorageSet values_kept;
    values_kept.resize(num_outputs, num_kept);
    values_kept.getVector().resize(Utils::size_mult(num_kept, num_outputs));

    Data2D<double> surpluses_kept(num_outputs, num_kept);

    num_kept = 0;
    for(int i=0; i<num_points; i++){
        if (pmap[i]){
            std::copy_n(points.getIndex(i), num_dimensions, point_kept.getStrip(num_kept));
            std::copy_n(values.getValues(i), num_outputs, values_kept.getValues(num_kept));
            std::copy_n(surpluses.getStrip(i), num_outputs, surpluses_kept.getStrip(num_kept));
            num_kept++;
        }
    }

    int dims = num_dimensions, outs = num_outputs;

    reset(false);
    num_dimensions = dims;
    num_outputs = outs;
    if (num_kept == 0) return 0; // trivial case, remove all

    points = MultiIndexSet(point_kept);

    values = std::move(values_kept);

    surpluses = std::move(surpluses_kept);

    buildTree();

    return points.getNumIndexes();
}

void GridLocalPolynomial::setHierarchicalCoefficients(const double c[], TypeAcceleration){
    #ifdef Tasmanian_ENABLE_CUDA
    clearCudaSurpluses();
    #endif
    if (points.empty()){
        points = std::move(needed);
        needed = MultiIndexSet();
    }else{
        clearRefinement();
    }
    surpluses.resize(num_outputs, getNumPoints());
    std::copy_n(c, surpluses.getTotalEntries(), surpluses.getVector().data());

    std::vector<double> &vals = values.getVector();
    vals.resize(surpluses.getTotalEntries());

    std::vector<double> x(((size_t) getNumPoints()) * ((size_t) num_dimensions));
    getPoints(x.data());
    evaluateBatch(x.data(), points.getNumIndexes(), vals.data());
}

void GridLocalPolynomial::clearAccelerationData(){
    #ifdef Tasmanian_ENABLE_CUDA
    cuda_cache.reset();
    cuda_cachef.reset();
    #endif
}

void GridLocalPolynomial::setFavorSparse(bool favor){
    // sparse_affinity == -1: use dense algorithms
    // sparse_affinity ==  1: use sparse algorithms
    // sparse_affinity ==  0: let Tasmanian decide
    // favor true/false bumps you upper or lower on the scale
    if (favor && (sparse_affinity < 1)) sparse_affinity++;
    if (!favor && (sparse_affinity > -1)) sparse_affinity--;
}

}

#endif
